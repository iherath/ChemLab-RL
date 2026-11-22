"""Synthetic, auditable liquid bookkeeping for the solution-prep subtask.

Liquid is NOT simulated by MuJoCo. The env measures geometry every control
step (plunger slide position, needle-tip location, vial tilt, cap weld,
agitation from vial accelerations) and feeds it to :class:`SolutionState`,
which moves volume between finite compartments:

    reservoir <-> syringe barrel -> vial | spilled

Volume is conserved exactly (``conservation_error_mL`` ~ 0) and solute mass is
conserved exactly (``mass_conservation_error_g`` ~ 0). Dissolution rate ``k``,
saturation and mixing rates are a synthetic family, not lab measurements.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

TOTAL_VOLUME_ML = 3.0   # target solvent volume in the receiving vial
VOLUME_TOL_ML = 0.05
SPILL_TOL_ML = 0.05

VIAL_INNER_RADIUS_M = 0.011
VIAL_INNER_HEIGHT_M = 0.050
POUR_RATE_ML_S = 5.0          # outflow at full excess tilt past the lip angle
MIX_RATE_PER_AGITATION = 1.0  # 1/s of h_mix relaxation per unit agitation
MIX_RATE_BASE = 0.002         # 1/s passive diffusion (~30 min unagitated)
MAX_AGITATION = 3.0


@dataclass(frozen=True)
class Syringe:
    """Cylindrical barrel; capacity == pi * r^2 * stroke (SI -> mL via 1e6)."""

    name: str
    capacity_mL: float
    barrel_inner_radius_m: float
    stroke_m: float

    @property
    def mL_per_m(self) -> float:
        """Barrel volume per metre of plunger travel."""
        return self.capacity_mL / self.stroke_m

    def as_dict(self) -> dict:
        return asdict(self)


def _syringe(name: str, capacity_mL: float, inner_diameter_mm: float) -> Syringe:
    r = inner_diameter_mm * 1e-3 / 2.0
    stroke = capacity_mL * 1e-6 / (math.pi * r * r)
    return Syringe(name, capacity_mL, r, stroke)


# BD-style inner diameters; stroke derived so the barrel geometry is consistent.
SYRINGES: dict[str, Syringe] = {
    "1mL": _syringe("1mL", 1.0, 4.78),
    "2mL": _syringe("2mL", 2.0, 8.90),
    "3mL": _syringe("3mL", 3.0, 8.66),
    "5mL": _syringe("5mL", 5.0, 12.06),
}
HELD_OUT_SYRINGE = "2mL"


@dataclass(frozen=True)
class Solid:
    """Hidden dissolution parameters: dm/dt = -k (a0 + a) m (1 - C/C_sat)."""

    dissolution_k: float      # 1/s per unit agitation
    base_agitation: float     # a0: convection/diffusion with no swirling
    c_sat_g_per_mL: float

    def as_dict(self) -> dict:
        return asdict(self)


# Agitation 1, 0.05-0.3 g in 3 mL -> <1% undissolved in ~7-30 s; agitation 0 -> >5 min.
SOLIDS: list[Solid] = [
    Solid(0.20, 0.020, 0.60),
    Solid(0.28, 0.020, 0.50),
    Solid(0.38, 0.020, 0.80),
    Solid(0.48, 0.020, 0.55),
    Solid(0.60, 0.020, 0.70),
    Solid(0.42, 0.020, 0.65),  # held out: k strictly inside the training range
]
HELD_OUT_SOLID = 5


def spill_tilt_threshold(vial_mL: float) -> float:
    """Tilt (0 upright, 1 inverted) at which the free surface reaches the lip.

    Pivot-about-centre approximation for a cylinder: the surface touches the
    rim when h + R tan(theta) >= H, i.e. theta* = atan((H - h) / R).
    """
    h = vial_mL * 1e-6 / (math.pi * VIAL_INNER_RADIUS_M ** 2)
    theta = math.atan2(max(0.0, VIAL_INNER_HEIGHT_M - h), VIAL_INNER_RADIUS_M)
    return theta / math.pi


@dataclass
class SolutionState:
    """Finite-volume accounting for syringe draw/dispense, spill, dissolution, mixing.

    ``h_mix`` (alias ``homogeneity``) is a stratification index of the liquid
    already in the vial: fresh solvent lowers it by its volume fraction and
    agitation relaxes it toward 1. ``mixed_fraction`` additionally requires the
    solid to be dissolved. ``capped`` is mirrored from the env's cap weld.
    """

    syringe: Syringe
    solid: Solid
    reservoir_mL: float = 5.0
    syringe_liquid_mL: float = 0.0
    syringe_air_mL: float = 0.0
    vial_mL: float = 0.0
    spilled_mL: float = 0.0
    undissolved_g: float = 0.15
    dissolved_g: float = 0.0
    spilled_solute_g: float = 0.0
    homogeneity: float = 1.0
    capped: bool = False
    initial_total_mL: float = field(init=False)
    initial_solid_g: float = field(init=False)

    def __post_init__(self) -> None:
        self.initial_total_mL = self.liquid_total_mL
        self.initial_solid_g = self.undissolved_g + self.dissolved_g + self.spilled_solute_g

    # -- step -----------------------------------------------------------------
    def step(self, dt: float, *, plunger_mL: float, tip_in_reservoir_liquid: bool,
             tip_in_vial_mouth: bool, vial_tilt: float, vial_capped: bool,
             agitation: float, tip_in_reservoir_mouth: bool = False) -> None:
        """``tip_in_reservoir_mouth``: tip inside the reservoir below its rim (drops
        expelled there fall back into it); drawing still needs a submerged tip."""
        self.capped = bool(vial_capped)
        self._move_plunger(plunger_mL, tip_in_reservoir_liquid, tip_in_vial_mouth,
                           tip_in_reservoir_liquid or tip_in_reservoir_mouth)
        self._spill(dt, float(np.clip(vial_tilt, 0.0, 1.0)))
        a = float(np.clip(agitation, 0.0, MAX_AGITATION))
        self._dissolve(dt, a)
        rate = MIX_RATE_BASE + MIX_RATE_PER_AGITATION * a
        if self.vial_mL > 1e-9:
            self.homogeneity = 1.0 - (1.0 - self.homogeneity) * math.exp(-rate * dt)

    def _move_plunger(self, plunger_mL: float, in_reservoir: bool, in_vial: bool,
                      over_reservoir: bool | None = None) -> None:
        over_reservoir = in_reservoir if over_reservoir is None else over_reservoir
        target = float(np.clip(plunger_mL, 0.0, self.syringe.capacity_mL))
        d = target - (self.syringe_liquid_mL + self.syringe_air_mL)
        if d > 0.0:  # draw
            liquid = min(d, self.reservoir_mL) if in_reservoir else 0.0
            self.reservoir_mL -= liquid
            self.syringe_liquid_mL += liquid
            self.syringe_air_mL += d - liquid
        elif d < 0.0:  # expel air first, then liquid
            out = -d
            air = min(out, self.syringe_air_mL)
            self.syringe_air_mL -= air
            liquid = min(out - air, self.syringe_liquid_mL)
            self.syringe_liquid_mL -= liquid
            if liquid > 0.0:
                if in_vial and not self.capped:
                    self._add_to_vial(liquid)
                elif over_reservoir:
                    self.reservoir_mL += liquid  # pushed/dripped back into the source
                else:
                    self.spilled_mL += liquid
        # Re-anchor the barrel to the plunger to kill float drift.
        self.syringe_air_mL = max(0.0, target - self.syringe_liquid_mL)

    def _add_to_vial(self, mL: float) -> None:
        total = self.vial_mL + mL
        self.homogeneity *= self.vial_mL / total  # fresh, unmixed layer
        self.vial_mL = total

    def _spill(self, dt: float, tilt: float) -> None:
        if self.capped or self.vial_mL <= 0.0:
            return
        thr = spill_tilt_threshold(self.vial_mL)
        if tilt <= thr:
            return
        out = min(self.vial_mL, POUR_RATE_ML_S * (tilt - thr) / max(1e-6, 1.0 - thr) * dt)
        solute = self.dissolved_g * out / self.vial_mL  # undissolved solid stays put
        self.vial_mL -= out
        self.spilled_mL += out
        self.dissolved_g -= solute
        self.spilled_solute_g += solute

    def _dissolve(self, dt: float, agitation: float) -> None:
        s = self.solid
        if self.vial_mL < 1e-6 or self.undissolved_g <= 0.0:
            return
        room = s.c_sat_g_per_mL * self.vial_mL - self.dissolved_g
        if room <= 0.0:
            return
        c = self.dissolved_g / self.vial_mL
        rate = s.dissolution_k * (s.base_agitation + agitation) * max(0.0, 1.0 - c / s.c_sat_g_per_mL)
        dm = min(self.undissolved_g * (1.0 - math.exp(-rate * dt)), room, self.undissolved_g)
        self.undissolved_g -= dm
        self.dissolved_g += dm

    # -- derived quantities ---------------------------------------------------
    @property
    def h_mix(self) -> float:
        return self.homogeneity

    @property
    def liquid_total_mL(self) -> float:
        return self.reservoir_mL + self.syringe_liquid_mL + self.vial_mL + self.spilled_mL

    @property
    def conservation_error_mL(self) -> float:
        return self.liquid_total_mL - self.initial_total_mL

    @property
    def mass_conservation_error_g(self) -> float:
        return self.undissolved_g + self.dissolved_g + self.spilled_solute_g - self.initial_solid_g

    @property
    def undissolved_fraction(self) -> float:
        total = self.undissolved_g + self.dissolved_g
        return self.undissolved_g / total if total > 0.0 else 0.0

    @property
    def mixed_fraction(self) -> float:
        return self.h_mix * (1.0 - self.undissolved_fraction)

    @property
    def homogeneous(self) -> bool:
        return (self.undissolved_g < 0.01 * self.initial_solid_g + 1e-12
                and self.h_mix >= 0.98
                and abs(self.vial_mL - TOTAL_VOLUME_ML) <= VOLUME_TOL_ML)

    @property
    def turbidity(self) -> float:
        """Render cue in [0, 1]: suspended solid plus schlieren from poor mixing."""
        if self.vial_mL < 1e-6:
            return 0.0
        u = self.undissolved_fraction
        return float(np.clip(1.0 - (1.0 - u) * (1.0 - 0.5 * (1.0 - self.h_mix)), 0.0, 1.0))

    def fill_height_m(self, vial_inner_radius: float = VIAL_INNER_RADIUS_M) -> float:
        return self.vial_mL * 1e-6 / (math.pi * vial_inner_radius ** 2)

    def as_dict(self) -> dict:
        return {
            "syringe": self.syringe.as_dict(), "solid": self.solid.as_dict(),
            "reservoir_mL": self.reservoir_mL, "syringe_liquid_mL": self.syringe_liquid_mL,
            "syringe_air_mL": self.syringe_air_mL, "vial_mL": self.vial_mL,
            "spilled_mL": self.spilled_mL, "undissolved_g": self.undissolved_g,
            "dissolved_g": self.dissolved_g, "spilled_solute_g": self.spilled_solute_g,
            "homogeneity": self.homogeneity, "mixed_fraction": self.mixed_fraction,
            "turbidity": self.turbidity, "capped": self.capped,
            "homogeneous": self.homogeneous,
            "conservation_error_mL": self.conservation_error_mL,
            "mass_conservation_error_g": self.mass_conservation_error_g,
        }
