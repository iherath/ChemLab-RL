"""Synthetic, interpretable coarse-grain material parameters for MuJoCo.

These are simulator parameters, not measured real-powder identities. Unlike the
legacy flow/onset vector, none prescribes a transferred mass or dosing rate.
Each grain's mass is its geometric sphere volume times its material density.
Scale lag/noise are instrument settings and are deliberately separate.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class GranularMaterial:
    density_kg_m3: float = 20e-6 / (4 / 3 * math.pi * 0.0015 ** 3)
    radius_m: float = 0.0015
    sliding_friction: float = 0.5
    torsional_friction_m: float = 0.00002
    rolling_friction_m: float = 0.00002
    adhesion_n: float = 0.0
    contact_time_constant_s: float = 0.0005
    contact_damping_ratio: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Material parameters must be finite")
        if self.density_kg_m3 <= 0 or self.radius_m <= 0:
            raise ValueError("Density and radius must be positive")
        if min(self.sliding_friction, self.torsional_friction_m, self.rolling_friction_m, self.adhesion_n) < 0:
            raise ValueError("Friction and adhesion must be nonnegative")
        if self.contact_time_constant_s <= 0 or self.contact_damping_ratio <= 0:
            raise ValueError("Contact solver parameters must be positive")

    @property
    def grain_volume_m3(self):
        return 4 / 3 * math.pi * self.radius_m ** 3

    @property
    def grain_mass_kg(self):
        return self.density_kg_m3 * self.grain_volume_m3

    @property
    def grain_mass_g(self):
        return 1000 * self.grain_mass_kg

    def as_dict(self):
        return {k: float(v) for k, v in asdict(self).items()}

    def geom_properties(self):
        """Native MuJoCo 3.16 sphere/contact attributes; no transfer callback."""
        return dict(size=[self.radius_m], mass=self.grain_mass_kg,
                    friction=[self.sliding_friction, self.torsional_friction_m, self.rolling_friction_m],
                    condim=3, priority=1, adhesion=self.adhesion_n,
                    solref=[self.contact_time_constant_s, self.contact_damping_ratio],
                    solimp=[0.95, 0.99, 0.0001, 0.5, 2], contype=1, conaffinity=1)


# Initial experiment varies density and sliding friction. Radius and solver
# settings are fixed. With condim=3, torsional/rolling coefficients are inactive
# and must not be described as randomized material properties.
BASE_DENSITY = GranularMaterial().density_kg_m3
GRANULAR_MATERIALS = (
    GranularMaterial(),
    GranularMaterial(density_kg_m3=0.85 * BASE_DENSITY, sliding_friction=0.35),
    GranularMaterial(density_kg_m3=1.15 * BASE_DENSITY, sliding_friction=0.35),
    GranularMaterial(density_kg_m3=0.85 * BASE_DENSITY, sliding_friction=0.65),
    GranularMaterial(density_kg_m3=1.15 * BASE_DENSITY, sliding_friction=0.65),
    GranularMaterial(density_kg_m3=1.05 * BASE_DENSITY, sliding_friction=0.55),
)
HELD_OUT_MATERIAL_IDS = (5,)


@dataclass(frozen=True)
class ScaleInstrument:
    lag_s: float = 0.25
    noise_g: float = 0.005
