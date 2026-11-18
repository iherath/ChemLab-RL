"""Mass-accounting powder dispenser for behavior-cloning demonstrations.

The pan reading is the mass that landed in the vessel, passed through a lag
and noise. It is not a contact-force sensor and not a particle simulation.
Material parameters are synthetic. They are stored on each demo so a later
split can hold out parameter combinations. They are not part of the
observation: the scripted demonstrator may use them, the cloned policy sees
the scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DT = 0.1
HORIZON = 250
TOL_G = 0.05
OBS_NAMES = (
    "tilt",
    "reading_g",
    "target_g",
    "error_g",
    "d_reading_g",
    "reading_hist_0",
    "reading_hist_1",
    "reading_hist_2",
    "reading_hist_3",
)


@dataclass(frozen=True)
class Material:
    flow_g_per_s: float
    onset: float
    delay_s: float
    residual_g: float
    lag_s: float
    noise_g: float
    spill_tilt: float
    cohesion: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def sample_material(rng: np.random.Generator, target_g: float) -> Material:
    """Draw one synthetic powder. Ranges are a family, not lab measurements."""
    return Material(
        flow_g_per_s=float(rng.uniform(0.35, 2.4)),
        onset=float(rng.uniform(0.12, 0.42)),
        delay_s=float(rng.uniform(0.1, 0.7)),
        residual_g=float(rng.uniform(0.0, min(0.35, 0.25 * target_g))),
        lag_s=float(rng.uniform(0.15, 0.7)),
        noise_g=float(rng.uniform(0.004, 0.02)),
        spill_tilt=float(rng.uniform(0.78, 0.95)),
        cohesion=float(rng.uniform(0.0, 0.45)),
    )


class PowderDispense:
    """Tilt a hopper over a pan until the lagged reading is inside tolerance."""

    def __init__(self, material: Material, target_g: float, rng: np.random.Generator):
        self.material = material
        self.target_g = float(target_g)
        self.rng = rng
        self.tilt = 0.0
        self.mass_g = 0.0
        self.spilled_g = 0.0
        self.reading_g = 0.0
        self._in_spout = 0.0
        self._above_for = 0.0
        self._hist = [0.0, 0.0, 0.0, 0.0]
        self.t = 0

    def obs(self, last_action: float = 0.0) -> np.ndarray:
        del last_action
        d_reading = self.reading_g - self._hist[-1]
        return np.array(
            [
                self.tilt,
                self.reading_g,
                self.target_g,
                self.reading_g - self.target_g,
                d_reading,
                *self._hist,
            ],
            dtype=np.float32,
        )

    def step(self, action: float) -> tuple[np.ndarray, dict]:
        action = float(np.clip(action, -1.0, 1.0))
        self.tilt = float(np.clip(self.tilt + 0.08 * action, 0.0, 1.0))
        m = self.material
        if self.tilt > m.onset:
            self._above_for += DT
        else:
            self._above_for = 0.0
        delivered = 0.0
        if self._above_for >= m.delay_s:
            rate = m.flow_g_per_s * (1.0 - m.cohesion) * (self.tilt - m.onset) / max(1e-6, 1.0 - m.onset)
            delivered = max(0.0, rate) * DT
            # Extra that will fall after the tilt drops back through onset.
            self._in_spout = m.residual_g * self.tilt
        elif self._in_spout > 0.0:
            dump = min(self._in_spout, 0.5 * self._in_spout + 0.02)
            delivered = dump
            self._in_spout -= dump
        if self.tilt > m.spill_tilt and delivered > 0.0:
            miss = (self.tilt - m.spill_tilt) / max(1e-6, 1.0 - m.spill_tilt)
            self.spilled_g += delivered * miss
            delivered *= 1.0 - miss
        self.mass_g += delivered
        lag = max(m.lag_s, DT)
        alpha = DT / lag
        self._hist = self._hist[1:] + [self.reading_g]
        self.reading_g += alpha * (self.mass_g - self.reading_g)
        self.reading_g += float(self.rng.normal(0.0, m.noise_g))
        self.t += 1
        info = {
            "mass_g": self.mass_g,
            "reading_g": self.reading_g,
            "spilled_g": self.spilled_g,
            "tilt": self.tilt,
            "success": self.success,
        }
        return self.obs(action), info

    @property
    def success(self) -> bool:
        settled = self.tilt < 0.03 and self._in_spout < 0.01
        on_target = abs(self.reading_g - self.target_g) <= TOL_G
        clean = self.spilled_g <= 0.05
        return bool(settled and on_target and clean)


def scripted_action(env: PowderDispense) -> float:
    """Close early enough that lag, delay, and residual do not overshoot."""
    m = env.material
    d_reading = env.reading_g - env._hist[-1]
    flow = max(0.0, d_reading) / DT
    coast = flow * (m.delay_s + m.lag_s) + env._in_spout
    if env.reading_g + coast >= env.target_g - 0.4 * TOL_G:
        return -1.0
    if env.tilt > m.spill_tilt - 0.08:
        return -0.4
    if flow < 0.15:
        return 0.8
    return 0.15


def rollout(material: Material, target_g: float, seed: int) -> dict | None:
    rng = np.random.default_rng(seed)
    env = PowderDispense(material, target_g, rng)
    obs, actions, next_obs = [], [], []
    o = env.obs()
    for _ in range(HORIZON):
        a = scripted_action(env)
        o2, info = env.step(a)
        obs.append(o)
        actions.append([a])
        next_obs.append(o2)
        o = o2
        if info["success"]:
            return {
                "obs": np.stack(obs).astype(np.float32),
                "actions": np.asarray(actions, dtype=np.float32),
                "next_obs": np.stack(next_obs).astype(np.float32),
                "target_g": target_g,
                "mass_g": float(info["mass_g"]),
                "reading_g": float(info["reading_g"]),
                "spilled_g": float(info["spilled_g"]),
                "seed": seed,
                "material": material.as_dict(),
            }
    return None
