"""Kinematic dual-arm cyclic-voltammetry sandbox.

This is not a Franka/UR simulator and not a procedure for a real cell.
Grippers snap to a nearby object when they close, which stands in for
contact. Subgoal completion is computed here for the learner only. It is
returned in ``info["done"]`` and is not part of the observation, so a
policy can be subgoal-agnostic in the sense of VaDar (Saxena et al., 2026).

Sequential subgoals, the order VaDar's next-future weights assume:

1. ``weigh_solid`` — left arm seats the solid on the scale. Mass is a
   noisy reading that the scripted policy lands inside the acceptance band.
2. ``dispense_electrolyte`` — right arm holds at the beaker until the noisy
   dispensed volume is in band, then lets go.
3. ``seat_cables`` — left arm seats each lead at the cell, right arm seats
   the other end at the potentiostat (working, counter, reference).
4. ``finish_sweep`` — the sweep advances only after 1–3 are true.

Observation layout (float32, length ``OBS_DIM``), with no goal index::

    0:3   left end-effector xyz
    3     left gripper (1 closed)
    4:7   right end-effector xyz
    7     right gripper
    8:29  object xyz, order solid, cell0, cell1, cell2, pot0, pot1, pot2
    29    mass reading (0 if the solid is not on the scale)
    30    dispensed volume
    31    sweep fraction in [0, 1]
    32:39 seated bits, same object order as xyz
"""

from __future__ import annotations

import numpy as np

SUBGOALS: tuple[str, ...] = (
    "weigh_solid",
    "dispense_electrolyte",
    "seat_cables",
    "finish_sweep",
)
N_SUBGOALS = len(SUBGOALS)
N_CABLES = 3
N_OBJ = 1 + 2 * N_CABLES  # solid, then three cell ends, then three pot ends
SOLID = 0
CELL_OBJS = (1, 2, 3)
POT_OBJS = (4, 5, 6)

# Object order in the observation matches these index blocks.
OBS_DIM = 39
ACTION_DIM = 8  # left xyz+grip, right xyz+grip, each component in [-1, 1]

STEP = 0.10  # meters per unit action
REACH = 0.06  # snap / seat radius
# Demonstrator actions stay inside [-ACTION_GAIN, ACTION_GAIN]. atanh(±1)
# is infinite, so a squashed-Gaussian NLL cannot target a saturated action.
ACTION_GAIN = 0.85
HORIZON = 80
SWEEP_STEPS = 10

MASS_TRUE = 1.0
MASS_LO, MASS_HI = 0.8, 1.2
VOL_LO, VOL_HI = 0.7, 1.3
# Scripted policy stops dispensing once volume reaches this. The per-step
# increment is ~0.2, so the reading stays inside [VOL_LO, VOL_HI].
DISPENSE_STOP = 0.95

_Z = 0.08
SOLID_START = np.array([0.10, 0.45, _Z])
SCALE = np.array([0.32, 0.45, _Z])
BEAKER = np.array([0.10, -0.45, _Z])
LEFT_HOME = np.array([0.10, 0.62, 0.28])
RIGHT_HOME = np.array([0.10, -0.62, 0.28])
PICK_X = 0.46
SEAT_X = 0.72
CELL_YS = (0.42, 0.22, 0.02)
POT_YS = (-0.22, -0.42, -0.62)

WS_MIN = np.array([-0.05, -0.85, 0.0])
WS_MAX = np.array([0.95, 0.85, 0.45])


def _seats() -> np.ndarray:
    seat = np.zeros((N_OBJ, 3), dtype=np.float64)
    seat[SOLID] = SCALE
    for k, y in enumerate(CELL_YS):
        seat[CELL_OBJS[k]] = (SEAT_X, y, _Z)
    for k, y in enumerate(POT_YS):
        seat[POT_OBJS[k]] = (SEAT_X, y, _Z)
    return seat


def _starts() -> np.ndarray:
    pos = np.zeros((N_OBJ, 3), dtype=np.float64)
    pos[SOLID] = SOLID_START
    for k, y in enumerate(CELL_YS):
        pos[CELL_OBJS[k]] = (PICK_X, y, _Z)
    for k, y in enumerate(POT_YS):
        pos[POT_OBJS[k]] = (PICK_X, y, _Z)
    return pos


SEATS = _seats()
STARTS = _starts()


class CVLab:
    """Dual-arm lab MDP. ``reset`` / ``step`` return a flat observation.

    ``info["done"]`` is a length-4 float vector of subgoal completions after
    the transition. ``info["terminated"]`` is full-task success. Hitting the
    horizon sets ``info["truncated"]`` and does not count as a terminal
    success, so a learner may still bootstrap that transition.
    """

    obs_dim = OBS_DIM
    action_dim = ACTION_DIM
    n_subgoals = N_SUBGOALS
    horizon = HORIZON

    def __init__(self, seed: int = 0, horizon: int = HORIZON) -> None:
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.seat = SEATS.copy()
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.ee_l = LEFT_HOME.astype(np.float64).copy()
        self.ee_r = RIGHT_HOME.astype(np.float64).copy()
        self.grip_l = 0.0
        self.grip_r = 0.0
        self.hold_l = -1
        self.hold_r = -1
        self.obj_pos = self._spawn_objects()
        self.seated = np.zeros(N_OBJ, dtype=bool)
        self.mass = 0.0
        self._mass_latched = False
        self.volume = 0.0
        self.sweep_steps = 0
        self.sweep = 0.0
        self._sync_objects()
        return self.observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.t += 1
        self.ee_l = self._clip_ws(self.ee_l + STEP * action[0:3])
        self.ee_r = self._clip_ws(self.ee_r + STEP * action[4:7])
        self._apply_hand("l", float(action[3]))
        self._apply_hand("r", float(action[7]))
        self._sync_objects()
        self._update_mass()
        if self._is_dispensing():
            bump = 0.2 + float(np.clip(self.rng.normal(0.0, 0.01), -0.03, 0.03))
            self.volume = float(np.clip(self.volume + bump, 0.0, 2.0))
        weigh, disp, cables = self._goal_flags()
        if weigh and disp and cables:
            self.sweep_steps = min(SWEEP_STEPS, self.sweep_steps + 1)
        self.sweep = self.sweep_steps / float(SWEEP_STEPS)

        done = self.done_vector()
        success = bool(done[-1] > 0.5)
        truncated = bool(self.t >= self.horizon and not success)
        info = {
            "done": done,
            "success": success,
            "terminated": success,
            "truncated": truncated,
            "t": self.t,
        }
        return self.observation(), info

    def done_vector(self) -> np.ndarray:
        """Subgoal completions at the current state. Shape ``(4,)``, values 0 or 1."""
        weigh, disp, cables = self._goal_flags()
        sweep = weigh and disp and cables and self.sweep_steps >= SWEEP_STEPS
        return np.array([weigh, disp, cables, sweep], dtype=np.float32)

    def observation(self) -> np.ndarray:
        parts = (
            self.ee_l,
            np.array([self.grip_l]),
            self.ee_r,
            np.array([self.grip_r]),
            self.obj_pos.reshape(-1),
            np.array([self.mass, self.volume, self.sweep]),
            self.seated.astype(np.float64),
        )
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape[0] != OBS_DIM:
            raise RuntimeError(f"observation length {obs.shape[0]} != {OBS_DIM}")
        return obs

    def scripted_action(self) -> np.ndarray:
        """Hand-written demonstrator used only to seed behavior cloning.

        The phased logic lives here, not in the learned policy. Actions are
        still plain 8-vectors, and the policy later trained on them never
        receives a subgoal id.
        """
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[3] = -ACTION_GAIN
        action[7] = -ACTION_GAIN
        done = self.done_vector()
        if done[0] < 0.5:
            action[0:4] = self._transport(self.ee_l, self.hold_l, SOLID)
            return action
        if done[1] < 0.5:
            action[4:8] = self._dispense_action()
            return action
        if done[2] < 0.5:
            action[0:4] = self._cable_action("l", CELL_OBJS)
            action[4:8] = self._cable_action("r", POT_OBJS)
            return action
        action[0:4] = self._delta(self.ee_l, LEFT_HOME, close=False)
        action[4:8] = self._delta(self.ee_r, RIGHT_HOME, close=False)
        return action

    def _goal_flags(self) -> tuple[bool, bool, bool]:
        weigh = bool(self.seated[SOLID] and MASS_LO <= self.mass <= MASS_HI)
        disp = bool(VOL_LO <= self.volume <= VOL_HI and not self._is_dispensing())
        cables = bool(self.seated[1:].all())
        return weigh, disp, cables

    def _is_dispensing(self) -> bool:
        near = float(np.linalg.norm(self.ee_r - BEAKER)) <= REACH
        return bool(self.grip_r > 0.5 and self.hold_r < 0 and near)

    def _spawn_objects(self) -> np.ndarray:
        pos = STARTS.copy()
        # Small xy jitter so offline demos are not identical. Seats stay fixed,
        # and the demonstrator tracks the spawned positions.
        jitter = self.rng.normal(0.0, 0.004, size=(N_OBJ, 2))
        pos[:, :2] += np.clip(jitter, -0.01, 0.01)
        return pos

    def _sync_objects(self) -> None:
        if self.hold_l >= 0:
            self.obj_pos[self.hold_l] = self.ee_l
            self.seated[self.hold_l] = False
        if self.hold_r >= 0:
            self.obj_pos[self.hold_r] = self.ee_r
            self.seated[self.hold_r] = False
        for i in range(N_OBJ):
            if i == self.hold_l or i == self.hold_r:
                continue
            dist = float(np.linalg.norm(self.obj_pos[i] - self.seat[i]))
            self.seated[i] = dist <= REACH

    def _update_mass(self) -> None:
        if self.seated[SOLID]:
            if not self._mass_latched:
                noise = float(np.clip(self.rng.normal(0.0, 0.02), -0.05, 0.05))
                self.mass = MASS_TRUE + noise
                self._mass_latched = True
        else:
            self.mass = 0.0
            self._mass_latched = False

    def _apply_hand(self, side: str, cmd: float) -> None:
        if side == "l":
            ee, other = self.ee_l, self.hold_r
            hold = self.hold_l
        else:
            ee, other = self.ee_r, self.hold_l
            hold = self.hold_r
        if cmd > 0.0:
            if hold < 0:
                grabbed = self._nearest(ee, exclude=other)
                if grabbed >= 0:
                    hold = grabbed
            grip = 1.0
        else:
            if hold >= 0:
                self._drop(hold, ee)
                hold = -1
            grip = 0.0
        if side == "l":
            self.hold_l, self.grip_l = hold, grip
        else:
            self.hold_r, self.grip_r = hold, grip

    def _nearest(self, ee: np.ndarray, exclude: int) -> int:
        best, best_d = -1, 1e9
        held = {self.hold_l, self.hold_r, exclude}
        for i in range(N_OBJ):
            if i in held:
                continue
            dist = float(np.linalg.norm(self.obj_pos[i] - ee))
            if dist <= REACH and dist < best_d:
                best, best_d = i, dist
        return best

    def _drop(self, idx: int, ee: np.ndarray) -> None:
        self.obj_pos[idx] = ee.copy()
        if float(np.linalg.norm(ee - self.seat[idx])) <= REACH:
            self.obj_pos[idx] = self.seat[idx].copy()

    def _clip_ws(self, ee: np.ndarray) -> np.ndarray:
        return np.clip(ee, WS_MIN, WS_MAX)

    def _delta(self, ee: np.ndarray, target: np.ndarray, close: bool) -> np.ndarray:
        """Proportional step toward ``target``. Zero motion once inside ``REACH``
        so the arrival step does not overshoot the snap radius.
        """
        delta = np.asarray(target, dtype=np.float64) - ee
        dist = float(np.linalg.norm(delta))
        if dist <= REACH:
            xyz = np.zeros(3, dtype=np.float32)
        else:
            xyz = np.clip(delta / STEP, -ACTION_GAIN, ACTION_GAIN).astype(np.float32)
        grip = np.float32(ACTION_GAIN if close else -ACTION_GAIN)
        return np.concatenate([xyz, [grip]])

    def _transport(self, ee: np.ndarray, hold: int, obj: int) -> np.ndarray:
        """Pick ``obj`` up and release it on its seat."""
        if hold == obj:
            arrived = float(np.linalg.norm(ee - self.seat[obj])) <= REACH
            return self._delta(ee, self.seat[obj], close=not arrived)
        if hold >= 0:
            return self._delta(ee, ee, close=False)
        pos = self.obj_pos[obj]
        arrived = float(np.linalg.norm(ee - pos)) <= REACH
        return self._delta(ee, pos, close=arrived)

    def _dispense_action(self) -> np.ndarray:
        if self.volume < DISPENSE_STOP:
            arrived = float(np.linalg.norm(self.ee_r - BEAKER)) <= REACH
            return self._delta(self.ee_r, BEAKER, close=arrived)
        park = BEAKER + np.array([0.0, 0.0, 0.15])
        return self._delta(self.ee_r, park, close=False)

    def _cable_action(self, side: str, objs: tuple[int, ...]) -> np.ndarray:
        ee = self.ee_l if side == "l" else self.ee_r
        hold = self.hold_l if side == "l" else self.hold_r
        home = LEFT_HOME if side == "l" else RIGHT_HOME
        for obj in objs:
            if not self.seated[obj]:
                return self._transport(ee, hold, obj)
        return self._delta(ee, home, close=False)
