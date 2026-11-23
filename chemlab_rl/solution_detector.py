"""Camera detector for undissolved solid (subtask 2), v1: image-located floor + local contrast.

Inputs a detector may use (the same a real robot has): the fixed ``inspect`` camera image
(also the policy's 4th view), its pose/intrinsics, and forward kinematics (the left tool
pose). It never reads the vial pose, grain state or liquid state.

Protocol: after a shake burst the vial is held still for ``SETTLE_S`` (plus a camera-only
stillness check) so undissolved grains sink to the floor. Then:

1. Search window: the nominal vial floor (left tool pose + planned body grasp) projected into
   the image. Only a window centre: a vial hanging tilted or shifted in the grip is fine.
2. Floor ROI found IN THE IMAGE: vial silhouette against the backdrop colour (window bottom
   rows), the bright liquid-floor disk in its lower part (Otsu on luminance), largest blob,
   convex hull (dark grains do not punch holes), eroded off the glass ring.
3. Score = ROI pixels whose colour differs from a ``MEDIAN_K`` median-filtered image by more
   than ``RESID_TAU``: grain-sized specks survive, step edges (meniscus, glass) do not.
   ``clear`` = score <= ``CLEAR_MAX_PX``.

Calibrated on a real episode (white solid, dark backdrop) against undissolved fraction:
78 %: ~700 px, 12 %: ~260, 0.8 %: ~70, 0.3 %: ~45, 0.1 %: ~13, 0 %: 1-3. So ``clear`` fires
only below ~0.1 % undissolved (success needs < 1 %). ``ground_truth_px`` (segmentation, glass
and liquid hidden) is for calibration only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RES = 896
SETTLE_S = 1.0
CAM = "inspect"                    # fixed inspection camera (also the policy's 4th view)
WIN = (170, 110, 140)              # search window: rows above / below, cols either side (px)
VIAL_TAU = 40.0                    # colour distance from the backdrop: vial pixel
MEDIAN_K = 9
RESID_TAU = 12.0
CLEAR_MAX_PX = 8
ERODE_PX = 4


@dataclass
class Look:
    score: int                     # solid pixels in the floor band
    floor_px: int                  # floor-band pixels (ROI size)
    ref_rgb: np.ndarray
    solid_rgb: np.ndarray | None   # mean colour of the solid pixels (None if none)
    clear: bool
    backdrop: str
    t: float
    extra: dict = field(default_factory=dict)

    @property
    def solid_luminance(self) -> float | None:
        return None if self.solid_rgb is None else float(self.solid_rgb @ [0.2126, 0.7152, 0.0722] / 255.0)


def luminance(rgb) -> float:
    return float(np.asarray(rgb, float) @ [0.2126, 0.7152, 0.0722])


def choose_backdrop(solid_rgb_255) -> str:
    """The half whose luminance is farther from the solid's (contrast)."""
    from chemlab_rl.solution_scene import BACKDROP_RGB
    ls = luminance(np.asarray(solid_rgb_255) / 255.0)
    return max(BACKDROP_RGB, key=lambda h: abs(luminance(BACKDROP_RGB[h]) - ls))


class GrainDetector:
    def __init__(self, env, res: int = RES):
        import mujoco
        self.mj, self.env, self.res = mujoco, env, res
        self.renderer = mujoco.Renderer(env.model, res, res)
        self.cam = env.model.camera(CAM).id
        self.fy = (res / 2) / np.tan(np.deg2rad(env.model.cam_fovy[self.cam]) / 2)
        self.looks: list[Look] = []

    # ------------------------------------------------------------ geometry
    def _vial_frame(self):
        """Vial base from the LEFT tool pose and the planned grasp (no vial state): the
        vial body is pinched BODY_GRIP_Z above its base, hanging upright."""
        from chemlab_rl.solution_skills import BODY_GRIP_Z
        p, _ = self.env.session.arms["left"].tool_pose()
        return p - np.array([0.0, 0.0, BODY_GRIP_Z])

    def _project(self, pts):
        d = self.env.data
        c, R = d.cam_xpos[self.cam], d.cam_xmat[self.cam].reshape(3, 3)
        q = (pts - c) @ R                       # camera frame: x right, y up, -z forward
        z = -q[:, 2]
        u = self.res / 2 + self.fy * q[:, 0] / z
        v = self.res / 2 - self.fy * q[:, 1] / z
        return u, v, z

    def window(self):
        """Search window (y0, y1, x0, x1) around the nominal vial floor."""
        from chemlab_rl.solution_scene import VIAL_FLOOR
        p = self._vial_frame() + [0.0, 0.0, VIAL_FLOOR]
        u, v, _ = self._project(p[None])
        cx, cy = int(u[0]), int(v[0])
        up, down, side = WIN
        return max(cy - up, 0), min(cy + down, self.res), max(cx - side, 0), min(cx + side, self.res)

    @staticmethod
    def floor_roi(win: np.ndarray):
        """Liquid-floor disk of the vial inside a window (float RGB), or None."""
        import cv2
        bg = np.median(win[-12:].reshape(-1, 3), axis=0)
        vial = np.linalg.norm(win - bg, axis=2) > VIAL_TAU
        rows = np.nonzero(vial.any(axis=1))[0]
        if len(rows) < 20:
            return None
        lum = win @ [0.2126, 0.7152, 0.0722]
        top, bot = rows.min(), rows.max()
        cand = vial.copy()
        cand[: int(bot - 0.40 * (bot - top))] = False
        thr, _ = cv2.threshold(np.clip(lum[cand], 0, 255).astype(np.uint8).reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = (cand & (lum > thr)).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
        if n < 2:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pts = np.column_stack(np.nonzero(lab == k)[::-1]).astype(np.int32)
        roi = np.zeros(vial.shape, np.uint8)
        cv2.fillConvexPoly(roi, cv2.convexHull(pts), 1)
        roi = cv2.erode(roi, np.ones((2 * ERODE_PX + 1,) * 2, np.uint8)).astype(bool)
        return roi if roi.sum() >= 200 else None

    def masks(self, img=None):
        """(floor ROI, window) in full-image coordinates (ROI None if the vial is not found)."""
        img = self.render() if img is None else img
        y0, y1, x0, x1 = self.window()
        roi = self.floor_roi(img[y0:y1, x0:x1].astype(float))
        if roi is None:
            return None, (y0, y1, x0, x1)
        full = np.zeros(img.shape[:2], bool)
        full[y0:y1, x0:x1] = roi
        return full, (y0, y1, x0, x1)

    # ------------------------------------------------------------ rendering
    def render(self) -> np.ndarray:
        self.env.mj.mj_forward(self.env.model, self.env.data)
        self.renderer.update_scene(self.env.data, camera=self.cam, scene_option=self.env._render_option)
        return self.renderer.render()

    def still(self, wait, tol: float = 2.0) -> bool:
        """Two frames, `wait()` apart: is the search window unchanged (mean |diff| < tol)?"""
        y0, y1, x0, x1 = self.window()
        a = self.render()[y0:y1, x0:x1].astype(float)
        wait()
        b = self.render()[y0:y1, x0:x1].astype(float)
        return float(np.abs(a - b).mean()) < tol

    def look(self, backdrop: str) -> Look:
        import cv2
        img = self.render()
        roi, (y0, y1, x0, x1) = self.masks(img)
        if roi is None:
            lk = Look(score=10 ** 6, floor_px=0, ref_rgb=np.zeros(3), solid_rgb=None, clear=False,
                      backdrop=backdrop, t=float(self.env.data.time), extra=dict(found=False))
            self.looks.append(lk)
            return lk
        med = cv2.medianBlur(np.ascontiguousarray(img), MEDIAN_K).astype(float)
        resid = np.abs(img.astype(float) - med).max(axis=2)
        solid = roi & (resid > RESID_TAU)
        score = int(solid.sum())
        lk = Look(score=score, floor_px=int(roi.sum()), ref_rgb=np.median(img[roi], axis=0),
                  solid_rgb=img[solid].astype(float).mean(axis=0) if score else None,
                  clear=score <= CLEAR_MAX_PX, backdrop=backdrop, t=float(self.env.data.time),
                  extra=dict(found=True))
        self.looks.append(lk)
        return lk

    # ------------------------------------------------ calibration only (privileged)
    def ground_truth_px(self) -> int:
        """Visible grain pixels in the floor band (segmentation with glass/liquid hidden)."""
        m, env = self.env.model, self.env
        hide = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(("vial_a_visual", "vial_a_liquid"))]
        saved = m.geom_rgba[hide, 3].copy()
        m.geom_rgba[hide, 3] = 0.0
        try:
            self.renderer.enable_segmentation_rendering()
            self.renderer.update_scene(env.data, camera=self.cam, scene_option=env._render_option)
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
            m.geom_rgba[hide, 3] = saved
        ids, kinds = seg[..., 0], seg[..., 1]
        grain = np.isin(ids, env.grains.geoms[env.grains.active]) & (kinds == self.mj.mjtObj.mjOBJ_GEOM)
        y0, y1, x0, x1 = self.window()
        return int(grain[y0:y1, x0:x1].sum())


def _erode(m):
    """Remove pixels without at least one neighbour (stray projection specks)."""
    n = np.zeros(m.shape, int)
    n[1:, :] += m[:-1, :]; n[:-1, :] += m[1:, :]; n[:, 1:] += m[:, :-1]; n[:, :-1] += m[:, 1:]
    return m & (n >= 2)
