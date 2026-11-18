"""Construct physical coarse grains; no transition-time material writes."""
import numpy as np
import mujoco


def configure_solver(spec):
    spec.option.timestep = 0.00025
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.solver = mujoco.mjtSolver.mjSOL_CG
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    spec.option.iterations = 400
    spec.option.noslip_iterations = 0
    # Milligram force sensing needs tighter convergence than visually stable
    # grain motion. A 1e-6 fixture showed 24.7 mg raw-force jitter at rest;
    # 1e-8 reduced that range below 0.5 mg without changing physical state.
    spec.option.tolerance = 1e-8
    # Sleeping contacts can disappear from static-body force sensors.
    # Keep every grain dynamic so the pan measures its supported load.


def add_grains(spec, material, count, source_origin):
    if not isinstance(count, int) or not 1 <= count <= 400:
        raise ValueError("Initial validated packing supports 1 to 400 grains")
    radius = material.radius_m
    spacing = 2 * radius + 0.00005
    xy = []
    for row in range(-4, 5):
        for col in range(-4, 5):
            x = spacing * (col + 0.5 * (row % 2))
            y = spacing * np.sqrt(3) / 2 * row
            if x * x + y * y <= (0.0121 - radius) ** 2:
                xy.append((x, y))
    if not xy:
        raise ValueError("Grain radius leaves no space inside source vial")
    for index in range(count):
        layer, slot = divmod(index, len(xy))
        x, y = xy[slot]
        z = 0.0021 + radius + spacing * layer
        if z + radius >= 0.047:
            raise ValueError("Initial grains must remain below the vial shoulder")
        body = spec.worldbody.add_body(name=f"grain_{index}",
                                      pos=(np.asarray(source_origin) + [x, y, z]).tolist())
        body.add_freejoint(name=f"grain_{index}_joint")
        body.add_geom(name=f"grain_{index}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      rgba=[0.72, 0.58, 0.32, 1], **material.geom_properties())
