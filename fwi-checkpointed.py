import numpy as np

from fwi import initial_setup, fwi_gradient
from fwiio import load_shot, Blob
from util import mat2vec, vec2mat, clip_boundary_and_numpy
from pyrevolve import Revolver
from examples.checkpointing import CheckpointOperator, DevitoCheckpoint
from devito import TimeFunction, Function
from overthrust import overthrust_solver_iso
from examples.seismic import Receiver
from dask_setup import setup_dask


def fwi_gradient_checkpointed(vp_in, nshots, client, solver, shots_container, scale_gradient=True,
                              mute_water=True, exclude_boundaries=True, water_depth=20, n_checkpoints=1000,
                              compression_params=None):
    # Create symbols to hold the gradient and residual
    model = solver.model
    so = solver.space_order
    geometry = solver.geometry
    grad = Function(name="grad", grid=model.grid, space_order=so)
    vp = Function(name="vp", grid=model.grid, space_order=so)
    smooth_d = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    objective = 0.
    time_order = 2

    if exclude_boundaries:
        vp_in = np.array(vec2mat(vp_in, solver.model.shape), dtype=solver.model.dtype)
    else:
        vp_in = np.array(vec2mat(vp_in, solver.model.vp.shape), dtype=solver.model.dtype)
    model.update("vp", vp_in)
    vp.data[:] = model.vp.data[:]

    dt = 1.75
    nt = smooth_d.data.shape[0] - 2
    u = TimeFunction(name='u', grid=model.grid, time_order=time_order,
                     space_order=so)
    v = TimeFunction(name='v', grid=model.grid, time_order=time_order,
                     space_order=so)
    fwd_op = solver.op_fwd(save=False)
    rev_op = solver.op_grad(save=False)
    cp = DevitoCheckpoint([u])
    for i in range(nshots):
        true_d, source_location, old_dt = load_shot(i)
        # Update source location
        solver.geometry.src_positions[0, :] = source_location[:]

        # Compute smooth data and full forward wavefield u0
        u.data[:] = 0.
        residual.data[:] = 0.
        v.data[:] = 0.
        smooth_d.data[:] = 0.

        wrap_fw = CheckpointOperator(fwd_op, src=solver.geometry.src, u=u, rec=smooth_d, vp=vp, dt=dt)
        wrap_rev = CheckpointOperator(rev_op, vp=vp, u=u, v=v, rec=residual, grad=grad, dt=dt)
        wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt, compression_params=compression_params)
        wrp.apply_forward()

        # Compute gradient from data residual and update objective function
        residual.data[:] = smooth_d.data[:] - true_d[:]

        objective += .5*np.linalg.norm(residual.data.ravel())**2
        wrp.apply_reverse()
        print(wrp.profiler.summary())
    if exclude_boundaries:
        grad = clip_boundary_and_numpy(grad, solver.model.nbl)

    if mute_water:
        if exclude_boundaries:
            muted_depth = water_depth
        else:
            muted_depth = water_depth + solver.model.nbl
        grad[:, 0:muted_depth] = 0

    if scale_gradient:
        grad /= np.max(np.abs(grad))
    return objective, -np.ravel(grad.data).astype(np.float64)


def verify_equivalence():
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    tn = 4000
    dtype = np.float32
    so = 6
    nbl = 40
    exclude_boundaries = True
    water_depth = 20
    nshots = 2
    client = setup_dask()

    shots_container = "shots-iso"
    scale_gradient = True
    mute_water = True
    exclude_boundaries = True

    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': 'm0', 'nbl': nbl}

    solver = overthrust_solver_iso(**solver_params)

    model, geometry, _ = initial_setup(initial_model_filename, tn, dtype, so, nbl, datakey="m0",
                                       exclude_boundaries=exclude_boundaries, water_depth=water_depth)

    if exclude_boundaries:
        v0 = mat2vec(clip_boundary_and_numpy(model.vp.data, model.nbl)).astype(np.float64)
    else:
        v0 = mat2vec(model.vp.data).astype(np.float64)

    result1 = fwi_gradient_checkpointed(v0, nshots, client, solver, shots_container, scale_gradient, mute_water,
                                        exclude_boundaries, water_depth)

    result2 = fwi_gradient(v0, nshots, client, solver, shots_container, scale_gradient, mute_water, exclude_boundaries,
                           water_depth)

    for r1, r2 in zip(result1, result2):
        np.testing.assert_allclose(r2, r1, rtol=0.01, atol=1e-8)


if __name__ == "__main__":
    verify_equivalence()
