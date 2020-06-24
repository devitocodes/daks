import numpy as np

from fwi import initial_setup, fwi_gradient
from fwiio import load_shot
from util import mat2vec, vec2mat
from pyrevolve import Revolver
from examples.checkpointing import CheckpointOperator, DevitoCheckpoint
from devito import TimeFunction, Function
from overthrust import overthrust_solver_iso
from examples.seismic import Receiver


def fwi_gradient_checkpointed(vp_in, model, geometry, nshots, client, solver_params, n_checkpoints=1000,
                              compression_params=None):
    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)
    vp = Function(name="vp", grid=model.grid)
    smooth_d = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    objective = 0.
    time_order = 2
    vp_in = vec2mat(vp_in, model.shape)

    assert(model.vp.shape == vp_in.shape)
    vp.data[:] = vp_in[:]
    filename = solver_params['filename']

    solver = overthrust_solver_iso(filename, datakey="m0")
    dt = solver.dt
    nt = smooth_d.data.shape[0] - 2
    u = TimeFunction(name='u', grid=model.grid, time_order=time_order,
                     space_order=4)
    v = TimeFunction(name='v', grid=model.grid, time_order=time_order,
                     space_order=4)
    fwd_op = solver.op_fwd(save=False)
    rev_op = solver.op_grad(save=False)
    cp = DevitoCheckpoint([u])
    for i in range(nshots):
        true_d, source_location = load_shot(i)
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

    return objective, -np.ravel(grad.data).astype(np.float64)


def verify_equivalence():

    model, geometry, _ = initial_setup()
    result1 = fwi_gradient_checkpointed(mat2vec(model.vp.data), model,
                                        geometry)

    result2 = fwi_gradient(mat2vec(model.vp.data), model, geometry)

    for r1, r2 in zip(result1, result2):
        np.testing.assert_allclose(r2, r1, rtol=0.01, atol=1e-8)
