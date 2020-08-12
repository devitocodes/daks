import numpy as np

from devito import Function, TimeFunction

from examples.checkpointing import DevitoCheckpoint, CheckpointOperator
from examples.seismic import Receiver

from pyrevolve import Revolver

from fwi.io import load_shot
from util import reinterpolate, trim_boundary


# This runs on the dask worker in the cloud.
# Anything passed into or returned from this function will be serialised and sent over the network.
def process_shot(i, solver, shots_container, exclude_boundaries=True, dt=None):
    rec_data, source_location, old_dt = load_shot(i, container=shots_container)

    solver.geometry.src_positions[0, :] = source_location[:]

    # TODO: Change to built-in
    rec = reinterpolate(rec_data, solver.geometry.nt, old_dt)

    if dt is None:
        dt = solver.model.critical_dt

    rec0, u0, _ = solver.forward(save=True, dt=dt)

    residual = Receiver(name='rec', grid=solver.model.grid, data=rec0.data - rec,
                        time_range=solver.geometry.time_axis,
                        coordinates=solver.geometry.rec_positions)

    objective = .5*np.linalg.norm(residual.data.ravel())**2

    grad, _ = solver.gradient(residual, u=u0, dt=dt)

    # Prepare for serialization before returning
    if exclude_boundaries:
        grad = trim_boundary(grad, solver.model.nbl)
    else:
        grad = grad.data

    grad = np.array(grad, dtype=solver.model.dtype)

    del u0
    del solver

    return objective, grad


def process_shot_checkpointed(i, solver, shots_container, exclude_boundaries=True, checkpoint_params=None):
    model = solver.model
    so = solver.space_order
    geometry = solver.geometry
    time_order = 2
    dt = solver.model.critical_dt  # 1.75
    nt = solver.geometry.time_axis.num - time_order
    if checkpoint_params is not None:
        n_checkpoints = checkpoint_params.pop('n_checkpoints', 1000)
        compression_params = checkpoint_params
    else:
        n_checkpoints = 1000
        compression_params = None

    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)

    smooth_d = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    residual = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)

    u = TimeFunction(name='u', grid=model.grid, time_order=time_order, space_order=so)
    v = TimeFunction(name='v', grid=model.grid, time_order=time_order, space_order=so)

    fwd_op = solver.op_fwd(save=False)
    rev_op = solver.op_grad(save=False)
    cp = DevitoCheckpoint([u])

    true_d, source_location, old_dt = load_shot(i, container=shots_container)
    true_d = reinterpolate(true_d, solver.geometry.nt, old_dt)
    # Update source location
    solver.geometry.src_positions[0, :] = source_location[:]

    wrap_fw = CheckpointOperator(fwd_op, src=solver.geometry.src, u=u, rec=smooth_d, dt=dt, vp=solver.model.vp)
    wrap_rev = CheckpointOperator(rev_op, u=u, v=v, rec=residual, grad=grad, dt=dt, vp=solver.model.vp)

    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt, compression_params=compression_params)
    wrp.apply_forward()

    # Compute gradient from data residual and update objective function
    residual.data[:] = smooth_d.data[:] - true_d[:]

    objective = .5*np.linalg.norm(residual.data.ravel())**2
    wrp.apply_reverse()

    # Prepare for serialization before returning
    if exclude_boundaries:
        grad = trim_boundary(grad, solver.model.nbl)
    else:
        grad = grad.data

    grad = np.array(grad, dtype=solver.model.dtype)

    return objective, grad
