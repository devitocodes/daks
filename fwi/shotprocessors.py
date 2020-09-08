import numpy as np

from devito import Function, TimeFunction

from examples.checkpointing import DevitoCheckpoint, CheckpointOperator
from examples.seismic import Receiver

from pyrevolve import Revolver

from fwi.io import load_shot
from fwi.overthrust import overthrust_solver_iso, overthrust_solver_density
from util import reinterpolate, trim_boundary


# This runs on the dask worker in the cloud.
# Anything passed into or returned from this function will be serialised and sent over the network.
def process_shot(i, vp, solver_params, shots_container, auth, exclude_boundaries=True, dt=None):
    rec_data, source_location, old_dt = load_shot(i, auth=auth, container=shots_container)

    if solver_params['kernel'] in ['OT2', 'OT4']:
        solver = overthrust_solver_iso(**solver_params)
    elif solver_params['kernel'] == "rho":
        del solver['kernel']
        solver = overthrust_solver_density(**solver_params)

    solver.geometry.src_positions[0, :] = source_location[:]
    solver.model.update("vp", vp)

    if dt is None:
        dt = solver.model.critical_dt

    solver.geometry.resample(dt)

    # TODO: Change to built-in
    rec = reinterpolate(rec_data, solver.geometry.nt, old_dt)

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


def process_shot_checkpointed(i, vp, solver_params, shots_container, auth, exclude_boundaries=True, dt=None,
                              checkpoint_params=None):
    if solver_params['kernel'] in ['OT2', 'OT4']:
        solver = overthrust_solver_iso(**solver_params)
    elif solver_params['kernel'] == "rho":
        del solver['kernel']
        solver = overthrust_solver_density(**solver_params)

    model = solver.model
    so = solver.space_order
    geometry = solver.geometry
    time_order = 2

    if checkpoint_params is not None:
        n_checkpoints = checkpoint_params.pop('n_checkpoints', 1000)
        compression_params = checkpoint_params
    else:
        n_checkpoints = 1000
        compression_params = None

    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)

    u = TimeFunction(name='u', grid=model.grid, time_order=time_order, space_order=so)
    v = TimeFunction(name='v', grid=model.grid, time_order=time_order, space_order=so)

    fwd_op = solver.op_fwd(save=False)
    rev_op = solver.op_grad(save=False)
    cp = DevitoCheckpoint([u])

    true_d, source_location, old_dt = load_shot(i, auth=auth, container=shots_container)

    # Update source location
    solver.geometry.src_positions[0, :] = source_location[:]
    solver.model.update("vp", vp)

    if dt is None:
        dt = solver.model.critical_dt

    solver._dt = dt
    solver.geometry.resample(dt)

    nt = solver.geometry.time_axis.num - time_order

    smooth_d = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)
    residual = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis, coordinates=geometry.rec_positions)

    true_d = reinterpolate(true_d, solver.geometry.nt, old_dt)

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
