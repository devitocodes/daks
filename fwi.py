import click
from distributed import wait, Client
from util import write_results, to_hdf5
import numpy as np
import h5py
import os
import time
from devito import Function, TimeFunction, clear_cache
from devito.logger import error
from examples.seismic import AcquisitionGeometry, Receiver, Model
from examples.seismic.acoustic import AcousticWaveSolver
from examples.checkpointing.checkpoint import (CheckpointOperator,
                                               DevitoCheckpoint)
from data.overthrust import overthrust_model_iso, create_geometry, overthrust_solver_iso
from functools import partial
from scipy.optimize import minimize, Bounds
from util import Profiler, clip_boundary_and_numpy, mat2vec, vec2mat, reinterpolate
from pyrevolve import Revolver
from dask_setup import setup_dask
from azureio import load_blob_to_hdf5
from fwiio import load_shot


profiler = Profiler()

@click.command()
@click.option("--initial-model-filename", default="overthrust_3D_initial_model_2D.h5", help="File to read the initial model from")
@click.option("--final-solution-basename", default="fwi", help="Filename for the final solution")
@click.option("--tn", default=4000, type=int, help="Number of timesteps to run")
@click.option("--nshots", default=20, type=int, help="Number of shots (already decided when generating shots)")
@click.option("--shots-container", default="shots", type=str, help="Name of container to read shots from")
@click.option("--so", default=6, type=int, help="Spatial discretisation order")
@click.option("--nbl", default=40, type=int, help="Number of absorbing boundary layers to add to the model")
@click.option("--kernel", default="OT2", help="Computation kernel to use (options: OT2, OT4)")
@click.option("--checkpointing/--no-checkpointing", default=False, help="Enable/disable checkpointing")
@click.option("--n-checkpoints", default=1000, type=int, help="Number of checkpoints to use")
@click.option("--compression", default=None, type=click.Choice([None, 'zfp', 'sz', 'blosc']), help="Compression scheme to use (checkpointing must be enabled to use compression)")
@click.option("--tolerance", default=None, type=int, help="Error tolerance for lossy compression, used as 10^-t")
def run(initial_model_filename, final_solution_basename, tn, nshots, shots_container, so, nbl, kernel, checkpointing, n_checkpoints, compression, tolerance):
    
    dtype = np.float32
    model, geometry, bounds = initial_setup(initial_model_filename, tn, dtype, so, nbl)

    solver_params = {'filename': initial_model_filename, 'tn': tn, 'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                         'nbl': nbl, 'origin': model.origin, 'spacing': model.spacing, 'shots_container': shots_container}

    client = setup_dask()
    
    f_args = [model, geometry, nshots, client, solver_params]
    
    if checkpointing:
        f_g = fwi_gradient_checkpointed
        compression_params = {'scheme': compression,
                          'tolerance': 10**(-tolerance)}
        f_args.append(n_checkpoints)
        f_args.append(compression_params)
    else:
        f_g = fwi_gradient

    clipped_vp = mat2vec(clip_boundary_and_numpy(model.vp.data, model.nbl))

    fwi_iteration = 0
    
    def callback(final_solution_basename, vec):
        callback.call_count += 1
        fwi_iteration = callback.call_count
        filename = "%s_%d.h5" % (final_solution_basename, fwi_iteration)
        with profiler.get_timer('io', 'write_progress'):
            to_hdf5(vec2mat(vec, model.shape), filename)
        print(profiler.summary())

    callback.call_count = 0

    partial_callback = partial(callback, final_solution_basename)
        
    solution_object = minimize(f_g,
                               clipped_vp,
                               args=tuple(f_args),
                               jac=True, method='L-BFGS-B',
                               callback=partial_callback, bounds=bounds,
                               options={'disp': True, 'maxiter': 60})

    final_model = vec2mat(solution_object.x, model.shape)

    true_model = overthrust_model_iso("overthrust_3D_true_model_2D.h5",
                           datakey="m", dtype=dtype, space_order=so,
                           nbl=nbl)
    true_model_vp = clip_boundary_and_numpy(true_model.vp.data, true_model.nbl)

    error_norm = np.linalg.norm(true_model_vp - final_model)
    print(error_norm)

    data = {'error_norm': error_norm,
            'checkpointing': checkpointing,
            'compression': compression,
            'tolerance': tolerance,
            'ncp': n_checkpoints}

    write_results(data, "fwi_experiment.csv")

    to_hdf5(final_model, '%s_final.h5' % output_filename)


def initial_setup(filename, tn, dtype, space_order, nbl, datakey="m0"):
    model = overthrust_model_iso(filename, datakey=datakey,
                      dtype=dtype, space_order=space_order, nbl=nbl)

    geometry = create_geometry(model, tn)

    clipped_model = clip_boundary_and_numpy(model.vp, model.nbl)
    vmax = np.ones(clipped_model.shape) * 6.5
    vmin = np.ones(clipped_model.shape) * 1.3

    vmax[:, 0:20] = clipped_model[:, 0:20]
    vmin[:, 0:20] = clipped_model[:, 0:20]
    b = Bounds(mat2vec(vmin), mat2vec(vmax))

    return model, geometry, b


# This runs on the dask worker in the cloud.
# Anything passed into or returned from this function will be serialised and sent over the network.  
def fwi_gradient_shot(vp_in, i, solver_params):
    error("Initialising solver")
    tn = solver_params['tn']
    nbl = solver_params['nbl']
    space_order = solver_params['space_order']
    dtype = solver_params['dtype']
    origin = solver_params['origin']
    spacing = solver_params['spacing']

    shots_container = solver_params['shots_container']
    
    true_d, source_location, old_dt = load_shot(i, container=shots_container)
    print(str(spacing), str(origin), str(vp_in.shape))
    model = Model(vp=vp_in, nbl=nbl, space_order=space_order, dtype=dtype, shape=vp_in.shape,
                  origin=origin, spacing=spacing, bcs="damp")
    geometry = create_geometry(model, tn, source_location)

    error("tn: %d, nt: %d, dt: %f.2" % (geometry.tn, geometry.nt, geometry.dt))

    error("Reinterpolate shot from %d samples to %d samples" % (true_d.shape[0], geometry.nt))
    true_d = reinterpolate(true_d, geometry.nt, old_dt)
    
    solver = AcousticWaveSolver(model, geometry, kernel='OT2',nbl=nbl,
                                space_order=space_order, dtype=dtype)

    grad = Function(name="grad", grid=model.grid)

    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)

    u0 = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=solver.space_order,
                      save=geometry.nt)
 

    error("Forward prop")
    smooth_d, _, _ = solver.forward(save=True, u=u0)
    error("Misfit")
    residual.data[:] = smooth_d.data[:] - true_d[:]

    objective = .5*np.linalg.norm(residual.data.ravel())**2
    error("Gradient")
    solver.gradient(rec=residual, u=u0, grad=grad)

    grad = clip_boundary_and_numpy(grad.data, model.nbl)
    
    return objective, -grad

def fwi_gradient(vp_in, model, geometry, nshots, client, solver_params):
    start_time = time.time()
    vp_in = vec2mat(vp_in, model.shape)
    f_vp_in = client.scatter(vp_in) # Dask enforces this for large arrays
    assert(model.shape == vp_in.shape)

    futures = []
    
    for i in range(nshots):
        futures.append(client.submit(fwi_gradient_shot, f_vp_in, i, solver_params,
                                     resources = {'tasks':1} # Ensure one task per worker (to run two, tasks=0.5)
                                         ))

    shape = model.shape
    
    def reduction(*args):
        grad = np.zeros(shape) # Closured from above
        objective = 0.
    
        for a in args:
            o, g = a
            objective += o
            grad += g
        return objective, grad

    reduce_future = client.submit(reduction, *futures)
    wait(reduce_future)

    objective, grad = reduce_future.result()
    elapsed_time = time.time() - start_time
    print("Objective function evaluation completed in %f seconds" % elapsed_time)

    # Scipy LBFGS misbehaves if type is not float64
    grad = mat2vec(np.array(grad)).astype(np.float64)
    
    return objective, grad


if __name__ == "__main__":
    run()

