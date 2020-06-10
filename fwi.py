import click
from distributed import wait, Client
from util import write_results, to_hdf5
import numpy as np
import h5py
import os
from devito import Function, TimeFunction, clear_cache
from devito.logger import error
from examples.seismic import AcquisitionGeometry, Receiver, Model
from examples.seismic.acoustic import AcousticWaveSolver
from examples.checkpointing.checkpoint import (CheckpointOperator,
                                               DevitoCheckpoint)
from overthrust import overthrust_model_iso, create_geometry, overthrust_solver_iso

from scipy.optimize import minimize, Bounds
from util import Profiler, exception_handler
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
@click.option("--so", default=6, type=int, help="Spatial discretisation order")
@click.option("--nbl", default=40, type=int, help="Number of absorbing boundary layers to add to the model")
@click.option("--kernel", default="OT2", help="Computation kernel to use (options: OT2, OT4)")
@click.option("--checkpointing/--no-checkpointing", default=False, help="Enable/disable checkpointing")
@click.option("--n-checkpoints", default=1000, type=int, help="Number of checkpoints to use")
@click.option("--compression", default=None, type=click.Choice([None, 'zfp', 'sz', 'blosc']), help="Compression scheme to use (checkpointing must be enabled to use compression)")
@click.option("--tolerance", default=None, type=int, help="Error tolerance for lossy compression, used as 10^-t")
def run(initial_model_filename, final_solution_basename, tn, nshots, so, nbl, kernel, checkpointing, n_checkpoints, compression, tolerance):
    global_ofile = final_solution_basename
    
    path_prefix = os.path.dirname(os.path.realpath(__file__))
    dtype = np.float32
    model, geometry, bounds = initial_setup(path_prefix, initial_model_filename, tn, dtype, so, nbl)

    solver_params = {'filename': initial_model_filename, 'tn': tn, 'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                         'nbl': nbl, 'origin': model.origin, 'spacing': model.spacing}
    client = setup_dask()
    #Client(processes = False, n_workers=1, memory_limit=None, threads_per_worker=None, resources = {'tasks':1}) #
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
        
    solution_object = minimize(f_g,
                               clipped_vp,
                               args=tuple(f_args),
                               jac=True, method='L-BFGS-B',
                               callback=fncallback, bounds=bounds,
                               options={'disp': True, 'maxiter': 60})

    final_model = vec2mat(solution_object.x, model)

    true_model = overthrust_model_iso("overthrust_3D_true_model_2D.h5",
                           datakey="m", dtype=dtype, space_order=so,
                           nbl=nbl)

    error_norm = np.linalg.norm(true_model.vp.data - final_model)
    print(error_norm)

    data = {'error_norm': error_norm,
            'checkpointing': checkpointing,
            'compression': compression,
            'tolerance': tolerance,
            'ncp': n_checkpoints}

    write_results(data, "fwi_experiment.csv")

    to_hdf5(final_model, '%s_final.h5' % output_filename)


def initial_setup(path_prefix, filename, tn, dtype, space_order, nbl):
    model = overthrust_model_iso(path_prefix+"/"+filename, datakey="m0",
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
    
    true_d, source_location = load_shot(i)
    
    model = Model(vp=vp_in, nbl=nbl,space_order=space_order, dtype=dtype, shape=vp_in.shape, origin=origin, spacing=spacing)
    geometry = create_geometry(model, tn, source_location)
    
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

    vp_in = vec2mat(vp_in, model)
    f_vp_in = client.scatter(vp_in) # Dask enforces this for large arrays
    assert(model.shape == vp_in.shape)

    
    futures = []
    
    for i in range(nshots):
        futures.append(fwi_gradient_shot(vp_in, i, solver_params))
        #futures.append(client.submit(fwi_gradient_shot, f_vp_in, i, solver_params,
        #                             resources = {'tasks':1} # Ensure one task per worker (to run two, tasks=0.5)
        #                                 ))

    shape = model.shape
    
    def reduction(*args):
        grad = np.zeros(shape) # Closured from above
        objective = 0.
    
        for a in args:
            o, g = a
            objective += o
            grad += g
        return objective, grad

    #reduce_future = client.submit(reduce, *futures)
    #wait(reduce_future)
    from functools import reduce
    objective, grad = reduce(reduction, futures)
    #objective, grad = reduce_future.result()
    # Scipy LBFGS misbehaves if type is not float64
    grad = mat2vec(np.array(grad)).astype(np.float64)
    print(objective, np.linalg.norm(grad))
    return objective, grad


def fwi_gradient_checkpointed(vp_in, model, geometry, n_checkpoints=1000,
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
    with profiler.get_timer('reshape', 'vec2mat'):
        vp_in = vec2mat(vp_in)

    assert(model.vp.shape == vp_in.shape)
    vp.data[:] = vp_in[:]
    
    with profiler.get_timer('solve', 'setup'):
        solver = overthrust_setup(path_prefix+"/"+filename, datakey="m0")
    dt = solver.dt
    nt = smooth_d.data.shape[0] - 2
    u = TimeFunction(name='u', grid=model.grid, time_order=time_order,
                     space_order=4)
    v = TimeFunction(name='v', grid=model.grid, time_order=time_order,
                     space_order=4)
    with profiler.get_timer('solve', 'setup'):
        fwd_op = solver.op_fwd(save=False)
        rev_op = solver.op_grad(save=False)
        cp = DevitoCheckpoint([u])
    for i in range(nshots):
        true_d, source_location = load_shot(i, path_prefix)
        with profiler.get_timer('solve', 'reset'):
            # Update source location
            solver.geometry.src_positions[0, :] = source_location[:]

            # Compute smooth data and full forward wavefield u0
            u.data[:] = 0.
            residual.data[:] = 0.
            v.data[:] = 0.
            smooth_d.data[:] = 0.
        with profiler.get_timer('solve', 'setup'):
            wrap_fw = CheckpointOperator(fwd_op, src=solver.geometry.src, u=u,
                                             rec=smooth_d, vp=vp, dt=dt)
            wrap_rev = CheckpointOperator(rev_op, vp=vp, u=u, v=v, rec=residual,
                                              grad=grad, dt=dt)
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt,
                               compression_params=compression_params)
        with profiler.get_timer('solve', 'forward'):
            wrp.apply_forward()

        with profiler.get_timer('solve', 'process'):
            # Compute gradient from data residual and update objective function
            residual.data[:] = smooth_d.data[:] - true_d[:]

            objective += .5*np.linalg.norm(residual.data.ravel())**2
        with profiler.get_timer('solve', 'reverse'):
            wrp.apply_reverse()
        print(wrp.profiler.summary())
    # grad.data[:] /= np.max(np.abs(grad.data[:]))
    return objective, -np.ravel(grad.data).astype(np.float64)


# Global to help write unique filenames when writing out intermediate results
iter = 0
global_ofile = ""


def mat2vec(mat):
    return np.ravel(mat)


def vec2mat(vec, model):
    if vec.shape == model.shape:
        return vec
    return np.reshape(vec, model.shape)


def fncallback(vec):
    global iter
    iter += 1
    global global_ofile
    filename = "%s_%d.h5" % (global_ofile, iter)
    with profiler.get_timer('io', 'write_progress'):
        to_hdf5(vec2mat(vec), filename)
    print(profiler.summary())


def verify_equivalence():
    result1 = fwi_gradient_checkpointed(mat2vec(model.vp.data), model,
                                        geometry)

    result2 = fwi_gradient(mat2vec(model.vp.data), model, geometry)

    for r1, r2 in zip(result1, result2):
        np.testing.assert_allclose(r2, r1, rtol=0.01, atol=1e-8)


def clip_boundary_and_numpy(mat, nbl):
    return np.array(mat.data[:])[nbl:-nbl, nbl:-nbl]


if __name__ == "__main__":
    run()

