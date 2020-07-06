import click
import matplotlib.pyplot as plt
import numpy as np
import time

from distributed import wait
from functools import partial
from scipy.optimize import minimize, Bounds
from util import write_results, to_hdf5

from examples.seismic import Receiver

from dask_setup import setup_dask
from fwiio import load_shot, Blob
from overthrust import overthrust_model_iso, create_geometry, overthrust_solver_iso
from util import Profiler, clip_boundary_and_numpy, mat2vec, vec2mat, reinterpolate


profiler = Profiler()


@click.command()
@click.option("--initial-model-filename", default="overthrust_3D_initial_model_2D.h5",
              help="File to read the initial model from")
@click.option("--final-solution-basename", default="fwi", help="Filename for the final solution")
@click.option("--tn", default=4000, type=int, help="Number of timesteps to run")
@click.option("--nshots", default=20, type=int, help="Number of shots (already decided when generating shots)")
@click.option("--shots-container", default="shots-iso", type=str, help="Name of container to read shots from")
@click.option("--so", default=6, type=int, help="Spatial discretisation order")
@click.option("--nbl", default=40, type=int, help="Number of absorbing boundary layers to add to the model")
@click.option("--kernel", default="OT2", help="Computation kernel to use (options: OT2, OT4)")
@click.option("--checkpointing/--no-checkpointing", default=False, help="Enable/disable checkpointing")
@click.option("--n-checkpoints", default=1000, type=int, help="Number of checkpoints to use")
@click.option("--compression", default=None, type=click.Choice([None, 'zfp', 'sz', 'blosc']),
              help="Compression scheme to use (checkpointing must be enabled to use compression)")
@click.option("--tolerance", default=None, type=int, help="Error tolerance for lossy compression, used as 10^-t")
def run(initial_model_filename, final_solution_basename, tn, nshots, shots_container, so, nbl, kernel, checkpointing,
        n_checkpoints, compression, tolerance):
    dtype = np.float32
    model, geometry, bounds = initial_setup(initial_model_filename, tn, dtype, so, nbl)

    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': 'm0', 'nbl': nbl}

    client = setup_dask()

    solver = overthrust_solver_iso(**solver_params)

    f_args = [model, geometry, nshots, client, solver, shots_container]

    v0 = mat2vec(clip_boundary_and_numpy(model.vp.data, model.nbl)).astype(np.float64)

    def callback(final_solution_basename, vec):
        callback.call_count += 1
        fwi_iteration = callback.call_count
        filename = "%s_%d.h5" % (final_solution_basename, fwi_iteration)
        with profiler.get_timer('io', 'write_progress'):
            to_hdf5(vec2mat(vec, model.shape), filename)

    callback.call_count = 0

    partial_callback = partial(callback, final_solution_basename)

    fwi_gradient.call_count = 0

    solution_object = minimize(fwi_gradient,
                               v0,
                               args=tuple(f_args),
                               jac=True, method='L-BFGS-B',
                               callback=partial_callback, bounds=bounds,
                               options={'disp': True, 'maxiter': 60})

    final_model = vec2mat(solution_object.x, model.vp.shape)

    true_model = overthrust_model_iso("overthrust_3D_true_model_2D.h5", datakey="m", dtype=dtype, space_order=so, nbl=nbl)
    true_model_vp = clip_boundary_and_numpy(true_model.vp.data, true_model.nbl)

    error_norm = np.linalg.norm(true_model_vp - final_model)
    print(error_norm)

    data = {'error_norm': error_norm,
            'checkpointing': checkpointing,
            'compression': compression,
            'tolerance': tolerance,
            'ncp': n_checkpoints}

    write_results(data, "fwi_experiment.csv")

    to_hdf5(final_model, '%s_final.h5' % final_solution_basename)


def initial_setup(filename, tn, dtype, space_order, nbl, datakey="m0"):
    model = overthrust_model_iso(filename, datakey=datakey, dtype=dtype, space_order=space_order, nbl=nbl)

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
def fwi_gradient_shot(vp_in, i, solver, shots_container):
    rec_data, source_location, old_dt = load_shot(i, container=shots_container)

    solver.geometry.src_positions[0, :] = source_location[:]
    solver.model.update("vp", vp_in)

    # TODO: Change to built-in
    rec = reinterpolate(rec_data, solver.geometry.nt, old_dt)

    rec0, u0, _ = solver.forward(save=True, dt=solver.model.critical_dt)

    residual = Receiver(name='rec', grid=solver.model.grid, data=rec0.data - rec,
                        time_range=solver.geometry.time_axis,
                        coordinates=solver.geometry.rec_positions)

    objective = .5*np.linalg.norm(residual.data.ravel())**2

    grad, _ = solver.gradient(residual, u=u0)
    
    dtype = solver.model.dtype

    del vp_in
    del solver

    return objective, np.array(grad.data, dtype=dtype)


def fwi_gradient(vp_in, model, geometry, nshots, client, solver, shots_container):
    fwi_gradient.call_count += 1

    start_time = time.time()
    vp_in = np.array(vec2mat(vp_in, model.shape), dtype=solver.model.dtype)

    f_vp_in = client.scatter(vp_in)  # Dask enforces this for large arrays

    solver.model.update("vp", vp_in)

    f_solver = client.scatter(solver)
    futures = []

    for i in range(nshots):
        futures.append(client.submit(fwi_gradient_shot, f_vp_in, i, f_solver, shots_container,
                                     resources={'tasks': 1}))  # Ensure one task per worker (to run two, tasks=0.5)
        # futures.append(fwi_gradient_shot(vp_in, i, solver_params))

    shape = model.vp.shape

    def reduction(*args):
        grad = np.zeros(shape)  # Closured from above
        objective = 0.

        for a in args:
            o, g = a
            objective += o
            grad += g
        return objective, grad

    reduce_future = client.submit(reduction, *futures)

    wait(reduce_future)

    objective, grad = reduce_future.result()
    # objective, grad = reduction(*futures)
    elapsed_time = time.time() - start_time
    print("Objective function evaluation completed in %f seconds. F=%f" % (elapsed_time, objective))

    # Scipy LBFGS misbehaves if type is not float64
    grad = mat2vec(clip_boundary_and_numpy(grad, solver.model.nbl)).astype(np.float64)
    # grad /= np.max(np.abs(grad)) # Scale the gradient

    from examples.seismic import plot_velocity
    model.update('vp', vp_in)

    plt.clf()
    plot_velocity(model)
    plt.savefig("progress/fwi-iter%d.pdf" % (fwi_gradient.call_count))
    return objective, -grad


if __name__ == "__main__":
    run()
