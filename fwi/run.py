import click
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Failed to import matplotlib. Plotting will not work")
import numpy as np
import time

from distributed import wait
from functools import partial
from scipy.optimize import minimize, Bounds

from fwi.dasksetup import setup_dask
from fwi.io import Blob
from fwi.overthrust import overthrust_model_iso, create_geometry, overthrust_solver_iso
from fwi.shotprocessors import process_shot, process_shot_checkpointed
from util import trim_boundary, mat2vec, vec2mat, write_results, to_hdf5


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
    water_depth = 20  # Number of points at the top of the domain that correspond to water
    exclude_boundaries = True  # Exclude the boundary regions from the optimisation problem
    scale_gradient = True  # Scale the gradient (pointwise) to be between 0-1
    mute_water = True  # Mute the gradient in the water region

    model, geometry, bounds = initial_setup(initial_model_filename, tn, dtype, so, nbl,
                                            datakey="m0", exclude_boundaries=exclude_boundaries, water_depth=water_depth)

    client = setup_dask()

    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': 'm0', 'nbl': nbl}

    solver = overthrust_solver_iso(**solver_params)
    solver._dt = 1.75
    solver.geometry.resample(1.75)

    f_args = [nshots, client, solver, shots_container, scale_gradient, mute_water, exclude_boundaries, water_depth]

    if exclude_boundaries:
        v0 = mat2vec(trim_boundary(model.vp.data, model.nbl)).astype(np.float64)
    else:
        v0 = mat2vec(model.vp.data).astype(np.float64)

    def callback(final_solution_basename, model, exclude_boundaries, vec):
        callback.call_count += 1
        fwi_iteration = callback.call_count
        filename = "%s_%d.h5" % (final_solution_basename, fwi_iteration)
        if exclude_boundaries:
            to_hdf5(vec2mat(vec, model.shape), filename)
        else:
            to_hdf5(vec2mat(vec, model.vp.shape), filename)

        from examples.seismic import plot_velocity
        plt.clf()
        plot_velocity(solver.model)
        plt.savefig("progress/fwi-iter%d.pdf" % (fwi_iteration))

    callback.call_count = 0

    partial_callback = partial(callback, final_solution_basename, model, exclude_boundaries)

    fwi_gradient.call_count = 0

    solution_object = minimize(fwi_gradient,
                               v0,
                               args=tuple(f_args),
                               jac=True, method='L-BFGS-B',
                               callback=partial_callback, bounds=bounds,
                               options={'disp': True, 'maxiter': 60})

    if exclude_boundaries:
        final_model = vec2mat(solution_object.x, model.shape)
    else:
        final_model = vec2mat(solution_object.x, model.vp.shape)

    true_model = overthrust_model_iso("overthrust_3D_true_model_2D.h5", datakey="m", dtype=dtype, space_order=so, nbl=nbl)
    true_model_vp = trim_boundary(true_model.vp.data, true_model.nbl)

    error_norm = np.linalg.norm(true_model_vp - final_model)
    print(error_norm)

    data = {'error_norm': error_norm,
            'checkpointing': checkpointing,
            'compression': compression,
            'tolerance': tolerance,
            'ncp': n_checkpoints}

    write_results(data, "fwi_experiment.csv")

    to_hdf5(final_model, '%s_final.h5' % final_solution_basename)


def initial_setup(filename, tn, dtype, space_order, nbl, datakey="m0", exclude_boundaries=True, water_depth=20):
    model = overthrust_model_iso(filename, datakey=datakey, dtype=dtype, space_order=space_order, nbl=nbl)

    geometry = create_geometry(model, tn)
    nbl = model.nbl

    if exclude_boundaries:
        v = trim_boundary(model.vp, model.nbl)
    else:
        v = model.vp.data

    # Define physical constraints on velocity - we know the maximum and minimum velocities we are expecting
    vmax = np.ones(v.shape) * 6.5
    vmin = np.ones(v.shape) * 1.3

    # Constrain the velocity for the water region. We know the velocity of water beforehand.
    if exclude_boundaries:
        vmax[:, 0:water_depth] = v[:, 0:water_depth]
        vmin[:, 0:water_depth] = v[:, 0:water_depth]
    else:
        vmax[:, 0:water_depth+nbl] = v[:, 0:water_depth+nbl]
        vmin[:, 0:water_depth+nbl] = v[:, 0:water_depth+nbl]

    b = Bounds(mat2vec(vmin), mat2vec(vmax))

    return model, geometry, b


def fwi_gradient(vp_in, nshots, client, solver, shots_container, scale_gradient=True, mute_water=True,
                 exclude_boundaries=True, water_depth=20, checkpointing=False, checkpoint_params=None):
    start_time = time.time()

    if exclude_boundaries:
        vp_in = np.array(vec2mat(vp_in, solver.model.shape), dtype=solver.model.dtype)
    else:
        vp_in = np.array(vec2mat(vp_in, solver.model.vp.shape), dtype=solver.model.dtype)

    solver.model.update("vp", vp_in)

    # Dask enforces this for large objects
    f_solver = client.scatter(solver, broadcast=True)

    futures = []

    for i in range(nshots):
        if checkpointing:
            futures.append(client.submit(process_shot_checkpointed, i, f_solver, shots_container, exclude_boundaries,
                                         checkpoint_params, resources={'tasks': 1}))
        else:
            futures.append(client.submit(process_shot, i, f_solver, shots_container, exclude_boundaries,
                                         resources={'tasks': 1}))  # Ensure one task per worker (to run two, tasks=0.5)

    if exclude_boundaries:
        gradient_shape = solver.model.shape
    else:
        gradient_shape = solver.model.vp.shape

    def reduction(*args):
        grad = np.zeros(gradient_shape)  # Closured from above
        objective = 0.

        for a in args:
            o, g = a
            objective += o
            grad += g
        return objective, grad

    reduce_future = client.submit(reduction, *futures)

    wait(reduce_future)

    objective, grad = reduce_future.result()

    if mute_water:
        if exclude_boundaries:
            muted_depth = water_depth
        else:
            muted_depth = water_depth + solver.model.nbl
        grad[:, 0:muted_depth] = 0

    # Scipy LBFGS misbehaves if type is not float64
    grad = mat2vec(grad).astype(np.float64)

    if scale_gradient:
        grad /= np.max(np.abs(grad))

    elapsed_time = time.time() - start_time
    print("Objective function evaluation completed in %f seconds. F=%f" % (elapsed_time, objective))

    return objective, -grad


if __name__ == "__main__":
    run()
