import click
import csv
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Failed to import matplotlib. Plotting will not work")
import numpy as np
import os
import time
import tikzplotlib

from distributed import wait
from functools import partial
from scipy.optimize import minimize, Bounds

from fwi.dasksetup import setup_dask, reset_cluster
from fwi.io import Blob
from fwi.overthrust import overthrust_model_iso, create_geometry, overthrust_solver_iso
from fwi.shotprocessors import process_shot, process_shot_checkpointed
from util import trim_boundary, mat2vec, vec2mat, write_results, to_hdf5, plot_model_to_file, eprint


plt.style.use("ggplot")


@click.command()
@click.option("--initial-model-filename", type=(str, str), default=("overthrust_3D_initial_model_2D.h5", "m0"),
              help="File (and key) to read the initial model from")
@click.option("--results-dir", default="fwiresults", help="Directory for results")
@click.option("--tn", default=4000, type=int, help="Number of timesteps to run")
@click.option("--nshots", default=20, type=int, help="Number of shots (already decided when generating shots)")
@click.option("--shots-container", default="shots-iso", type=str, help="Name of container to read shots from")
@click.option("--so", default=6, type=int, help="Spatial discretisation order")
@click.option("--nbl", default=40, type=int, help="Number of absorbing boundary layers to add to the model")
@click.option("--kernel", default="OT2", help="Computation kernel to use (options: OT2, OT4)")
@click.option("--scale-gradient", default=None, type=click.Choice([None, 'L', 'W']),
              help="Scale the gradient passed to LBFGS")
@click.option("--checkpointing/--no-checkpointing", default=False, help="Enable/disable checkpointing")
@click.option("--n-checkpoints", default=1000, type=int, help="Number of checkpoints to use")
@click.option("--compression", default=None, type=click.Choice([None, 'zfp', 'sz', 'blosc']),
              help="Compression scheme to use (checkpointing must be enabled to use compression)")
@click.option("--tolerance", default=None, type=int, help="Error tolerance for lossy compression, used as 10^-t")
@click.option("--reference-solution", default=None, type=str,
              help="Objective function history file for reference solution (to include in convergence plots)")
def run(initial_model_filename, results_dir, tn, nshots, shots_container, so, nbl, kernel, scale_gradient, checkpointing,
        n_checkpoints, compression, tolerance, reference_solution):
    dtype = np.float32
    water_depth = 22  # Number of points at the top of the domain that correspond to water
    exclude_boundaries = True  # Exclude the boundary regions from the optimisation problem
    mute_water = True  # Mute the gradient in the water region

    initial_model_filename, datakey = initial_model_filename

    model, geometry, bounds = initial_setup(initial_model_filename, tn, dtype, so, nbl,
                                            datakey=datakey, exclude_boundaries=exclude_boundaries, water_depth=water_depth)

    client = setup_dask()

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    intermediates_dir = os.path.join(results_dir, "intermediates")

    if not os.path.exists(intermediates_dir):
        os.mkdir(intermediates_dir)

    progress_dir = os.path.join(results_dir, "progress")

    if not os.path.exists(progress_dir):
        os.mkdir(progress_dir)

    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': datakey, 'nbl': nbl}

    solver = overthrust_solver_iso(**solver_params)
    solver._dt = 1.75
    solver.geometry.resample(1.75)

    f_args = [nshots, client, solver, shots_container, scale_gradient, mute_water, exclude_boundaries, water_depth]

    if exclude_boundaries:
        v0 = mat2vec(trim_boundary(model.vp, model.nbl)).astype(np.float64)
    else:
        v0 = mat2vec(model.vp).astype(np.float64)

    def callback(progress_dir, intermediates_dir, model, exclude_boundaries, vec):
        global plot_model_to_file
        callback.call_count += 1

        if not hasattr(callback, "obj_fn_history"):
            callback.obj_fn_history = []

        callback.obj_fn_history.append(fwi_gradient.obj_fn_cache[vec.tobytes()])

        fwi_iteration = callback.call_count
        filename = os.path.join(intermediates_dir, "solution%d.h5" % fwi_iteration)
        if exclude_boundaries:
            to_hdf5(vec2mat(vec, model.shape), filename)
        else:
            to_hdf5(vec2mat(vec, model.vp.shape), filename)

        progress_filename = os.path.join(progress_dir, "fwi-iter%d.pdf" % (fwi_iteration))
        plot_model_to_file(solver.model, progress_filename)

    callback.call_count = 0

    partial_callback = partial(callback, progress_dir, intermediates_dir, model, exclude_boundaries)

    fwi_gradient.call_count = 0

    solution_object = minimize(fwi_gradient,
                               v0,
                               args=tuple(f_args),
                               jac=True, method='L-BFGS-B',
                               callback=partial_callback, bounds=bounds,
                               options={'disp': True, 'maxiter': 30})

    if exclude_boundaries:
        final_model = vec2mat(solution_object.x, model.shape)
    else:
        final_model = vec2mat(solution_object.x, model.vp.shape)

    solver.model.update("vp", final_model)

    # Save plot of final model
    final_plot_filename = os.path.join(results_dir, "final_model.pdf")
    plot_model_to_file(solver.model, final_plot_filename)

    # Save objective function values to CSV
    obj_fn_history = callback.obj_fn_history
    obj_fn_vals_filename = os.path.join(results_dir, "objective_function_values.csv")
    with open(obj_fn_vals_filename, "w") as vals_file:
        vals_writer = csv.writer(vals_file)
        for r in obj_fn_history:
            vals_writer.writerow([r])

    # Plot objective function values
    plt.title("FWI convergence")
    plt.xlabel("Iteration number")
    plt.ylabel("Objective function value")
    obj_fun_plt_filename = os.path.join(results_dir, "convergence.tex")
    obj_fun_plt_pdf_filename = os.path.join(results_dir, "convergence.pdf")

    plt.clf()
    if reference_solution is None:
        plt.plot(obj_fn_history)
    else:
        # Load reference solution convergence history
        with open(reference_solution, 'r') as reference_file:
            vals_reader = csv.reader(reference_file)
            reference_solution_values = []
            for r in vals_reader:
                reference_solution_values.append(float(r[0]))
        # Pad with 0s to ensure same size

        # Plot with legends
        plt.plot(obj_fn_history, label="Lossy FWI")
        plt.plot(reference_solution_values, label="Reference FWI")
        # Display legend
        plt.legend()
    plt.savefig(obj_fun_plt_pdf_filename)
    tikzplotlib.save(obj_fun_plt_filename)

    true_model = overthrust_model_iso("overthrust_3D_true_model_2D.h5", datakey="m", dtype=dtype, space_order=so, nbl=nbl)
    true_model_vp = trim_boundary(true_model.vp, true_model.nbl)

    error_norm = np.linalg.norm(true_model_vp - final_model)
    print("L2 norm of final solution vs true solution: %f" % error_norm)

    data = {'error_norm': error_norm,
            'checkpointing': checkpointing,
            'compression': compression,
            'tolerance': tolerance,
            'ncp': n_checkpoints}

    write_results(data, "fwi_experiment.csv")

    # Final solution
    final_solution_filename = os.path.join(results_dir, "final_solution.h5")
    to_hdf5(final_model, final_solution_filename)


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


def fwi_gradient(vp_in, nshots, client, solver, shots_container, scale_gradient=None, mute_water=True,
                 exclude_boundaries=True, water_depth=20, checkpointing=False, checkpoint_params=None):
    start_time = time.time()

    reset_cluster(client)

    if not hasattr(fwi_gradient, "obj_fn_cache"):
        fwi_gradient.obj_fn_cache = {}

    if exclude_boundaries:
        vp = np.array(vec2mat(vp_in, solver.model.shape), dtype=solver.model.dtype)
    else:
        vp = np.array(vec2mat(vp_in, solver.model.vp.shape), dtype=solver.model.dtype)

    solver.model.update("vp", vp)

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

    if scale_gradient is not None:
        if scale_gradient == "W":
            if not hasattr(fwi_gradient, "gradient_scaling_factor"):
                fwi_gradient.gradient_scaling_factor = np.max(np.abs(grad))

            grad /= fwi_gradient.gradient_scaling_factor
        elif scale_gradient == "L":
            grad /= np.max(np.abs(grad))
        else:
            raise ValueError("Invalid value %s for gradient scaling. Allowed: None, L, W" % scale_gradient)

    fwi_gradient.obj_fn_cache[vp_in.tobytes()] = objective

    elapsed_time = time.time() - start_time
    eprint("Objective function evaluation completed in %f seconds. F=%f" % (elapsed_time, objective))

    return objective, -grad


if __name__ == "__main__":
    run()
