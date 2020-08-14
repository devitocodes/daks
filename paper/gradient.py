import click
import numpy as np

from devito import Function

from fwi.io import Blob, load_shot
from fwi.overthrust import overthrust_model_iso, overthrust_solver_iso, overthrust_solver_density
from fwi.shotprocessors import process_shot, process_shot_checkpointed

from util import mat2vec, write_results


@click.command()
@click.option("--initial-model-filename", type=(str, str), default=("overthrust_3D_initial_model_2D.h5", "m0"),
              help="File (and key) to read the initial model from")
@click.option("--results-dir", default="fwiresults", help="Directory for results")
@click.option("--tn", default=4000, type=int, help="Number of timesteps to run")
@click.option("--nshots", default=20, type=int, help="Number of shots (already decided when generating shots)")
@click.option("--shots-container", default="shots-iso-40-nbl-40-so-16", type=str, help="Name of container to read shots from")
@click.option("--so", default=4, type=int, help="Spatial discretisation order")
@click.option("--nbl", default=40, type=int, help="Number of absorbing boundary layers to add to the model")
@click.option("--kernel", default="OT2", help="Computational kernel to use", type=click.Choice(['OT2', 'OT4', 'rho']))
@click.option("--scale-gradient", default=None, type=click.Choice([None, 'L', 'W']),
              help="Scale the gradient passed to LBFGS")
@click.option("--max-iter", default=30, type=int, help="Maximum number of iterations")
@click.option("--checkpointing/--no-checkpointing", default=False, help="Enable/disable checkpointing")
@click.option("--n-checkpoints", default=1000, type=int, help="Number of checkpoints to use")
@click.option("--compression", default=None, type=click.Choice([None, 'zfp', 'sz', 'blosc']),
              help="Compression scheme to use (checkpointing must be enabled to use compression)")
@click.option("--tolerance", default=None, type=int, help="Error tolerance for lossy compression, used as 10^-t")
@click.option("--reference-solution", default=None, type=str,
              help="Objective function history file for reference solution (to include in convergence plots)")
@click.option("--dtype", default='float32', type=click.Choice(['float32', 'float64']),
              help="Dtype to use in computation")
def run(initial_model_filename, results_dir, tn, nshots, shots_container, so, nbl, kernel, scale_gradient, max_iter,
        checkpointing, n_checkpoints, compression, tolerance, reference_solution, dtype):

    if dtype == 'float32':
        dtype = np.float32
    elif dtype == 'float64':
        dtype = np.float64
    else:
        raise ValueError("Invalid dtype")
    shot_id = 20
    water_depth = 20
    initial_model_filename, datakey = initial_model_filename

    rec, source_location, _ = load_shot(shot_id, container=shots_container)
    print("Source", source_location)
    print("rec", np.linalg.norm(rec))
    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': datakey, 'nbl': nbl,
                     'src_coordinates': source_location, 'opt': ('noop', {'openmp': True, 'par-dynamic-work': 1000})}

    if kernel in ['OT2', 'OT4']:
        solver_params['kernel'] = kernel
        solver = overthrust_solver_iso(**solver_params)
    elif kernel == "rho":
        solver_params['water_depth'] = water_depth
        solver_params['calculate_density'] = False
        solver = overthrust_solver_density(**solver_params)
    if not checkpointing:
        F0, gradient = process_shot(shot_id, solver, shots_container, exclude_boundaries=False)
    else:
        F0, gradient = process_shot_checkpointed(shot_id, solver, shots_container, exclude_boundaries=False,
                                                 checkpoint_params={'n_checkpoints': n_checkpoints, 'scheme': compression,
                                                                    'tolerance': tolerance})

    error1, error2, H = gradient_test_errors(solver, rec, F0, gradient)

    data = dict(zip(H, error2))
    data['compression'] = compression
    data['tolerance'] = tolerance

    write_results(data, "linearization.csv")


def gradient_test_errors(solver, rec, F0, gradient):
    true_model_filename = "overthrust_3D_true_model_2D.h5"
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    dtype = solver.model.dtype
    so = solver.space_order
    nbl = solver.model.nbl

    model_t = overthrust_model_iso(true_model_filename, datakey="m",
                                   dtype=dtype, space_order=so, nbl=nbl)
    model0 = overthrust_model_iso(initial_model_filename, datakey="m0",
                                  dtype=dtype, space_order=so, nbl=nbl)
    v = model_t.vp
    v0 = model0.vp
    dm = np.float64(v.data**(-2) - v0.data**(-2))
    print("dm", np.linalg.norm(dm))
    G = np.dot(mat2vec(gradient.data), dm.reshape(-1))
    print("G", G)
    # FWI Gradient test
    H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
    error1 = np.zeros(7)
    error2 = np.zeros(7)
    for i in range(0, 7):
        # Add the perturbation to the model
        def initializer(data):
            data[:] = np.sqrt(v0.data**2 * v.data**2 /
                              ((1 - H[i]) * v.data**2 + H[i] * v0.data**2))
        vloc = Function(name='vloc', grid=solver.model.grid, space_order=so,
                        initializer=initializer)

        # Data for the new model
        d = solver.forward(vp=vloc, dt=solver.model.critical_dt)[0]
        # First order error Phi(m0+dm) - Phi(m0)
        F_i = .5*np.linalg.norm((d.data - rec.data).reshape(-1))**2
        print("F%d" % i, F_i)
        error1[i] = np.absolute(F_i - F0)
        # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
        error2[i] = np.absolute(F_i - F0 - H[i] * G)

    return error1, error2, H


def plot_errors(error1, error2, H):
    import matplotlib.pyplot as plt
    plt.plot(H, error1, label="error1")
    plt.plot(H, error2, label="error2")
    plt.legend()
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.title('Gradient test')
    plt.xlabel('H')
    plt.ylabel('error')
    plt.show()


if __name__ == "__main__":
    run()
