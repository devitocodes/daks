import click
import numpy as np
import os

from fwi.io import Blob
from fwi.overthrust import overthrust_solver_iso
from fwi.run import initial_setup
from fwi.shotprocessors import process_shot


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
@click.option("--shot-number", default=40, type=int,
              help="Identifier of source to model")
def run(initial_model_filename, results_dir, tn, nshots, shots_container, so, nbl, kernel, scale_gradient, shot_number):
    dtype = np.float64
    water_depth = 22  # Number of points at the top of the domain that correspond to water
    exclude_boundaries = True  # Exclude the boundary regions from the optimisation problem

    initial_model_filename, datakey = initial_model_filename

    model, geometry, bounds = initial_setup(initial_model_filename, tn, dtype, so, nbl,
                                            datakey=datakey, exclude_boundaries=exclude_boundaries, water_depth=water_depth)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    print(initial_model_filename)
    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': datakey, 'nbl': nbl}
    print(solver_params)
    solver = overthrust_solver_iso(**solver_params)
    solver._dt = 1.75
    solver.geometry.resample(1.75)

    process_shot(shot_number, solver, shots_container, exclude_boundaries)


if __name__ == "__main__":
    run()
