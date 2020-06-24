import numpy as np
import click
from overthrust import overthrust_model_density, overthrust_solver_density
from azureio import create_container
from fwiio import save_shot, Blob
from distributed import wait
from dask_setup import setup_dask


@click.command()
@click.option("--model-filename", default="overthrust_3D_true_model_2D.h5", help="Filename for true velocity model")
@click.option("--tn", default=4000, type=int, help="Number of timesteps to run")
@click.option("--nshots", default=20, type=int, help="Number of shots (already decided when generating shots)")
@click.option("--so", default=6, type=int, help="Spatial discretisation order")
@click.option("--nbl", default=40, type=int, help="Number of absorbing boundary layers to add to the model")
@click.option("--container", default="shots", type=str, help="Name of container to store generated shots")
def run(model_filename, tn, nshots, so, nbl, container):

    dtype = np.float32

    model = overthrust_model_density(Blob("models", model_filename), datakey="m", dtype=dtype, space_order=so, nbl=nbl)

    create_container(container)

    client = setup_dask()

    solver_params = {'tn': tn, 'space_order': so, 'dtype': dtype, 'datakey': 'm', 'nbl': nbl}

    src_coords = get_source_locations(model, nshots, dtype)

    print("Generating shots")

    futures = client.map(generate_shot, list(enumerate(src_coords)), solver_params=solver_params, container=container,
                         filename=model_filename)

    wait(futures)

    results = [f.result() for f in futures]

    if all(results):
        print("Successfully generated %d shots and uploaded to blob storage container %s" % (nshots, container))
    else:
        raise Exception("Some error occurred. Please check remote logs (currently logs can't come to local system)")


def get_source_locations(model, nshots, dtype):
    spacing = model.spacing
    src_locations = np.linspace(0, model.domain_size[0], nshots)
    src_coords = np.empty((nshots, 2), dtype=dtype)
    for i in range(nshots):
        src_coords[i, 0] = model.origin[0] + src_locations[i]
        src_coords[i, 1] = model.origin[1] + 2*spacing[1]
    return src_coords


def generate_shot(shot_info, solver_params, filename, container):
    shot_id, src_coords = shot_info
    solver_params['src_coordinates'] = src_coords

    solver = overthrust_solver_density(Blob("models", filename), **solver_params)

    rec, u, _ = solver.forward()

    save_shot(shot_id, rec.data, src_coords, solver.geometry.dt, container=container)
    return True


if __name__ == "__main__":
    run()
