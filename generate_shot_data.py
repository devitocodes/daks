import numpy as np
import click
from overthrust import overthrust_model_density, overthrust_solver_density, overthrust_model_iso, overthrust_solver_iso
from azureio import load_blob_to_hdf5, create_container
from fwiio import save_shot
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
    model_data = load_blob_to_hdf5("models", model_filename)
    model = overthrust_model_density(model_data, datakey="m", dtype=dtype, space_order=so, nbl=nbl)
    model_data.close()
    spacing = model.spacing

    create_container(container)

    src_locations = np.linspace(0, model.domain_size[0], nshots)
    client = setup_dask()

    solver_params = {'filename': model_filename, 'tn': tn, 'space_order': so, 'dtype': dtype, 'datakey': 'm',
                         'nbl': nbl, 'container': container}

    print("Generating shots")
    futures = []
    for i in range(nshots):
        src_coords = np.empty((1, 2), dtype=np.float32)
        src_coords[0, 0] = model.origin[0] + src_locations[i]
        src_coords[0, 1] = model.origin[1] + 2*spacing[1]

        futures.append(client.submit(generate_shot, i, src_coords, solver_params))

    wait(futures)
    results = [f.result() for f in futures]
    
    if all(results):
        print("Successfully generated %d shots and uploaded to blob storage container %s" % (nshots, container))
    else:
        raise Error("Some error occurred. Please check remote logs (currently logs can't come to local system)")


def generate_shot(shot_id, src_coords, solver_params):

    solver_params['src_coordinates'] = src_coords

    filename = solver_params.pop('filename')

    container = solver_params.pop('container')
    
    model_data = load_blob_to_hdf5("models", filename)
    solver = overthrust_solver_density(model_data, **solver_params)
    model_data.close()

    rec, u, _ = solver.forward()
    dt = solver.geometry.dt
    save_shot(shot_id, rec.data, src_coords, dt, container=container)
    return True

if __name__ == "__main__":
    run()
