import numpy as np
import click
from fwi.overthrust import overthrust_model_density, overthrust_solver_density, overthrust_solver_iso
from azureio import create_container
from fwi.io import save_shot, Blob, default_auth
from distributed import wait
from fwi.dasksetup import setup_dask


@click.command()
@click.option("--model-filename", default="overthrust_3D_true_model_2D.h5", help="Filename for true velocity model")
@click.option("--tn", default=4000, type=int, help="Number of timesteps to run")
@click.option("--nshots", default=20, type=int, help="Number of shots (already decided when generating shots)")
@click.option("--so", default=6, type=int, help="Spatial discretisation order")
@click.option("--nbl", default=40, type=int, help="Number of absorbing boundary layers to add to the model")
@click.option("--shots-container", default="shots", type=str, help="Name of container to store generated shots")
@click.option("--kernel", default="rho", help="Computational kernel to use", type=click.Choice(['OT2', 'OT4', 'rho']))
@click.option("--dtype", default='float32', type=click.Choice(['float32', 'float64']),
              help="Dtype to use in computation")
def run(model_filename, tn, nshots, so, nbl, shots_container, kernel, dtype):

    if dtype == 'float32':
        dtype = np.float32
    elif dtype == 'float64':
        dtype = np.float64
    else:
        raise ValueError("Invalid dtype")

    auth = default_auth()

    model = overthrust_model_density(Blob("models", model_filename, auth=auth), datakey="m", dtype=dtype, space_order=so,
                                     nbl=nbl)

    create_container(shots_container, auth=auth)

    client = setup_dask()

    solver_params = {'tn': tn, 'space_order': so, 'dtype': dtype, 'datakey': 'm', 'nbl': nbl, 'water_depth': 20,
                     'calculate_density': True, 'kernel': kernel, 'h5_file': Blob("models", model_filename, auth=auth)}

    src_coords = get_source_locations(model, nshots, dtype)

    print("Generating shots")

    futures = []
    for i in range(nshots):
        futures.append(client.submit(generate_shot, (i, src_coords[i]),
                                     solver_params=solver_params, container=shots_container, auth=auth,
                                     resources={'tasks': 1}))

    wait(futures)

    results = [f.result() for f in futures]

    if all(results):
        print("Successfully generated %d shots and uploaded to blob storage container %s" % (nshots, shots_container))
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


def generate_shot(shot_info, solver_params, container, auth):
    shot_id, src_coords = shot_info
    solver_params['src_coordinates'] = src_coords

    kernel = solver_params['kernel']

    if kernel in ['OT2', 'OT4']:
        solver_params.pop('water_depth')
        solver_params.pop('calculate_density')
        solver = overthrust_solver_iso(**solver_params)
    elif kernel == "rho":
        solver_params.pop('kernel')
        solver = overthrust_solver_density(**solver_params)
    else:
        raise ValueError("Invalid value for kernel: %s" % kernel)

    rec, u, _ = solver.forward(dt=1.75)  # solver.model.critical_dt)

    save_shot(shot_id, rec.data, src_coords, solver.geometry.dt, auth=auth, container=container)
    return True


if __name__ == "__main__":
    run()
