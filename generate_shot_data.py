import numpy as np
from simple import overthrust_setup
from util import from_hdf5
from azureio import load_blob_to_hdf5, save_shot
from distributed import Client, wait


filename = "overthrust_3D_true_model_2D.h5"
nsrc = 10

model_data = load_blob_to_hdf5("models", filename)
model = from_hdf5(model_data, datakey="m", dtype=np.float32, space_order=2,
                  nbpml=40)
spacing = model.spacing

basename = "shots"

src_locations = np.linspace(0, model.domain_size[0], nsrc)

def generate_shot(shot_id, src_coords):
    model_data = load_blob_to_hdf5("models", filename)
    solver = overthrust_setup(model_data, src_coordinates=src_coords,
                              datakey="m")

    rec, u, _ = solver.forward()
    save_shot(shot_id, rec.data, src_coords)


client = Client('51.11.43.137:8786')


futures = []
for i in range(nsrc):
    src_coords = np.empty((1, 2), dtype=np.float32)
    src_coords[0, 0] = model.origin[0] + src_locations[i]
    src_coords[0, 1] = model.origin[1] + 2*spacing[1]

    futures.append(client.submit(generate_shot, i, src_coords))

wait(futures)

print("Successfully generated %d shots and uploaded to Azure blob storage"%nsrc)

