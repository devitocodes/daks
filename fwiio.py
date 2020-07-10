import io
import h5py
from azureio import blob_from_bytes, load_blob_to_hdf5


def load_shot(num, container="shots"):
    filename = "shot_%d.h5" % num

    with load_blob_to_hdf5(Blob(container, filename)) as f:
        data = f['data'][()]
        src_coords = f['src_coords'][()]
        dt = f['dt'][()]
    return data, src_coords, dt


def save_shot(shot_id, data, src_coords, dt, container="shots"):
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        f.create_dataset("data", data=data, dtype=data.dtype)
        f.create_dataset("src_coords", data=src_coords, dtype=src_coords.dtype)
        f.create_dataset('dt', data=dt, dtype=src_coords.dtype)

    blob_from_bytes(Blob(container, "shot_%s.h5" % str(shot_id)),
                    bio.getvalue())


def load_model(model_name, datakey):
    with load_blob_to_hdf5(Blob("models", model_name)) as f:
        data = f[datakey][()]
    return data


class Blob():
    def __init__(self, container, filename):
        self.container = container
        self.filename = filename
