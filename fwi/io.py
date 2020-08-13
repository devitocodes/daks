import io
import h5py

from fwi.azureio import blob_from_bytes, load_blob_to_hdf5


def load_shot(num, auth, container="shots"):
    filename = "shot_%d.h5" % num

    with load_blob_to_hdf5(Blob(container, filename, auth=auth)) as f:
        data = f['data'][()]
        src_coords = f['src_coords'][()]
        dt = f['dt'][()]
    return data, src_coords, dt


def save_shot(shot_id, data, src_coords, dt, auth, container="shots"):
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        f.create_dataset("data", data=data, dtype=data.dtype)
        f.create_dataset("src_coords", data=src_coords, dtype=src_coords.dtype)
        f.create_dataset('dt', data=dt, dtype=src_coords.dtype)

    blob_from_bytes(Blob(container, "shot_%s.h5" % str(shot_id), auth=auth),
                    bio.getvalue())


def load_model(model_name, datakey, auth):
    with load_blob_to_hdf5(Blob("models", model_name, auth=auth)) as f:
        data = f[datakey][()]
    return data


class BlobAuth(object):
    def __init__(self, account_name, account_key):
        assert(account_name is not None and account_key is not None)
        self.account_name = account_name
        self.account_key = account_key


class Blob(object):
    def __init__(self, container, filename, auth=None, account_name=None, account_key=None):
        assert(auth is not None or (account_name is not None and account_key is not None))
        self.container = container
        self.filename = filename
        if auth:
            self.auth = auth
        else:
            self.auth = BlobAuth(account_name, account_key)
