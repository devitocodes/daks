import h5py
import io
import numpy as np
from azure.storage.blob import BlockBlobService

azure_config = {'account_name': "navdaks",
                    "account_key": ""}


def load_model(model_name, datakey):
    with load_blob_to_hdf5("models", model_name) as f:
        data = f[datakey][()]
    return data

def save_shot(shot_id, data, src_coords):
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        f['data'] = data
        f['src_coords'] = src_coords
    account_name = azure_config['account_name']
    account_key = azure_config['account_key']
    block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
    block_blob_service.create_blob_from_bytes("shots", "shot_%s.h5"%str(shot_id), bio.getvalue())

def load_blob_to_hdf5(container, blob):
    account_name = azure_config['account_name']
    account_key = azure_config['account_key']
    block_blob_service = BlockBlobService(account_name=account_name, account_key=account_key)
    data = block_blob_service.get_blob_to_bytes(container, blob)

    file = h5py.File(io.BytesIO(data.content), "r")
    return file


if __name__ == "__main__":
    
    data = load_model("overthrust_3D_true_model_2D.h5", "m")
    save_shot(0, data, np.empty((1, 2), dtype=np.float32))

    
