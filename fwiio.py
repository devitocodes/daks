import io
import h5py
from azureio import blob_from_bytes, load_blob_to_hdf5

def load_shot(num):
    basepath = "shots"
    
    filename = "shot_%d.h5" % num
    
    with load_blob_to_hdf5(basepath, filename) as f:
        data = f['data'][()]
        src_coords = f['src_coords'][()]
    return data, src_coords

def save_shot(shot_id, data, src_coords):
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        f['data'] = data
        f['src_coords'] = src_coords
    
    blob_from_bytes("shots", "shot_%s.h5"%str(shot_id), bio.getvalue())

def load_model(model_name, datakey):
    with load_blob_to_hdf5("models", model_name) as f:
        data = f[datakey][()]
    return data
