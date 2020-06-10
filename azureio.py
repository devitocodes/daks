import h5py
import io
import numpy as np
from azure.storage.blob import BlockBlobService

azure_config = {'account_name': "navjot",
                    "account_key": "<USE YOUR OWN KEY>"}

def get_blob_service():
    account_name = azure_config['account_name']
    account_key = azure_config['account_key']
    return BlockBlobService(account_name=account_name, account_key=account_key)
    
def create_container(container_name):
    service = get_blob_service()
    service.create_container(container_name)
    
def upload_file_to_blob(filename, blob_container, blob_name=None, progress_callback=None):
    if blob_name is None:
        blob_name = filename.split("/")[-1]
    service = get_blob_service()
    service.create_blob_from_path(blob_container, blob_name, filename, progress_callback=progress_callback)

def load_blob_to_hdf5(container, blob):
    block_blob_service = get_blob_service()
    data = block_blob_service.get_blob_to_bytes(container, blob)

    file = h5py.File(io.BytesIO(data.content), "r")
    return file

def blob_from_bytes(container, name, data):
    block_blob_service = get_blob_service()
    block_blob_service.create_blob_from_bytes(container, name, data)

    
