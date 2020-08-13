import h5py
import io
import os
from azure.storage.blob import BlockBlobService


def get_blob_service():
    account_name = os.environ['BLOB_ACCOUNT_NAME']
    account_key = os.environ['BLOB_ACCOUNT_KEY']
    return BlockBlobService(account_name=account_name, account_key=account_key)


def create_container(container_name):
    service = get_blob_service()
    service.create_container(container_name)


def upload_file_to_blob(filename, blob_container, blob_name=None, progress_callback=None):
    if blob_name is None:
        blob_name = filename.split("/")[-1]
    service = get_blob_service()
    service.create_blob_from_path(blob_container, blob_name, filename, progress_callback=progress_callback)


def load_blob_to_hdf5(blob):
    block_blob_service = get_blob_service()
    data = block_blob_service.get_blob_to_bytes(blob.container, blob.filename)

    f = h5py.File(io.BytesIO(data.content), "r")
    return f


def blob_from_bytes(blob, data):
    block_blob_service = get_blob_service()
    block_blob_service.create_blob_from_bytes(blob.container, blob.filename, data)
