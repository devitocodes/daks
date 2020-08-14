import h5py
import io

from azure.storage.blob import BlockBlobService


def get_blob_service(auth):
    assert(hasattr(auth, "account_name") and hasattr(auth, "account_key"))
    return BlockBlobService(account_name=auth.account_name, account_key=auth.account_key)


def create_container(container_name, auth):
    service = get_blob_service(auth)
    service.create_container(container_name)


def upload_file_to_blob(filename, blob, progress_callback=None):
    service = get_blob_service(blob.auth)
    service.create_blob_from_path(blob.container, blob.filename, filename, progress_callback=progress_callback)


def load_blob_to_hdf5(blob):
    block_blob_service = get_blob_service(blob.auth)
    data = block_blob_service.get_blob_to_bytes(blob.container, blob.filename)

    f = h5py.File(io.BytesIO(data.content), "r")
    return f


def blob_from_bytes(blob, data):
    block_blob_service = get_blob_service(blob.auth)
    block_blob_service.create_blob_from_bytes(blob.container, blob.filename, data)
