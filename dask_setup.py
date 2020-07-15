import os
from distributed import Client


files_to_upload = ['azureio.py', 'fwiio.py', 'solvers.py', 'overthrust.py', 'util.py', 'dask_setup.py', 'fwi.py']


def setup_dask():
    if "DASK_SERVER_IP" not in os.environ:
        raise ValueError("DASK_SERVER_IP not set")
    server_address = os.environ['DASK_SERVER_IP']
    client = Client('%s:8786' % server_address)
    client.restart()
    for f in files_to_upload:
        print("Uploading %s" % f)
        client.upload_file(f)

    return client
