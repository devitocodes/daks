import os
import traceback

from distributed import Client
from zipfile import ZipFile

files_to_upload = {'fwi': ['azureio.py', 'io.py', 'solvers.py', 'overthrust.py', 'run.py'],
                   'util': ['__init__.py']}


def setup_dask():
    if "DASK_SERVER_IP" not in os.environ:
        raise ValueError("DASK_SERVER_IP not set")

    server_address = os.environ['DASK_SERVER_IP']
    client = Client('%s:8786' % server_address)

    client.restart()

    upload_modules(client, files_to_upload)

    return client


def upload_modules(client, files_to_upload):
    for module, files in files_to_upload.items():
        with CompressedModule(module, files) as m:
            print("Uploading %s" % module)
            client.upload_file(m)


class CompressedModule(object):
    def __init__(self, module, files):
        self.module = module
        self.files = files

    def __enter__(self):
        self.compressed = self.compress()
        return self.compressed

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False
        else:
            os.remove(self.compressed)
            return True

    def compress(self):
        filename = '%s.zip' % self.module
        with ZipFile(filename, 'w') as zipmodule:
            for f in self.files:
                zipmodule.write(os.path.join(self.module, f))
        return filename
