import pytest
import numpy as np

from fwi.io import Blob, default_auth
from fwi.overthrust import overthrust_solver_iso
from fwi.dasksetup import setup_dask


@pytest.fixture
def auth():
    return default_auth()


@pytest.fixture
def model():
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    datakey = "m0"
    return "%s:%s" % (initial_model_filename, datakey)


@pytest.fixture
def tn():
    return 4000


@pytest.fixture
def so():
    return 6


@pytest.fixture
def dtype():
    return np.float32


@pytest.fixture
def nbl():
    return 40


@pytest.fixture
def solver_params(model, auth, tn, so, dtype, nbl):
    initial_model_filename, datakey = model.split(":")
    return {'h5_file': Blob("models", initial_model_filename, auth=auth), 'tn': tn,
            'space_order': so, 'dtype': dtype, 'datakey': datakey, 'nbl': nbl,
            'opt': ('noop', {'openmp': True, 'par-dynamic-work': 1000})}


@pytest.fixture
def solver(solver_params):
    return overthrust_solver_iso(**solver_params)


@pytest.fixture
def client():
    return setup_dask()
