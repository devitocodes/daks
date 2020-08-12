import numpy as np

from distributed import wait

from fwi.dasksetup import setup_dask
from fwi.overthrust import overthrust_solver_iso
from fwi.io import Blob


def test_dask_upload():
    client = setup_dask()

    def remote_test():
        try:
            from fwi.io import load_shot # noqa
            return True
        except:
            return False

    future = client.submit(remote_test)

    wait(future)
    result = future.result()
    assert(result)


def test_remote_devito():
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    tn = 4000
    so = 6
    dtype = np.float32
    datakey = "m0"
    nbl = 40
    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': datakey, 'nbl': nbl,
                     'opt': ('noop', {'openmp': True, 'par-dynamic-work': 1000})}

    solver = overthrust_solver_iso(**solver_params)

    client = setup_dask()
    future = client.submit(solver.forward)
    rec1, u1, _ = solver.forward()
    wait(future)
    rec2, u2, _ = future.result()

    assert((rec1.data == rec2.data).all())
    assert((u1.data == u2.data).all())
