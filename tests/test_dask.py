import numpy as np
import pytest

from distributed import wait


def test_dask_upload(client):

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


def test_dask_pickling(solver, client):
    rec1, u1, _ = solver.forward()

    def noop_function(x):
        return x

    rec_future = client.submit(noop_function, rec1)
    u_future = client.submit(noop_function, u1)

    wait(rec_future)

    rec2 = rec_future.result()

    assert(np.allclose(rec1.data, rec2.data, atol=0., rtol=0.))

    wait(u_future)

    u2 = u_future.result()

    assert(np.allclose(u1.data, u2.data, atol=0., rtol=0.))


@pytest.mark.skip(reason="Numerical mismatch")
def test_remote_devito(solver, client):
    future = client.submit(solver.forward)
    rec1, u1, _ = solver.forward()
    wait(future)
    rec2, u2, _ = future.result()
    print(np.linalg.norm(rec1.data))
    print(np.linalg.norm(rec2.data))
    assert(np.allclose(rec1.data, rec2.data, atol=0., rtol=0.))
    assert((u1.data == u2.data).all())
