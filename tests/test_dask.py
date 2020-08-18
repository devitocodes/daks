import numpy as np

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

    def noop_function(x):
        return x

    rec1, u1, _ = solver.forward()
    rec_future = client.submit(noop_function, rec1)
    u_future = client.submit(noop_function, u1)

    wait(rec_future)

    rec2 = rec_future.result()

    assert(np.allclose(rec1.data, rec2.data, atol=0., rtol=0.))

    wait(u_future)

    u2 = u_future.result()

    assert(np.allclose(u1.data, u2.data, atol=0., rtol=0.))

    solver_future = client.submit(noop_function, solver)

    wait(solver_future)

    solver2 = solver_future.result()

    assert(solver2.model.shape == solver.model.shape)
    assert(solver2.model.nbl == solver.model.nbl)
    assert(solver2.model.origin == solver.model.origin)
    assert(solver2.model.spacing == solver.model.spacing)
    assert(solver2.model.dtype == solver.model.dtype)
    assert((solver2.geometry.src_positions == solver.geometry.src_positions).all())
    assert((solver2.geometry.rec_positions == solver.geometry.rec_positions).all())
    assert(solver2.geometry.f0 == solver.geometry.f0)
    assert(solver2.geometry.tn == solver.geometry.tn)
    assert(solver2.geometry.t0 == solver.geometry.t0)
    assert(solver2.geometry.dt == solver.geometry.dt)
    assert(solver2.geometry.nt == solver.geometry.nt)
    assert(solver2.space_order == solver.space_order)
    assert(solver2.kernel == solver.kernel)
    assert(solver2._kwargs == solver._kwargs)

    assert(str(solver2.op_fwd()) == str(solver.op_fwd()))
    assert(str(solver2.op_grad()) == str(solver.op_grad()))


def test_remote_devito(solver, client):
    future = client.submit(solver.forward)
    rec1, u1, _ = solver.forward()
    wait(future)
    rec2, u2, _ = future.result()
    print(np.linalg.norm(rec1.data))
    print(np.linalg.norm(rec2.data))
    assert(np.allclose(rec1.data, rec2.data, atol=0., rtol=0.))
    assert(np.allclose(u1.data, u2.data, atol=0., rtol=0.))
