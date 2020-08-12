import numpy as np
import pytest

from distributed import wait

from fwi.run import initial_setup, fwi_gradient
from fwi.dasksetup import setup_dask
from fwi.io import Blob
from fwi.overthrust import overthrust_solver_iso
from fwi.shotprocessors import process_shot, process_shot_checkpointed
from util import mat2vec, trim_boundary, vec2mat


@pytest.fixture
def shots_container():
    return "shots-iso-40-nbl-40-so-16"


def fwi_gradient_local(vp_in, nshots, solver, shots_container):
    model = solver.model

    vp_in = np.array(vec2mat(vp_in, solver.model.vp.shape), dtype=solver.model.dtype)

    assert(model.vp.shape == vp_in.shape)

    solver.model.update("vp", vp_in)

    objective = 0.

    grad = np.zeros(model.vp.shape)

    for i in range(nshots):
        o, g = process_shot(i, solver, shots_container, exclude_boundaries=False)
        objective += o
        grad += g

    return objective, -mat2vec(grad).astype(np.float64)


@pytest.mark.skip(reason="Hangs indefinitely")
def test_equivalence_local_remote_single_shot(shots_container):
    initial_model_filename, tn, dtype, so, nbl = "overthrust_3D_initial_model_2D.h5", 4000, np.float32, 6, 40
    model, _, bounds = initial_setup(filename=Blob("models", initial_model_filename), tn=tn, dtype=dtype,
                                     space_order=so, nbl=nbl)

    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn, 'space_order': so, 'dtype': dtype,
                     'datakey': 'm0', 'nbl': nbl}

    solver = overthrust_solver_iso(**solver_params)

    v0 = mat2vec(model.vp.data).astype(np.float64)

    local_results = fwi_gradient_local(v0, 1, solver, shots_container)

    client = setup_dask()

    remote_results = fwi_gradient(v0, 1, client, solver, shots_container, exclude_boundaries=False,
                                  scale_gradient=False, mute_water=False)

    np.testing.assert_approx_equal(local_results[0], remote_results[0])

    np.testing.assert_array_almost_equal(local_results[1], remote_results[1])


def test_vec2mat():
    shape = (2, 2, 2)
    vec = np.arange(8)
    mat = vec.reshape(shape)

    assert(np.array_equal(mat, vec2mat(vec, shape)))

    back = mat2vec(mat)

    assert(np.array_equal(back, vec))


@pytest.mark.xfail(reason="Numerical mismatch")
@pytest.mark.parametrize('shot_id', [20])
@pytest.mark.parametrize('exclude_boundaries', [True, False])
def test_shot(shot_id, shots_container, exclude_boundaries):
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    tn = 4000
    dtype = np.float32
    so = 6
    nbl = 40
    exclude_boundaries = True
    shot_id = 1
    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': 'm0', 'nbl': nbl,
                     'opt': ('noop', {'openmp': True, 'par-dynamic-work': 1000})}

    solver1 = overthrust_solver_iso(**solver_params)
    o1, grad1 = process_shot(shot_id, solver1, shots_container, exclude_boundaries)
    client = setup_dask()
    future = client.submit(process_shot, shot_id, solver1, shots_container, exclude_boundaries)
    wait(future)

    o2, grad2 = future.result()

    assert(np.allclose(grad1, grad2, atol=0., rtol=0.))
    assert(o1 == o2)


def test_equivalence_shot_checkpointing(shots_container):
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    tn = 4000
    dtype = np.float32
    so = 6
    nbl = 40
    exclude_boundaries = True
    water_depth = 20
    shot_id = 1

    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': 'm0', 'nbl': nbl,
                     'opt': ('noop', {'openmp': True, 'par-dynamic-work': 1000})}

    solver1 = overthrust_solver_iso(**solver_params)
    solver2 = overthrust_solver_iso(**solver_params)

    model, geometry, _ = initial_setup(initial_model_filename, tn, dtype, so, nbl,
                                       datakey="m0", exclude_boundaries=exclude_boundaries,
                                       water_depth=water_depth)
    o2, grad2 = process_shot(shot_id, solver1, shots_container, exclude_boundaries)
    o1, grad1 = process_shot_checkpointed(shot_id, solver2, shots_container, exclude_boundaries)

    np.testing.assert_approx_equal(o1, o2, significant=5)
    assert(np.allclose(grad1, grad2, rtol=1e-4))


@pytest.mark.skip(reason="Hangs indefinitely")
@pytest.mark.parametrize('mute_water', [True, False])
@pytest.mark.parametrize('scale_gradient', [None, 'L', 'W'])
@pytest.mark.parametrize('exclude_boundaries', [True, False])
def test_equivalence_checkpointing(shots_container, exclude_boundaries, scale_gradient, mute_water):
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    tn = 4000
    dtype = np.float32
    so = 6
    nbl = 40
    water_depth = 20
    nshots = 1
    client = setup_dask()

    solver_params = {'h5_file': Blob("models", initial_model_filename), 'tn': tn,
                     'space_order': so, 'dtype': dtype, 'datakey': 'm0', 'nbl': nbl}

    solver = overthrust_solver_iso(**solver_params)

    model, geometry, _ = initial_setup(initial_model_filename, tn, dtype, so, nbl,
                                       datakey="m0", exclude_boundaries=exclude_boundaries,
                                       water_depth=water_depth)

    if exclude_boundaries:
        v0 = mat2vec(np.array(trim_boundary(model.vp, model.nbl))).astype(np.float64)
    else:
        v0 = mat2vec(model.vp.data).astype(np.float64)

    o1, grad1 = fwi_gradient(v0, nshots, client, solver, shots_container, scale_gradient, mute_water,
                             exclude_boundaries, water_depth, checkpointing=True)

    o2, grad2 = fwi_gradient(v0, nshots, client, solver, shots_container, scale_gradient, mute_water,
                             exclude_boundaries, water_depth)
    print(o1, np.linalg.norm(grad1), grad1.shape)
    print(o2, np.linalg.norm(grad2), grad2.shape)

    grad1_diag = [grad1[k, k] for k in range(40)]
    grad2_diag = [grad2[k, k] for k in range(40)]

    print(grad1_diag)
    print(grad2_diag)

    np.testing.assert_approx_equal(o1, o2, significant=5)

    np.testing.assert_array_almost_equal(grad1, grad2, significant=5)


if __name__ == "__main__":
    test_equivalence_local_remote_single_shot()
