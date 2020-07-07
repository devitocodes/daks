import numpy as np
from fwi import fwi_gradient, initial_setup, mat2vec, clip_boundary_and_numpy, vec2mat, fwi_gradient_shot
from dask_setup import setup_dask
from fwiio import Blob
from overthrust import overthrust_solver_iso


def fwi_gradient_local(vp_in, model, geometry, nshots, solver, shots_container):
    vp_in = vec2mat(vp_in, model.shape)

    assert(model.shape == vp_in.shape)

    objective = 0.

    grad = np.zeros(model.vp.shape)

    for i in range(nshots):
        o, g = fwi_gradient_shot(vp_in, i, solver, shots_container)
        objective += o
        grad += g

    return objective, mat2vec(grad).astype(np.float64)


def test_equivalence_local_remote_single_shot():
    initial_model_filename, tn, dtype, so, nbl = "overthrust_3D_initial_model_2D.h5", 4000, np.float32, 6, 40
    model, geometry, bounds = initial_setup(filename=Blob("models", initial_model_filename), tn=tn, dtype=dtype,
                                            space_order=so, nbl=nbl)

    solver_params = {'h5_file': initial_model_filename, 'tn': tn, 'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                     'nbl': nbl}
    shots_container = "shots-iso"
    solver = overthrust_solver_iso(**solver_params)

    v0 = mat2vec(clip_boundary_and_numpy(model.vp.data, model.nbl)).astype(np.float64)

    local_results = fwi_gradient_local(v0, model, geometry, 1, solver, shots_container)

    client = setup_dask()
    fwi_gradient.call_count = 0
    remote_results = fwi_gradient(v0, model, geometry, 1, client, solver, shots_container)

    assert(np.isclose(local_results[0], remote_results[0], rtol=1e-4))

    assert(np.allclose(local_results[1], remote_results[1], rtol=1e-5))


def test_vec2mat():
    shape = (2, 2, 2)
    vec = np.arange(8)
    mat = vec.reshape(shape)

    assert(np.array_equal(mat, vec2mat(vec, shape)))

    back = mat2vec(mat)

    assert(np.array_equal(back, vec))

# ToTest: initial_setup, mat2vec, fwi_gradient_shot, clip_boundary_and_numpy


if __name__ == "__main__":
    test_equivalence_local_remote_single_shot()
