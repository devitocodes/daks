import numpy as np
from fwi import fwi_gradient, initial_setup, mat2vec, clip_boundary_and_numpy, vec2mat, fwi_gradient_shot
from dask_setup import setup_dask


def fwi_gradient_local(vp_in, model, geometry, nshots, solver_params):
    vp_in = vec2mat(vp_in, model)

    assert(model.shape == vp_in.shape)

    objective = 0.

    grad = np.zeros(model.shape)
    
    for i in range(nshots):
        o, g = fwi_gradient_shot(vp_in, i, solver_params)
        objective += o
        grad += g

    return objective, mat2vec(grad).astype(np.float64)


def test_equivalence_local_remote_single_shot():
    initial_model_filename, tn, dtype, so, nbl = "overthrust_3D_initial_model_2D.h5", 4000, np.float32, 6, 40
    model, geometry, bounds = initial_setup(filename=initial_model_filename, tn=tn, dtype=dtype, space_order=so, nbl=nbl)

    solver_params = {'filename': initial_model_filename, 'tn': tn, 'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                         'nbl': nbl, 'origin': model.origin, 'spacing': model.spacing}

    clipped_vp = mat2vec(clip_boundary_and_numpy(model.vp.data, model.nbl))

    local_results = fwi_gradient_local(clipped_vp, model, geometry, 1, solver_params)

    client = setup_dask()
    remote_results = fwi_gradient(clipped_vp, model, geometry, 1, client, solver_params)

    assert(np.isclose(local_results[0], remote_results[0], rtol=1e-4))

    assert(np.allclose(local_results[1], remote_results[1], rtol=1e-5))


def test_vec2mat():
    shape = (2, 2, 2)
    vec = np.arange(8)
    mat = vec.reshape(shape)

    assert(np.array_equal(mat, vec2mat(vec, shape)))

# ToTest: initial_setup, mat2vec, fwi_gradient_shot, clip_boundary_and_numpy


if __name__ == "__main__":
    test_equivalence_local_remote_single_shot()
