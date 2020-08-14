import numpy as np
import pytest

from fwi.io import Blob
from fwi.overthrust import overthrust_solver_iso


@pytest.mark.parametrize('kernel', ['OT2', 'OT4'])
@pytest.mark.parametrize('tn', [2000, 4000])
@pytest.mark.parametrize('src_coordinates', [(0, 0), (50, 50)])
@pytest.mark.parametrize('space_order', [2, 4])
@pytest.mark.parametrize('nbl', [20, 40])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_overthrust_solver_iso(kernel, tn, src_coordinates, space_order, nbl, dtype, auth):
    filename = "overthrust_3D_true_model_2D.h5"
    src_coordinates = np.array(src_coordinates)
    solver = overthrust_solver_iso(Blob("models", filename, auth=auth), kernel, tn, src_coordinates, space_order, "m",
                                   nbl, dtype)

    assert(solver.kernel == kernel)
    assert(solver.geometry.tn == tn)
    assert(np.array_equal(solver.geometry.src_positions[0], src_coordinates))
    assert(solver.space_order == space_order)
    assert(solver.model.nbl == nbl)
    assert(solver.model.dtype == dtype)
