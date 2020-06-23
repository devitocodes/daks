import numpy as np
from numpy import linalg

from devito import Function, info
from fwi import fwi_gradient_shot
from fwiio import load_shot
from util import mat2vec, clip_boundary_and_numpy, vec2mat
from data.overthrust import overthrust_solver_iso, overthrust_model_iso


class TestGradient(object):

    def test_gradientFWI(self):
        r"""
        This test ensures that the FWI gradient computed with devito
        satisfies the Taylor expansion property:
        .. math::
            \Phi(m0 + h dm) = \Phi(m0) + \O(h) \\
            \Phi(m0 + h dm) = \Phi(m0) + h \nabla \Phi(m0) + \O(h^2) \\
            \Phi(m0) = .5* || F(m0 + h dm) - D ||_2^2
        where
        .. math::
            \nabla \Phi(m0) = <J^T \delta d, dm> \\
            \delta d = F(m0+ h dm) - D \\
        with F the Forward modelling operator.
        """
        initial_model_filename = "overthrust_3D_initial_model_2D.h5"

        tn = 4000

        dtype = np.float32

        so = 6

        nbl = 40

        true_model_filename = "overthrust_3D_true_model_2D.h5"

        shots_container = "shots-rho"

        shot_id = 1

        model0 = overthrust_model_iso(initial_model_filename, datakey="m0",
                                      dtype=dtype, space_order=so, nbl=nbl)

        model_t = overthrust_model_iso(true_model_filename, datakey="m",
                                       dtype=dtype, space_order=so, nbl=nbl)

        solver_params = {'h5_file': initial_model_filename, 'tn': tn,
                         'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                         'nbl': nbl, 'origin': model0.origin,
                         'spacing': model0.spacing,
                         'shots_container': shots_container}

        solver = overthrust_solver_iso(**solver_params)

        rec, source_location, old_dt = load_shot(shot_id,
                                                 container=shots_container)

        v0 = mat2vec(clip_boundary_and_numpy(model0.vp.data, model0.nbl))

        v_t = mat2vec(clip_boundary_and_numpy(model_t.vp.data, model_t.nbl))

        dm = np.float64(v_t**(-2) - v0**(-2))

        print("dm", np.linalg.norm(dm), dm.shape)

        F0, gradient = fwi_gradient_shot(vec2mat(v0, model0.shape),
                                         shot_id, solver_params)

        G = np.dot(gradient.reshape(-1), dm.reshape(-1))

        # FWI Gradient test
        H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
        error1 = np.zeros(7)
        error2 = np.zeros(7)
        for i in range(0, 7):
            # Add the perturbation to the model
            def initializer(data):
                data[:] = np.sqrt(v0**2 * v_t**2 /
                                  ((1 - H[i]) * v_t**2 + H[i] * v0**2))
                print("data", np.linalg.norm(data), H[i])
            vloc = Function(name='vloc', grid=model0.grid, space_order=so,
                            initializer=initializer)
            
            # Data for the new model
            d = solver.forward(vp=vloc)[0]
            # First order error Phi(m0+dm) - Phi(m0)
            F_i = .5*linalg.norm((d.data - rec).reshape(-1))**2
            error1[i] = np.absolute(F_i - F0)
            # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
            error2[i] = np.absolute(F_i - F0 - H[i] * G)
            print(i, F0, F_i, H[i]*G)

        # Test slope of the  tests
        p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
        p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
        info('1st order error, Phi(m0+dm)-Phi(m0): %s' % (p1))
        info(r'2nd order error, Phi(m0+dm)-Phi(m0) - <J(m0)^T \delta d, dm>: %s' % (p2))
        print("Error 1:")
        print(error1)
        print("***")
        print("Error 2:")
        print(error2)
        assert np.isclose(p1[0], 1.0, rtol=0.1)
        assert np.isclose(p2[0], 2.0, rtol=0.1)


if __name__ == "__main__":
    TestGradient().test_gradientFWI()
