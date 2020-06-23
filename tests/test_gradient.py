import numpy as np
import pytest
from numpy import linalg

from devito import Function, info
from examples.seismic.acoustic.acoustic_example import smooth, acoustic_setup as setup
from examples.seismic import Receiver
from fwi import initial_setup, fwi_gradient_shot
from fwiio import load_shot
from dask_setup import setup_dask
from util import mat2vec, clip_boundary_and_numpy, vec2mat
from overthrust import overthrust_solver_iso


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

        dtype=np.float32

        so = 6

        nshots = 20

        nbl = 40

        true_model_filename = "overthrust_3D_true_model_2D.h5"

        shots_container = "shots-iso"

        shot_id = 1

        model0, geometry, bounds = initial_setup(initial_model_filename, tn, dtype, so, nbl)

        model_t, geometry, bounds = initial_setup(true_model_filename, tn, dtype, so, nbl, datakey="m")

        solver_params = {'h5_file': initial_model_filename, 'tn': tn, 'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                         'nbl': nbl, 'origin': model0.origin, 'spacing': model0.spacing, 'shots_container': shots_container}

        solver = overthrust_solver_iso(**solver_params)

        rec, source_location, old_dt = load_shot(shot_id, container=shots_container)

        v0 = mat2vec(clip_boundary_and_numpy(model0.vp.data, model0.nbl))

        v_t = mat2vec(clip_boundary_and_numpy(model_t.vp.data, model_t.nbl))

        dm = np.float64(v_t**(-2) - v0**(-2))

        F0, gradient = fwi_gradient_shot(vec2mat(v0, model0.shape), shot_id, solver_params)
        
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
            vloc = Function(name='vloc', grid=model0.grid, space_order=so,
                            initializer=initializer)
            # Data for the new model
            d = solver.forward(vp=vloc)[0]
            # First order error Phi(m0+dm) - Phi(m0)
            F_i = .5*linalg.norm((d.data - rec.data).reshape(-1))**2
            error1[i] = np.absolute(F_i - F0)
            # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
            error2[i] = np.absolute(F_i - F0 - H[i] * G)

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

    @pytest.mark.parametrize('space_order', [4])
    @pytest.mark.parametrize('kernel', ['OT2'])
    @pytest.mark.parametrize('shape', [(70, 80)])
    def test_gradientJ(self, shape, kernel, space_order):
        r"""
        This test ensures that the Jacobian computed with devito
        satisfies the Taylor expansion property:
        .. math::
            F(m0 + h dm) = F(m0) + \O(h) \\
            F(m0 + h dm) = F(m0) + J dm + \O(h^2) \\
        with F the Forward modelling operator.
        """
        spacing = tuple(15. for _ in shape)
        wave = setup(shape=shape, spacing=spacing, dtype=np.float64,
                     kernel=kernel, space_order=space_order,
                     tn=1000., nbl=10+space_order/2)

        v0 = Function(name='v0', grid=wave.model.grid, space_order=space_order)
        smooth(v0, wave.model.vp)
        v = wave.model.vp.data
        dm = np.float64(wave.model.vp.data**(-2) - v0.data**(-2))
        linrec = Receiver(name='rec', grid=wave.model.grid,
                          time_range=wave.geometry.time_axis,
                          coordinates=wave.geometry.rec_positions)

        # Compute receiver data and full wavefield for the smooth velocity
        rec, u0, _ = wave.forward(vp=v0, save=False)

        # Gradient: J dm
        Jdm, _, _, _ = wave.jacobian(dm, rec=linrec, vp=v0)
        # FWI Gradient test
        H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
        error1 = np.zeros(7)
        error2 = np.zeros(7)
        for i in range(0, 7):
            # Add the perturbation to the model
            def initializer(data):
                data[:] = np.sqrt(v0.data**2 * v**2 /
                                  ((1 - H[i]) * v**2 + H[i] * v0.data**2))
            vloc = Function(name='vloc', grid=wave.model.grid, space_order=space_order,
                            initializer=initializer)
            # Data for the new model
            d = wave.forward(vp=vloc)[0]
            delta_d = (d.data - rec.data).reshape(-1)
            # First order error F(m0 + hdm) - F(m0)

            error1[i] = np.linalg.norm(delta_d, 1)
            # Second order term F(m0 + hdm) - F(m0) - J dm
            error2[i] = np.linalg.norm(delta_d - H[i] * Jdm.data.reshape(-1), 1)

        # Test slope of the  tests
        p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
        p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
        info('1st order error, Phi(m0+dm)-Phi(m0) with slope: %s compared to 1' % (p1[0]))
        info(r'2nd order error, Phi(m0+dm)-Phi(m0) - <J(m0)^T \delta d, dm>with slope:'
             ' %s comapred to 2' % (p2[0]))
        assert np.isclose(p1[0], 1.0, rtol=0.1)
        assert np.isclose(p2[0], 2.0, rtol=0.1)


if __name__ == "__main__":
    TestGradient().test_gradientFWI()
