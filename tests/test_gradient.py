import numpy as np
from numpy import linalg

from devito import Function, info, error, TimeFunction

from examples.seismic import Model, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from overthrust import overthrust_solver_iso, overthrust_model_iso, create_geometry
from fwi import fwi_gradient_shot
from fwiio import load_shot
from util import mat2vec, clip_boundary_and_numpy, vec2mat
from examples.seismic.acoustic.acoustic_example import smooth, acoustic_setup as setup

class TestGradient(object):

    def test_gradientFWI(self):
        """
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
        true_model_filename = "overthrust_3D_true_model_2D.h5"
        initial_model_filename = "overthrust_3D_initial_model_2D.h5"
        tn = 4000
        dtype = np.float32
        so = 8
        nbl = 40
        shot_id = 20
        shots_container = "shots-iso-40-nbl-40-so-8"
        model0 = overthrust_model_iso(initial_model_filename, datakey="m0",
                                          dtype=dtype, space_order=so, nbl=nbl)

        model_t = overthrust_model_iso(true_model_filename, datakey="m",
                                           dtype=dtype, space_order=so, nbl=nbl)

        rec, source_location, _ = load_shot(shot_id, container=shots_container)
        solver_params = {'h5_file': initial_model_filename, 'tn': tn,
                             'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                             'nbl': nbl,
                             'src_coordinates': source_location}
        solver = overthrust_solver_iso(**solver_params)

        v = model_t.vp.data
        v0 = model0.vp
        dm = np.float64(v**(-2) - v0.data**(-2))
    
        F0, gradient = fwi_gradient_shot(v0.data, shot_id, solver, shots_container)
    
        basic_gradient_test(solver, so, v0.data, v, rec, F0, gradient, dm)


def from_scratch_gradient_test(shape=(70, 70), kernel='OT2', space_order=6):
    spacing = tuple(10. for _ in shape)
    wave = setup(shape=shape, spacing=spacing, dtype=np.float64,
                     kernel=kernel, space_order=space_order,
                     nbl=40)
    v0 = Function(name='v0', grid=wave.model.grid, space_order=space_order)
    smooth(v0, wave.model.vp)
    v = wave.model.vp.data
    
    dm = np.float64(v**(-2) - v0.data**(-2))

    # Compute receiver data for the true velocity
    rec, _, _ = wave.forward()

    # Compute receiver data and full wavefield for the smooth velocity
    rec0, u0, _ = wave.forward(vp=v0, save=True)

    # Objective function value
    F0 = .5*linalg.norm(rec0.data - rec.data)**2

    # Gradient: <J^T \delta d, dm>
    residual = Receiver(name='rec', grid=wave.model.grid, data=rec0.data - rec.data,
                        time_range=wave.geometry.time_axis,
                        coordinates=wave.geometry.rec_positions)

    gradient, _ = wave.jacobian_adjoint(residual, u0, vp=v0)
    v0 = v0.data
    basic_gradient_test(wave, space_order, v0, v, rec, F0, gradient, dm)

def basic_gradient_test(wave, space_order, v0, v, rec, F0, gradient, dm):

    G = np.dot(mat2vec(gradient.data), dm.reshape(-1))

    # FWI Gradient test
    H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
    error1 = np.zeros(7)
    error2 = np.zeros(7)
    for i in range(0, 7):
        # Add the perturbation to the model
        def initializer(data):
            data[:] = np.sqrt(v0**2 * v**2 /
                              ((1 - H[i]) * v**2 + H[i] * v0**2))
        vloc = Function(name='vloc', grid=wave.model.grid, space_order=space_order,
                        initializer=initializer)
        # Data for the new model
        d = wave.forward(vp=vloc, dt=wave.model.critical_dt)[0]
        # First order error Phi(m0+dm) - Phi(m0)
        F_i = .5*linalg.norm((d.data - rec.data).reshape(-1))**2
        error1[i] = np.absolute(F_i - F0)
        # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
        error2[i] = np.absolute(F_i - F0 - H[i] * G)
    plot_errors(error1, error2, H)
    # Test slope of the  tests
    p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
    p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
    info('1st order error, Phi(m0+dm)-Phi(m0): %s' % (p1))
    info(r'2nd order error, Phi(m0+dm)-Phi(m0) - <J(m0)^T \delta d, dm>: %s' % (p2))
    assert np.isclose(p1[0], 1.0, rtol=0.1)
    assert np.isclose(p2[0], 2.0, rtol=0.1)

def plot_errors(error1, error2, H):
    import matplotlib.pyplot as plt
    plt.plot(H, error1, label="error1")
    plt.plot(H, error2, label="error2")
    plt.legend()
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.title('Gradient test')
    plt.xlabel('H')
    plt.ylabel('error')
    plt.show()


if __name__ == "__main__":
    #from_scratch_gradient_test()
    #overthrust_from_scratch()
    TestGradient().test_gradientFWI()
