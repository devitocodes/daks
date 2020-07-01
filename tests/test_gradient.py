import numpy as np
from numpy import linalg

from devito import Function, info, error, TimeFunction

from examples.seismic import Model, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from overthrust import overthrust_solver_iso, overthrust_model_iso, create_geometry
from fwi import initial_setup
from fwiio import load_shot
from util import mat2vec, clip_boundary_and_numpy, vec2mat
from examples.seismic.acoustic.acoustic_example import smooth, acoustic_setup as setup

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
        true_model_filename = "overthrust_3D_true_model_2D.h5"

        tn = 4000

        dtype = np.float32

        so = 6

        nbl = 40

        shots_container = "shots-iso"

        shot_id = 10

        ##########

        model0 = overthrust_model_iso(initial_model_filename, datakey="m0",
                                      dtype=dtype, space_order=so, nbl=nbl)

        model_t = overthrust_model_iso(true_model_filename, datakey="m",
                                       dtype=dtype, space_order=so, nbl=nbl)

        _, geometry, _ = initial_setup(initial_model_filename, tn, dtype, so, nbl, datakey="m0")
        # rec, source_location, old_dt = load_shot(shot_id,
        #                                         container=shots_container)
        source_location = geometry.src_positions
        solver_params = {'h5_file': initial_model_filename, 'tn': tn,
                         'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                         'nbl': nbl, 'origin': model0.origin,
                         'spacing': model0.spacing,
                         'shots_container': shots_container,
                         'src_coordinates': source_location}

        solver = overthrust_solver_iso(**solver_params)

        true_solver_params = solver_params.copy()

        true_solver_params['h5_file'] = true_model_filename
        true_solver_params['datakey'] = "m"

        solver_true = overthrust_solver_iso(**true_solver_params)

        rec, _, _ = solver_true.forward()

        v0 = mat2vec(clip_boundary_and_numpy(model0.vp.data, model0.nbl))

        v_t = mat2vec(clip_boundary_and_numpy(model_t.vp.data, model_t.nbl))

        dm = np.float64(v_t**(-2) - v0**(-2))

        print("dm", np.linalg.norm(dm), dm.shape)

        F0, gradient = fwi_gradient_shot(vec2mat(v0, model0.shape),
                                         shot_id, solver_params, source_location)

        G = np.dot(gradient.reshape(-1), dm.reshape(-1))

        # FWI Gradient test
        H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
        error1 = np.zeros(7)
        error2 = np.zeros(7)
        for i in range(0, 7):
            # Add the perturbation to the model
            vloc = np.sqrt(v0**2 * v_t**2 /
                           ((1 - H[i]) * v_t**2 + H[i] * v0**2))
            m = Model(vp=vloc, nbl=nbl, space_order=so, dtype=dtype, shape=model0.shape,
                      origin=model0.origin, spacing=model0.spacing, bcs="damp")
            # Data for the new model
            d = solver.forward(vp=m.vp)[0]
            # First order error Phi(m0+dm) - Phi(m0)
            F_i = .5*linalg.norm((d.data - rec.data).reshape(-1))**2
            error1[i] = np.absolute(F_i - F0)
            # Second order term r Phi(m0+dm) - Phi(m0) - <J(m0)^T \delta d, dm>
            error2[i] = np.absolute(F_i - F0 - H[i] * G)
            print(i, F0, F_i, H[i]*G)
        plot_errors(error1, error2, H)
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


def fwi_gradient_shot(vp_in, i, solver_params, source_location):
 
    true_model_filename = "overthrust_3D_true_model_2D.h5"
    shots_container = "shots-iso"
    
    rec, source_location, old_dt = load_shot(i, container=shots_container)

    solver = overthrust_solver_iso(**solver_params)
    error("Forward prop")
    rec0, u0, _ = solver.forward(save=True)
    
    residual = Receiver(name='rec', grid=solver.model.grid, data=rec0.data - rec.data,
                        time_range=solver.geometry.time_axis,
                        coordinates=solver.geometry.rec_positions)

    print("failing")
    print("rec0", np.linalg.norm(rec0.data))
    print("rec", np.linalg.norm(rec.data))
    objective = .5*np.linalg.norm(residual.data.ravel())**2
    error("Gradient")
    grad, _ = solver.gradient(residual, u0)

    #grad = clip_boundary_and_numpy(grad.data, model.nbl)

    return objective, grad.data


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

def overthrust_from_scratch():
    true_model_filename = "overthrust_3D_true_model_2D.h5"
    initial_model_filename = "overthrust_3D_initial_model_2D.h5"
    tn = 4000
    dtype = np.float32
    so = 6
    nbl = 40
    shot_id = 10
    shots_container = "shots-iso"
    model0 = overthrust_model_iso(initial_model_filename, datakey="m0",
                                      dtype=dtype, space_order=so, nbl=nbl)
    _, geometry, _ = initial_setup(true_model_filename, tn, dtype, so, nbl, datakey="m")

    _, source_location, _ = load_shot(shot_id, container=shots_container)
    solver_params = {'h5_file': initial_model_filename, 'tn': tn,
                         'space_order': so, 'dtype': dtype, 'datakey': 'm0',
                         'nbl': nbl, 'origin': model0.origin,
                         'spacing': model0.spacing,
                         'src_coordinates': source_location}

    solver = overthrust_solver_iso(**solver_params)

    true_solver_params = solver_params.copy()

    true_solver_params['h5_file'] = true_model_filename
    true_solver_params['datakey'] = "m"

    solver_true = overthrust_solver_iso(**true_solver_params)
    #####
    #v = clip_boundary_and_numpy(solver_true.model.vp, solver_true.model.nbl)
    #v0 = clip_boundary_and_numpy(model0.vp, solver.model.nbl)
    v = solver_true.model.vp.data
    v0 = model0.vp
    dm = np.float64(v**(-2) - v0.data**(-2))
    
    # Compute receiver data for the true velocity
    rec, _, _ = solver_true.forward()

    # Compute receiver data and full wavefield for the smooth velocity
    rec0, u0, _ = solver.forward(save=True)

    # Objective function value
    F0 = .5*linalg.norm(rec0.data - rec.data)**2
    print("passing")
    print("F0", F0)
    print("rec0", np.linalg.norm(rec0.data))
    print("rec", np.linalg.norm(rec.data))
    # Gradient: <J^T \delta d, dm>
    #residual = Receiver(name='rec', grid=solver_true.model.grid, data=rec0.data - rec.data,
    #                    time_range=solver_true.geometry.time_axis,
    #                    coordinates=solver_true.geometry.rec_positions)

    #gradient, _ = solver.jacobian_adjoint(residual, u0)
    
    F0, gradient = fwi_gradient_shot(v0, shot_id, solver_params, source_location)
    
    basic_gradient_test(solver_true, so, v0.data, v, rec, F0, gradient, dm)

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
        d = wave.forward(vp=vloc)[0]
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
    overthrust_from_scratch()
    #TestGradient().test_gradientFWI()
