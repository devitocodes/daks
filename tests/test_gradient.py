import numpy as np
from numpy import linalg

from devito import Function, info, error, TimeFunction

from examples.seismic import Model, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from overthrust import overthrust_solver_iso, overthrust_model_iso, create_geometry
from fwi import initial_setup
from util import mat2vec, clip_boundary_and_numpy, vec2mat


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
    error("Initialising solver")
    tn = solver_params['tn']
    nbl = solver_params['nbl']
    space_order = solver_params['space_order']
    dtype = solver_params['dtype']
    origin = solver_params['origin']
    spacing = solver_params['spacing']
    true_model_filename = "overthrust_3D_true_model_2D.h5"
    # shots_container = solver_params['shots_container']

    # true_d, source_location, old_dt = load_shot(i, container=shots_container)

    true_solver_params = solver_params.copy()

    true_solver_params['h5_file'] = true_model_filename
    true_solver_params['datakey'] = "m"

    solver_true = overthrust_solver_iso(**true_solver_params)

    true_d, _, _ = solver_true.forward()

    model = Model(vp=vp_in, nbl=nbl, space_order=space_order, dtype=dtype, shape=vp_in.shape,
                  origin=origin, spacing=spacing, bcs="damp")
    geometry = create_geometry(model, tn, source_location)

    error("tn: %d, nt: %d, dt: %f" % (geometry.tn, geometry.nt, geometry.dt))

    # error("Reinterpolate shot from %d samples to %d samples" % (true_d.shape[0], geometry.nt))
    # true_d = reinterpolate(true_d, geometry.nt, old_dt)

    solver = AcousticWaveSolver(model, geometry, kernel='OT2', nbl=nbl,
                                space_order=space_order, dtype=dtype)

    grad = Function(name="grad", grid=model.grid)

    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)

    u0 = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=solver.space_order,
                      save=geometry.nt)

    error("Forward prop")
    smooth_d, _, _ = solver.forward(save=True, u=u0)
    error("Misfit")
    residual.data[:] = smooth_d.data[:] - true_d.data[:]

    objective = .5*np.linalg.norm(residual.data.ravel())**2
    error("Gradient")
    solver.gradient(rec=residual, u=u0, grad=grad)

    grad = clip_boundary_and_numpy(grad.data, model.nbl)

    return objective, -grad


if __name__ == "__main__":
    TestGradient().test_gradientFWI()
