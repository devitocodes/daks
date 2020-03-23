from argparse import ArgumentParser
from util import from_hdf5
import numpy as np
import h5py
import os
from devito import Function, TimeFunction, clear_cache
from examples.seismic import AcquisitionGeometry, Receiver
from examples.checkpointing.checkpoint import (CheckpointOperator,
                                               DevitoCheckpoint)
from simple import overthrust_setup
from scipy.optimize import minimize, Bounds
from util import Profiler
from pyrevolve import Revolver
from azureio import load_blob_to_hdf5
from distributed import Client, wait


filename = "overthrust_3D_initial_model_2D.h5"
tn = 4000
nshots = 10
nbpml = 40
profiler = Profiler()
print("Connecting to scheduler")
client = Client('51.11.43.137:8786')
print("Uploading python modules")
print("util.py")
client.upload_file('util.py')
print("azureio.py")
client.upload_file('azureio.py')
print("simple.py")
client.upload_file('simple.py')

def load_shot(shot_id):
    with load_blob_to_hdf5("shots", "shot_%s.h5"%str(shot_id)) as f:
        data = f["data"][()]
        src_coords = f['src_coords'][()]
    return data, src_coords

def fwi_gradient(vp_in, model, geometry, *args):
    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)
    vp = Function(name="vp", grid=model.grid)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    objective = 0.

    with profiler.get_timer('reshape', 'vec2mat'):
        vp_in = vec2mat(vp_in)

    assert(model.vp.shape == vp_in.shape)
    vp.data[:] = vp_in[:]

    with profiler.get_timer('solve', 'setup'):
        solver = overthrust_setup(path_prefix+"/"+filename, datakey="m0")
    u0 = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                      save=geometry.nt)
    for i in range(nshots):
        true_d, source_location = load_shot(i, path_prefix)

        # Update source location
        solver.geometry.src_positions[0, :] = source_location[:]

        # Compute smooth data and full forward wavefield u0
        u0.data[:] = 0.

        smooth_d, _, _ = solver.forward(vp=vp, save=True, u=u0)

        # Compute gradient from data residual and update objective function
        residual.data[:] = smooth_d.data[:] - true_d[:]

        objective += .5*np.linalg.norm(residual.data.ravel())**2
        solver.gradient(rec=residual, u=u0, vp=vp, grad=grad)
    # grad.data[:] /= np.max(np.abs(grad.data[:]))
    return objective, -np.ravel(grad.data).astype(np.float64)


def fwi_gradient_single_shot(vp_in, shot_id):
    model_data = load_blob_to_hdf5("models", filename)
    model = from_hdf5(model_data, datakey="m0", dtype=np.float32, space_order=2,
                  nbpml=40)
    grad = Function(name="grad", grid=model.grid)
    vp = Function(name="vp", grid=model.grid)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    vp_in = vec2mat(vp_in)

    assert(model.vp.shape == vp_in.shape)
    vp.data[:] = vp_in[:]

    true_d, src_coords = load_shot(shot_id)

    solver = overthrust_setup(model_data, src_coordinates=src_coords,
                                  datakey="m0", tn=tn, nbpml=nbpml)

    u0 = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                      save=geometry.nt)

    smooth_d, _, _ = solver.forward(vp=vp, save=True, u=u0)

    residual.data[:] = smooth_d.data[:] - true_d[:]

    objective = .5*np.linalg.norm(residual.data.ravel())**2
    solver.gradient(rec=residual, u=u0, vp=vp, grad=grad)

    return objective, grad.data[nbpml:-nbpml, nbpml:-nbpml]


def fwi_gradient_dask(vp_in, *args):
    futures = []
    for i in range(nshots):
        futures.append(client.submit(fwi_gradient_single_shot, np.array(vp_in), i))

    wait(futures)

    objective = 0.
    grad = np.zeros_like(vp_in)
    for f in futures:
        o, g = f.result()
        objective += o
        grad += g

    return objective, -np.ravel(grad).astype(np.float64)


def fwi_gradient_checkpointed(vp_in, model, geometry, n_checkpoints=1000,
                              compression_params=None):
    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)
    vp = Function(name="vp", grid=model.grid)
    smooth_d = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    objective = 0.
    time_order = 2
    with profiler.get_timer('reshape', 'vec2mat'):
        vp_in = vec2mat(vp_in)

    assert(model.vp.shape == vp_in.shape)
    vp.data[:] = vp_in[:]
    # Creat forward wavefield to reuse to avoid memory overload
    with profiler.get_timer('solve', 'setup'):
        solver = overthrust_setup(path_prefix+"/"+filename, datakey="m0")
    dt = solver.dt
    nt = smooth_d.data.shape[0] - 2
    u = TimeFunction(name='u', grid=model.grid, time_order=time_order,
                     space_order=4)
    v = TimeFunction(name='v', grid=model.grid, time_order=time_order,
                     space_order=4)
    with profiler.get_timer('solve', 'setup'):
        fwd_op = solver.op_fwd(save=False)
        rev_op = solver.op_grad(save=False)
        cp = DevitoCheckpoint([u])
    for i in range(nshots):
        true_d, source_location = load_shot(i, path_prefix)
        with profiler.get_timer('solve', 'reset'):
            # Update source location
            solver.geometry.src_positions[0, :] = source_location[:]

            # Compute smooth data and full forward wavefield u0
            u.data[:] = 0.
            residual.data[:] = 0.
            v.data[:] = 0.
            smooth_d.data[:] = 0.
        with profiler.get_timer('solve', 'setup'):
            wrap_fw = CheckpointOperator(fwd_op, src=solver.geometry.src, u=u,
                                             rec=smooth_d, vp=vp, dt=dt)
            wrap_rev = CheckpointOperator(rev_op, vp=vp, u=u, v=v, rec=residual,
                                              grad=grad, dt=dt)
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt,
                               compression_params=compression_params)
        with profiler.get_timer('solve', 'forward'):
            wrp.apply_forward()

        with profiler.get_timer('solve', 'process'):
            # Compute gradient from data residual and update objective function
            residual.data[:] = smooth_d.data[:] - true_d[:]

            objective += .5*np.linalg.norm(residual.data.ravel())**2
        with profiler.get_timer('solve', 'reverse'):
            wrp.apply_reverse()
        print(wrp.profiler.summary())
    # grad.data[:] /= np.max(np.abs(grad.data[:]))
    return objective, -np.ravel(grad.data).astype(np.float64)


# Global to help write unique filenames when writing out intermediate results
iter = 0
global_ofile = ""


def apply_box_constraint(vp):
    tv = False
    vp = np.clip(vp, 1.4, 6.1)
    if tv:
        vp = denoise_tv_chambolle(vp, weight=50)
    return vp


def mat2vec(mat):
    return np.ravel(mat)


def vec2mat(vec):
    if vec.shape == model.vp.shape:
        return vec
    return np.reshape(vec, model.vp.shape)


def fncallback(vec):
    global iter
    iter += 1
    global global_ofile
    filename = "%s_%d.h5" % (global_ofile, iter)
    with profiler.get_timer('io', 'write_progress'):
        to_hdf5(vec2mat(vec), filename)
    print(profiler.summary())


def verify_equivalence_checkpointed():
    result1 = fwi_gradient_checkpointed(mat2vec(model.vp.data), model,
                                        geometry)

    result2 = fwi_gradient(mat2vec(model.vp.data), model, geometry)

    for r1, r2 in zip(result1, result2):
        np.testing.assert_allclose(r2, r1, rtol=0.01, atol=1e-8)


def verify_equivalence_dask():
    print("Remote gradient")
    result1 = fwi_gradient_dask(mat2vec(model.vp.data))
    print("Local gradient")
    result2 = fwi_gradient(mat2vec(model.vp.data), model, geometry)

    for r1, r2 in zip(result1, result2):
        np.testing.assert_allclose(r2, r1, rtol=0.01, atol=1e-8) 
    print("Equivalence verified")

def initial_setup(path_prefix):
    model_data = load_blob_to_hdf5("models", filename)
    model = from_hdf5(model_data, datakey="m0",
                      dtype=np.float32, space_order=2, nbpml=nbpml)
    spacing = model.spacing
    shape = model.vp.shape
    nrec = shape[0]
    src_coordinates = np.empty((1, len(spacing)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    if len(shape) > 1:
        src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    if len(shape) > 1:
        rec_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
        rec_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]

    # Create solver object to provide relevant operator
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=tn, src_type='Ricker', f0=0.008)

    vmax = np.ones(model.vp.shape) * 6.5
    vmin = np.ones(model.vp.shape) * 1.3

    vmax[:, 0:20+nbpml] = model.vp.data[:, 0:20+nbpml]
    vmin[:, 0:20+nbpml] = model.vp.data[:, 0:20+nbpml]
    b = Bounds(mat2vec(vmin), mat2vec(vmax))

    return model, geometry, b


if __name__ == "__main__":

    description = ("Example script for running a complete FWI")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--ncp", default=1000, type=int)
    parser.add_argument("--compression", choices=[None, 'zfp', 'sz', 'blosc'],
                        default=None)
    parser.add_argument("--tolerance", default=6, type=int)
    parser.add_argument("--nbpml", default=40,
                        type=int,
                        help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")

    parser.add_argument("--checkpointing", default=False, action='store_true',
                        help="Enable checkpointing")

    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLE) mode")

    parser.add_argument("--filename", default="fwi_final_field", type=str,
                        help="Filename for the final solution")
    args = parser.parse_args()
    compression_params = {'scheme': args.compression,
                          'tolerance': 10**(-args.tolerance)}
    
    if args.checkpointing:
        f_g = fwi_gradient_checkpointed
    else:
        f_g = fwi_gradient

    output_filename = args.filename

    global_ofile = output_filename
    path_prefix = os.path.dirname(os.path.realpath(__file__))

    model, geometry, b = initial_setup(path_prefix)

    verify_equivalence_dask()
    import sys
    sys.exit(0)
    solution_object = minimize(f_g,
                               mat2vec(model.vp.data),
                               args=(model, geometry, args.ncp,
                                     compression_params),
                               jac=True, method='L-BFGS-B',
                               callback=fncallback, bounds=b,
                               options={'disp': True, 'maxiter': 60})

    final_model = vec2mat(solution_object.x)

    true_model = from_hdf5(path_prefix+"/"+"overthrust_3D_true_model_2D.h5",
                           datakey="m", dtype=np.float32, space_order=2,
                           nbpml=nbpml)

    error_norm = np.linalg.norm(true_model.vp.data - final_model)
    print(error_norm)

    data = {'error_norm': error_norm,
            'checkpointing': args.checkpointing,
            'compression': args.compression,
            'tolerance': args.tolerance,
            'ncp': args.ncp}

    write_results(data, "fwi_experiment.csv")

    to_hdf5(final_model, '%s_final.h5' % output_filename)
