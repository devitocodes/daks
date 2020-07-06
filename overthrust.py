import numpy as np
import h5py

from examples.seismic import AcquisitionGeometry, Model
from examples.seismic.acoustic import AcousticWaveSolver

from azureio import load_blob_to_hdf5
from fwiio import Blob
from solvers import DensityWaveSolver, DensityModel


def overthrust_solver_iso(h5_file, kernel='OT2', tn=4000, src_coordinates=None,
                          space_order=2, datakey='m0', nbl=40, dtype=np.float32):
    model = overthrust_model_iso(h5_file, datakey, space_order, nbl, dtype)

    geometry = create_geometry(model, tn, src_coordinates)

    solver = AcousticWaveSolver(model, geometry, kernel=kernel,
                                space_order=space_order, dtype=dtype)
    return solver


def overthrust_model_iso(h5_file, datakey, space_order, nbl, dtype):
    model_params = from_hdf5(h5_file, datakey, space_order=space_order, nbl=nbl,
                             dtype=dtype, bcs="damp")

    return Model(**model_params)


def overthrust_model_density(h5_file, datakey, space_order, nbl, dtype):
    model_params = from_hdf5(h5_file, datakey, space_order=space_order,
                             nbl=nbl, dtype=dtype, bcs="damp")
    data_vp = model_params['vp']
    data_rho = 0.31 * (1e3*data_vp)**0.25
    data_irho = 1/data_rho
    model_params['irho'] = data_irho

    return DensityModel(**model_params)


def overthrust_solver_density(h5_file, tn=4000, src_coordinates=None,
                              space_order=2, datakey='m0', nbl=40, dtype=np.float32):
    model = overthrust_model_density(h5_file, datakey, space_order, nbl, dtype)

    geometry = create_geometry(model, tn, src_coordinates)

    solver = DensityWaveSolver(model, geometry, space_order=space_order)
    return solver


def create_geometry(model, tn, src_coordinates=None):
    shape = model.shape
    spacing = model.spacing
    nrec = shape[0]

    if src_coordinates is None:
        src_coordinates = np.empty((1, len(spacing)))

        src_coordinates[0, :] = np.array(model.domain_size) * .5
        if len(shape) > 1:
            src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]
    elif len(src_coordinates.shape) == 1:
        src_coordinates = np.expand_dims(src_coordinates, axis=0)

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    if len(shape) > 1:
        rec_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
        rec_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=tn, src_type='Ricker', f0=0.008).resample(1.75)

    return geometry


def from_hdf5(f, datakey, **kwargs):
    if isinstance(f, Blob):
        f = load_blob_to_hdf5(f)
        close = True
    elif not isinstance(f, h5py.File):
        f = h5py.File(f, 'r')
        close = True
    else:
        close = False

    if datakey is None:
        raise ValueError("datakey must be known - what is the name of the" +
                         "data in the file?")
    model_params = kwargs

    dtype = model_params.get('dtype')

    data_m = f[datakey][()]
    data_vp = np.sqrt(1/data_m).astype(dtype)

    if len(data_vp.shape) > 2:
        data_vp = np.transpose(data_vp, (1, 2, 0))
    else:
        data_vp = np.transpose(data_vp, (1, 0))

    model_params['vp'] = data_vp

    model_params['shape'] = data_vp.shape

    if "origin" not in model_params:
        origin_key = model_params.pop('origin_key', 'o')
        model_params['origin'] = f[origin_key][()]

    if "spacing" not in model_params:
        spacing_key = model_params.pop('spacing_key', 'd')
        model_params['spacing'] = f[spacing_key][()]

    if close:
        f.close()

    return model_params
