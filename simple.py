import numpy as np

from argparse import ArgumentParser

from examples.seismic import (Receiver, TimeAxis, RickerSource,
                              AcquisitionGeometry)
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.tti import AnisotropicWaveSolver
from util import from_hdf5

def overthrust_setup(filename, kernel='OT2', tn=4000, src_coordinates=None,
                     space_order=2, datakey='m0', nbpml=40, dtype=np.float32,
                     **kwargs):
    model = from_hdf5(filename, space_order=space_order, nbpml=nbpml,
                      datakey=datakey, dtype=dtype)
    shape = model.vp.shape
    spacing = model.spacing
    nrec = shape[0]

    if src_coordinates is None:
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
    solver = AcousticWaveSolver(model, geometry, kernel=kernel,
                                space_order=space_order, **kwargs)
    return solver
