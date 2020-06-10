import numpy as np
from devito.tools import memoized_meth
from examples.seismic import (Receiver, PointSource, Model)
from devito import TimeFunction, Function, Eq, Operator


class DensityWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    kernel : str, optional
        Type of discretization, centered or shifted.
    space_order: int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """
    def __init__(self, model, geometry, space_order=4, **kwargs):
        self.model = model
        self.geometry = geometry

        assert self.model == geometry.model

        self.space_order = space_order

        self.dt = self.model.critical_dt

        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                                space_order=self.space_order,
                               **self._kwargs)

    def forward(self, src=None, rec=None, u=None, vp=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            Stores the computed wavefield.
        vp : Function or float, optional
            The time-constant velocity.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.geometry.time_axis,
                              coordinates=self.geometry.rec_positions)

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        # Pick vp from model unless explicitly provided
        vp = vp or self.model.vp

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec=rec, u=u, vp=vp,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, summary


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium with density. 

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """
    m, damp, irho = model.m, model.damp, model.irho

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = PointSource(name='src', grid=geometry.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = density_stencil(u, m, s, damp, irho)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src*s**2/(irho*m))

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)
    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)

def density_stencil(field, m, s, damp, irho, **kwargs):
    """
    Stencil for the acoustic wave-equation with density:

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    m : Function or float
        Square slowness.
    s : float or Scalar
        The time dimension spacing.
    damp : Function
        The damping field for absorbing boundary condition.
    forward : bool
        The propagation direction. Defaults to True.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    """
    # Define time step to be updated
    next = field.forward if kwargs.get('forward', True) else field.backward
    prev = field.backward if kwargs.get('forward', True) else field.forward
    # Get the spacial FD
    lap = laplacian(field, m, s, irho)
    # Get source
    q = kwargs.get('q', 0)
    # Define PDE and update rule
    # Bypass solve due to sympy+dask issue
    # solve(field.dt2 - H - q + damp * field.dt, next)
    eq_time = ((lap + q) * s**2 + s * damp * field +
               m * irho * (2 * field - prev))/(s * damp + m * irho)

    # return the Stencil with H replaced by its symbolic expression
    return [Eq(next, eq_time)]


def laplacian(field, m, s, irho):
    """
    Spacial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    m : Function or float
        Square slowness.
    s : float or Scalar
        The time dimension spacing.
    """
    so = irho.space_order // 2
    Lap = sum([getattr(irho * getattr(field, 'd%s'%d.name)(x0=d + d.spacing/2, fd_order=so), 'd%s'% d.name)(x0=d - d.spacing/2, fd_order=so) for d in irho.dimensions])
    
    return Lap

class DensityModel(Model):
    def __init__(self, origin, spacing, shape, space_order, vp, irho=None, nbl=20,
                 dtype=np.float32, subdomains=(), bcs="mask", grid=None, **kwargs):
        super(DensityModel, self).__init__(origin, spacing, shape, space_order, vp, nbl, dtype,
                                    subdomains, bcs, grid, **kwargs)

        self.irho = self._gen_phys_param(irho, 'irho', space_order)
