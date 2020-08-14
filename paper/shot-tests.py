import numpy as np
from fwi.overthrust import overthrust_solver_iso, overthrust_model_iso, overthrust_solver_density
from fwi.io import Blob
from examples.seismic import plot_shotrecord
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt  # noqa: E402


initial_model_filename = "overthrust_3D_initial_model_2D.h5"
true_model_filename = "overthrust_3D_true_model_2D.h5"
so = 16
nbl = 40
dtype = np.float32
tn = 4000
dt = 1.75
true_model = overthrust_model_iso(Blob("models", true_model_filename), datakey="m", space_order=so, nbl=nbl, dtype=dtype)

initial_model = overthrust_model_iso(Blob("models", initial_model_filename), datakey="m0", space_order=so, nbl=nbl,
                                     dtype=dtype)

true_model_solver_density = overthrust_solver_density(Blob("models", true_model_filename), datakey="m", tn=tn, space_order=so,
                                                      nbl=nbl, dtype=dtype)

true_model_solver_iso = overthrust_solver_iso(Blob("models", true_model_filename), datakey="m", tn=tn, space_order=so, nbl=nbl,
                                              dtype=dtype)

initial_model_solver_iso = overthrust_solver_iso(Blob("models", initial_model_filename), datakey="m0", tn=tn, space_order=so,
                                                 nbl=nbl, dtype=dtype)

# True with density
rec_true_dens, _, _ = true_model_solver_density.forward(dt=dt)
plot_shotrecord(rec_true_dens.data, true_model, t0=0, tn=tn, cmap="PuOr")
plt.savefig("true_with_density_so16.pdf")

# True no density
rec_true_no_dens, _, _ = true_model_solver_iso.forward(dt=dt)
plt.clf()
plot_shotrecord(rec_true_no_dens.data, true_model, t0=0, tn=tn, cmap="PuOr")
plt.savefig("true_no_density_so16.pdf")

# True diff density
diff = rec_true_dens.data - rec_true_no_dens.data
plt.clf()
plot_shotrecord(diff, true_model, t0=0, tn=tn, cmap="PuOr")
plt.savefig("diff_true_dens_no_dens_so16.pdf")

# Init no density
rec_init_no_density, _, _ = initial_model_solver_iso.forward(dt=dt)
plt.clf()
plot_shotrecord(rec_init_no_density.data, true_model, t0=0, tn=tn, cmap="PuOr")
plt.savefig("init_no_density_so16.pdf")

# Diff true dens iso init
diff_true_dens_init_iso = rec_true_dens.data - rec_init_no_density.data
plt.clf()
plot_shotrecord(diff_true_dens_init_iso, true_model, t0=0, tn=tn, cmap="PuOr")
plt.savefig("diff_true_dens_init_iso_so16.pdf")
