import click
import h5py
import numpy as np
import os
from fwi.overthrust import overthrust_model_iso
from util import plot_model_to_file


@click.command()
@click.option('--results', '-r')
@click.option('--ofile', type=str, default="model_residual.pdf")
def run(results, ofile):
    with h5py.File("overthrust_3D_true_model_2D.h5", "r") as f:
        true_model = f['m'][()]

    final_model_file = os.path.join(results, 'final_solution.h5')
    with h5py.File(final_model_file, "r") as f:
        final_model = f['data'][()]

    true_model = np.transpose(true_model)
    residual = true_model - final_model

    print(np.max(residual), np.min(residual), np.linalg.norm(residual), np.linalg.norm(residual, ord=1), residual.shape)

    model = overthrust_model_iso("overthrust_3D_true_model_2D.h5", "m")
    model.update("vp", residual)
    plot_model_to_file(model, ofile, cmap="RdBu")


if __name__ == "__main__":
    run()
