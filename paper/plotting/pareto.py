import click
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

from util.results import read_objective_function_values


@click.command()
@click.option('--results', '-r', multiple=True)
@click.option('--ofile', type=str, default="pareto.pdf")
def run(results, ofile):

    for r in results:
        name = r.split("/")[-1]
        obj_fn, model_error = process_curve(r)

        plt.plot(obj_fn, model_error, label=name)

    plt.legend()
    plt.xlabel('L2 Data residual')
    plt.ylabel('L1 model error')
    plt.savefig(ofile)


def process_curve(dir_name):
    with h5py.File("overthrust_3D_true_model_2D.h5", "r") as f:
        true_model = f['m'][()]
    model_errors = []
    for s in natsorted(glob.glob("%s/intermediates/*.h5" % dir_name)):
        with h5py.File(s, "r") as f:
            intermediate = f['data'][()]
        model_errors.append(np.linalg.norm(intermediate.ravel() - true_model.ravel(), ord=1))

    obj_fn_values = read_objective_function_values(dir_name)
    return obj_fn_values, model_errors


if __name__ == "__main__":
    run()
