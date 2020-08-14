import click
import matplotlib.pyplot as plt
from util.results import read_objective_function_values


@click.command()
@click.option('--results', '-r', multiple=True)
@click.option('--ofile', type=str, default="relative.pdf")
def run(results, ofile):
    for r in results:
        obj_fn_values = read_objective_function_values(r)

        max_value = max(obj_fn_values)

        obj_fn_values = [float(x/max_value) for x in obj_fn_values]

        plt.plot(obj_fn_values, label=r)

    plt.xlabel('Iteration number')
    plt.ylabel('Relative objective function value')
    plt.title('Relative reduction in objective function for %s' % str(results))
    plt.legend()
    plt.savefig(ofile)


if __name__ == "__main__":
    run()
