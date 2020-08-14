import csv
import os


def read_objective_function_values(dir_name):
    obj_fn_file = os.path.join(dir_name, "objective_function_values.csv")
    obj_fn_values = []
    with open(obj_fn_file, 'r') as objfile:
        vals_reader = csv.reader(objfile)
        for r in vals_reader:
            obj_fn_values.append(float(r[0]))
    return obj_fn_values
