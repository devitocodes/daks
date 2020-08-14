import os
import csv


filename = "my-apps.tsv"

os.system("az ad app list --show-mine -o tsv > %s" % filename)

with open(filename, 'r') as listfile:
    reader = csv.reader(listfile, delimiter='\t')
    for r in reader:
        app_id = r[29]
        os.system("az ad app delete --id %s" % app_id)

os.remove(filename)
