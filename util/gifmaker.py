import glob
import click
from natsort import natsorted
import os


@click.command()
@click.option("--path", type=str)
def run(path):
    files = glob.glob(path + "*.pdf")

    for f in natsorted(files):
        pdf_to_png(f)

    os.system("convert -delay 40 -loop 0 %s/pngs/*.png %s/thealmighty.gif" % (path, path))


def pdf_to_png(path):
    dirname, pdfname = path.split("/")
    target_dir = dirname + "/pngs"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    basename = pdfname.split(".")[0]
    pngname = "%s/pngs/%s.png" % (dirname, basename)

    os.system("convert -density 300 -quality 95 %s %s" % (path, pngname))


if __name__ == "__main__":
    run()
