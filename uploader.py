import click
from tqdm import tqdm
from azureio import upload_file_to_blob, create_container
from azure.common import AzureMissingResourceHttpError


def upload_with_progressbar(filename, container, blob_name):
    with tqdm(total=100) as pbar:
        def update(done, total):
            pbar.update(done/total)
        upload_file_to_blob(filename, container, blob_name, progress_callback=update)


@click.command()
@click.option('--filename', help='The file to upload')
@click.option('--container', help="Name of the container to upload to")
@click.option('--blob-name', default=None, help="Name of the blob(file) in storage")
def run(filename, container, blob_name):
    try:
        upload_with_progressbar(filename, container, blob_name)
    except AzureMissingResourceHttpError:
        print("Container %s does not exist. Creating..." % container)
        create_container(container)
        print("Created container. Trying again")
        upload_with_progressbar(filename, container, blob_name)


if __name__ == '__main__':
    run()
