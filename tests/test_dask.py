from distributed import wait

from fwi.dasksetup import setup_dask


def test_dask_upload():
    client = setup_dask()

    def remote_test():
        try:
            from fwi.io import load_shot # noqa
            return True
        except:
            return False

    future = client.submit(remote_test)

    wait(future)
    result = future.result()
    assert(result)
