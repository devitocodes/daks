import os
import pytest

from fwi.io import BlobAuth


@pytest.fixture
def auth():
    account_name = os.environ['BLOB_ACCOUNT_NAME']
    account_key = os.environ['BLOB_ACCOUNT_KEY']
    return BlobAuth(account_name, account_key)
