"""
Transient object store = the R2 "loading dock" for in-flight submissions.

Distinct from server.archive (the keep-forever tier). The browser PUTs raw files here via the
Worker's presigned URLs; the Action GETs them to validate; the VM GETs them to score, then
DELETEs them (so R2 only ever holds in-flight submissions and stays free).

This module is the Python side (Action + VM): GET / DELETE / EXISTS by key. Presigning for
uploads is the Worker's job (JS, S3-compatible). Backends:
    OBJECTSTORE_BACKEND = "r2" (default) | "local" (filesystem, for tests/dev)

R2 env: R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
Local env: R2_LOCAL_DIR

Layout (set by the Worker): incoming/{model_id}_val_{strategy}/{filename}
"""

import os
import shutil
from abc import ABC, abstractmethod


def incoming_key(model_id, val_strategy, filename):
    """Canonical transient key for one raw file."""
    return f"incoming/{model_id}_val_{val_strategy}/{os.path.basename(filename)}"


class ObjectStoreError(RuntimeError):
    pass


class ObjectStore(ABC):
    @abstractmethod
    def get(self, key, dest_path):
        """Download object `key` to `dest_path`."""

    @abstractmethod
    def delete(self, key):
        """Delete object `key` (idempotent)."""

    @abstractmethod
    def exists(self, key):
        """True if `key` is present."""


class R2ObjectStore(ObjectStore):
    """Cloudflare R2 via its S3-compatible API (boto3)."""

    def __init__(self, endpoint=None, bucket=None, access_key=None, secret_key=None):
        self.bucket = bucket or os.environ.get("R2_BUCKET")
        self._endpoint = endpoint or os.environ.get("R2_ENDPOINT")
        self._access_key = access_key or os.environ.get("R2_ACCESS_KEY_ID")
        self._secret_key = secret_key or os.environ.get("R2_SECRET_ACCESS_KEY")
        self._client = None

    def _s3(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as e:
                raise ObjectStoreError("boto3 not installed (`pip install boto3`)") from e
            missing = [n for n, v in [("R2_ENDPOINT", self._endpoint), ("R2_BUCKET", self.bucket),
                                      ("R2_ACCESS_KEY_ID", self._access_key),
                                      ("R2_SECRET_ACCESS_KEY", self._secret_key)] if not v]
            if missing:
                raise ObjectStoreError(f"R2 config missing: {missing}")
            self._client = boto3.client(
                "s3", endpoint_url=self._endpoint, region_name="auto",
                aws_access_key_id=self._access_key, aws_secret_access_key=self._secret_key,
            )
        return self._client

    def get(self, key, dest_path):
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        self._s3().download_file(self.bucket, key, dest_path)

    def delete(self, key):
        self._s3().delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key):
        from botocore.exceptions import ClientError
        try:
            self._s3().head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


class LocalObjectStore(ObjectStore):
    """Filesystem-backed store for tests/dev (keys are paths under R2_LOCAL_DIR)."""

    def __init__(self, root=None):
        self.root = root or os.environ.get("R2_LOCAL_DIR")
        if not self.root:
            raise ObjectStoreError("R2_LOCAL_DIR not set for local object store")

    def _p(self, key):
        return os.path.join(self.root, key)

    def get(self, key, dest_path):
        src = self._p(key)
        if not os.path.exists(src):
            raise ObjectStoreError(f"object not found: {key}")
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        shutil.copy2(src, dest_path)

    def delete(self, key):
        try:
            os.remove(self._p(key))
        except FileNotFoundError:
            pass

    def exists(self, key):
        return os.path.exists(self._p(key))

    def put(self, key, src_path):
        """Test/dev helper to seed the store (the browser does this via presigned PUT in prod)."""
        dest = self._p(key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src_path, dest)


_BACKENDS = {"r2": R2ObjectStore, "local": LocalObjectStore}


def get_object_store():
    name = os.environ.get("OBJECTSTORE_BACKEND", "r2").lower()
    if name not in _BACKENDS:
        raise ObjectStoreError(f"unknown OBJECTSTORE_BACKEND={name!r}; choose {sorted(_BACKENDS)}")
    return _BACKENDS[name]()
