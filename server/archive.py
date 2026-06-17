"""
Keep-forever archive abstraction for raw submission files.

The VM moves each scored submission's raw prediction files out of the transient R2 loading
dock into a keep-forever archive, recording an opaque `archive_pointer` URI + per-file content
hash in metadata.yaml. Nothing outside this module should ever construct or parse a concrete
archive path — callers deal only in (model_id, val_strategy, filename) keys and opaque pointers.

Backend selection (env):
    ARCHIVE_BACKEND   "filesystem" (default) | "dropbox"
    ARCHIVE_BASE      root of the archive. PLACEHOLDER until deploy.
                        filesystem: a directory, e.g. /srv/flux-archive
                        dropbox:    a Dropbox path, e.g. /FluxArchive
    DROPBOX_TOKEN     (dropbox backend only) OAuth2 access token

Layout under the base is always:  {model_id}_val_{strategy}/{filename}

Typical use (Phase 4):
    backend = get_archive_backend()
    key = submission_key(model_id, val_strategy, filename)
    pointer = backend.store(local_path, key)          # -> opaque URI for metadata.yaml
    ...
    backend.retrieve(pointer, dest, expected_sha256)  # re-score: fetch back from archive
"""

import hashlib
import os
import shutil
from abc import ABC, abstractmethod

# Placeholder roots — overridden by ARCHIVE_BASE at deploy time.
DEFAULT_FS_BASE = "/srv/flux-archive"
DEFAULT_DROPBOX_BASE = "/FluxArchive"


def sha256_of(path, chunk=1 << 20):
    """Content hash used as the per-file integrity check in metadata.yaml."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def submission_key(model_id, val_strategy, filename):
    """Canonical archive key for one raw file: {model_id}_val_{strategy}/{filename}."""
    return f"{model_id}_val_{val_strategy}/{os.path.basename(filename)}"


class ArchiveError(RuntimeError):
    pass


class ArchiveBackend(ABC):
    """Stores/retrieves raw files by opaque key, returning opaque URI pointers."""

    scheme = None  # e.g. "file" / "dropbox"

    @abstractmethod
    def store(self, local_path, key):
        """Copy `local_path` into the archive under `key`. Return an opaque pointer URI."""

    @abstractmethod
    def retrieve(self, pointer, dest_path, expected_sha256=None):
        """Fetch the file at `pointer` to `dest_path`. Verify hash if given."""

    @abstractmethod
    def exists(self, pointer):
        """True if the object referenced by `pointer` is present in the archive."""

    def _verify(self, path, expected_sha256):
        if expected_sha256 is not None:
            got = sha256_of(path)
            if got != expected_sha256:
                raise ArchiveError(
                    f"Hash mismatch for {path}: expected {expected_sha256}, got {got}"
                )


class FilesystemArchiveBackend(ArchiveBackend):
    """Local or network-mounted filesystem (the default; e.g. campus research storage)."""

    scheme = "file"

    def __init__(self, base=None):
        self.base = os.path.abspath(base or os.environ.get("ARCHIVE_BASE", DEFAULT_FS_BASE))

    def _path_for_key(self, key):
        return os.path.join(self.base, key)

    def _path_from_pointer(self, pointer):
        if not pointer.startswith("file://"):
            raise ArchiveError(f"Not a filesystem pointer: {pointer}")
        return pointer[len("file://"):]

    def store(self, local_path, key):
        dest = self._path_for_key(key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(local_path, dest)
        return f"file://{dest}"

    def retrieve(self, pointer, dest_path, expected_sha256=None):
        src = self._path_from_pointer(pointer)
        if not os.path.exists(src):
            raise ArchiveError(f"Archive object not found: {src}")
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        shutil.copy2(src, dest_path)
        self._verify(dest_path, expected_sha256)

    def exists(self, pointer):
        try:
            return os.path.exists(self._path_from_pointer(pointer))
        except ArchiveError:
            return False


class DropboxArchiveBackend(ArchiveBackend):
    """Dropbox/Drive fallback. Functional when the `dropbox` SDK + DROPBOX_TOKEN are present.

    Kept structurally complete so the archive can be relocated by flipping ARCHIVE_BACKEND;
    no concrete path handling leaks to callers.
    """

    scheme = "dropbox"

    def __init__(self, base=None, token=None):
        self.base = (base or os.environ.get("ARCHIVE_BASE", DEFAULT_DROPBOX_BASE)).rstrip("/")
        self._token = token or os.environ.get("DROPBOX_TOKEN")
        self._client = None

    def _dbx(self):
        if self._client is None:
            try:
                import dropbox  # lazy: not a hard dependency of the repo
            except ImportError as e:
                raise ArchiveError("dropbox SDK not installed (`pip install dropbox`)") from e
            if not self._token:
                raise ArchiveError("DROPBOX_TOKEN is not set")
            self._client = dropbox.Dropbox(self._token)
        return self._client

    def _remote_path(self, key):
        return f"{self.base}/{key}"

    def _path_from_pointer(self, pointer):
        if not pointer.startswith("dropbox:"):
            raise ArchiveError(f"Not a dropbox pointer: {pointer}")
        return pointer[len("dropbox:"):]

    def store(self, local_path, key):
        import dropbox
        remote = self._remote_path(key)
        with open(local_path, "rb") as f:
            self._dbx().files_upload(f.read(), remote, mode=dropbox.files.WriteMode.overwrite)
        return f"dropbox:{remote}"

    def retrieve(self, pointer, dest_path, expected_sha256=None):
        remote = self._path_from_pointer(pointer)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        self._dbx().files_download_to_file(dest_path, remote)
        self._verify(dest_path, expected_sha256)

    def exists(self, pointer):
        import dropbox
        try:
            self._dbx().files_get_metadata(self._path_from_pointer(pointer))
            return True
        except dropbox.exceptions.ApiError:
            return False


_BACKENDS = {
    "filesystem": FilesystemArchiveBackend,
    "dropbox": DropboxArchiveBackend,
}


def get_archive_backend():
    """Construct the backend named by ARCHIVE_BACKEND (default: filesystem)."""
    name = os.environ.get("ARCHIVE_BACKEND", "filesystem").lower()
    if name not in _BACKENDS:
        raise ArchiveError(f"Unknown ARCHIVE_BACKEND={name!r}; choose one of {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


if __name__ == "__main__":
    # Self-test: round-trip a file through the filesystem backend in a temp dir.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ARCHIVE_BACKEND"] = "filesystem"
        os.environ["ARCHIVE_BASE"] = os.path.join(tmp, "archive")
        src = os.path.join(tmp, "spatial-easy40_GPP_demo_val_mean_predictions.csv")
        with open(src, "w") as f:
            f.write("y_true,y_pred,env,site_id,time\n1.0,1.1,AR-CCg,AR-CCg,2020-01-01 00:00:00\n")
        digest = sha256_of(src)

        backend = get_archive_backend()
        key = submission_key("demo", "mean", os.path.basename(src))
        pointer = backend.store(src, key)
        assert backend.exists(pointer), "stored object should exist"

        out = os.path.join(tmp, "fetched.csv")
        backend.retrieve(pointer, out, expected_sha256=digest)
        assert sha256_of(out) == digest, "round-trip hash mismatch"
        print(f"OK  backend={backend.scheme}  key={key}\n    pointer={pointer}\n    sha256={digest}")
