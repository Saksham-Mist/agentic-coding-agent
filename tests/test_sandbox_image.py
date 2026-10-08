"""Step 2.1 done-when check: the sandbox image starts fast.

Requires Docker running locally and the `agentic-sandbox:latest`
image built (see docker/sandbox/Dockerfile). Skips if either is
unavailable, so this doesn't fail CI/a fresh clone that hasn't built
the image yet.
"""
import shutil
import subprocess
import time

import pytest


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, check=True
        )
        return True
    except Exception:
        return False


def _image_built() -> bool:
    result = subprocess.run(
        ["docker", "images", "-q", "agentic-sandbox:latest"],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


@pytest.mark.skipif(not _docker_available(), reason="Docker not running")
@pytest.mark.skipif(not _image_built(), reason="agentic-sandbox:latest not built")
def test_sandbox_container_starts_under_one_second():
    start = time.monotonic()
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--network=none", "--memory=512m", "--cpus=1",
            "agentic-sandbox:latest",
        ],
        input="{}",
        capture_output=True,
        text=True,
        timeout=10,
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 0, result.stderr
    # On Docker Desktop (Windows/macOS VM layer) this lands around
    # 500-700ms; on a native Linux daemon (the GCP deploy target) it's
    # expected to be faster. 1s is the hard ceiling either way.
    assert elapsed < 1.0, f"container start took {elapsed:.3f}s, expected < 1.0s"
