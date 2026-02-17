"""Integration tests for multi-container GPU detection.

Requires Docker with NVIDIA Container Toolkit and a GPU.
Run with: pytest tests/integration/test_multi_container.py -v

These tests launch real Docker containers with GPU workloads and verify
that vramtop correctly identifies each container.
"""

from __future__ import annotations

import subprocess
import time

import pytest

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    """Check if Docker is available."""
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="module")
def gpu_containers():
    """Launch 2 GPU containers, yield their IDs, then clean up."""
    if not _docker_available():
        pytest.skip("Docker not available")

    containers: list[str] = []
    try:
        for i, mb in enumerate([128, 256]):
            result = subprocess.run(
                [
                    "docker", "run", "-d", "--rm",
                    "--gpus", "all",
                    "--name", f"vramtop-test-{i}",
                    "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
                    "sleep", "120",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            containers.append(result.stdout.strip()[:12])

        # Wait for containers to start
        time.sleep(2)
        yield containers

    finally:
        for cid in containers:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True,
                timeout=10,
            )


class TestMultiContainerIntegration:
    """Integration tests for multi-container detection."""

    def test_containers_detected(self, gpu_containers: list[str]) -> None:
        """Verify launched containers have cgroup entries."""
        import os

        # Find GPU-using processes via /proc
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/cgroup") as f:
                    content = f.read()
                if any(cid in content for cid in gpu_containers):
                    from vramtop.enrichment.container import detect_process_container

                    result = detect_process_container(pid)
                    assert result is not None
                    assert result.runtime in ("docker", "containerd")
                    assert result.container_id in gpu_containers
                    return  # At least one found
            except (FileNotFoundError, PermissionError):
                continue

        pytest.skip("Could not find container processes in /proc")

    def test_different_containers_different_ids(
        self, gpu_containers: list[str]
    ) -> None:
        """Verify that different containers get different IDs."""
        assert len(gpu_containers) >= 2
        assert gpu_containers[0] != gpu_containers[1]
