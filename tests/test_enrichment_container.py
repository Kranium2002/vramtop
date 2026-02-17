"""Tests for container detection (Docker, Podman, cgroup)."""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from vramtop.enrichment.container import (
    ContainerInfo,
    _parse_container_from_cgroup,
    detect_container,
    detect_process_container,
    get_nvidia_visible_devices,
    _reset_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset container detection cache before each test."""
    _reset_cache()
    yield
    _reset_cache()


class TestDockerDetection:
    """Test Docker container detection."""

    def test_dockerenv_present(self):
        cgroup_data = "12:devices:/docker/abc123def456\n"

        def exists_side_effect(path):
            return path == "/.dockerenv"

        with (
            patch("os.path.exists", side_effect=exists_side_effect),
            patch("builtins.open", mock_open(read_data=cgroup_data)),
        ):
            result = detect_container()

        assert result is not None
        assert result.runtime == "docker"
        assert result.container_id == "abc123def456"

    def test_dockerenv_present_no_cgroup(self):
        def exists_side_effect(path):
            return path == "/.dockerenv"

        with (
            patch("os.path.exists", side_effect=exists_side_effect),
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            result = detect_container()

        assert result is not None
        assert result.runtime == "docker"
        assert result.container_id is None


class TestPodmanDetection:
    """Test Podman container detection."""

    def test_containerenv_present(self):
        cgroup_data = "12:devices:/podman/deadbeef1234\n"

        def exists_side_effect(path):
            if path == "/.dockerenv":
                return False
            if path == "/run/.containerenv":
                return True
            return False

        with (
            patch("os.path.exists", side_effect=exists_side_effect),
            patch("builtins.open", mock_open(read_data=cgroup_data)),
        ):
            result = detect_container()

        assert result is not None
        assert result.runtime == "podman"
        assert result.container_id == "deadbeef1234"


class TestNoContainer:
    """Test non-container environment."""

    def test_no_container_markers(self):
        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            result = detect_container()

        assert result is None

    def test_cgroup_without_container_id(self):
        cgroup_data = "12:cpu:/user.slice/user-1000.slice\n"

        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open", mock_open(read_data=cgroup_data)),
        ):
            result = detect_container()

        assert result is None


class TestCaching:
    """Test that detection result is cached."""

    def test_cached_after_first_call(self):
        with (
            patch("os.path.exists", return_value=False),
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            r1 = detect_container()

        # Second call should not touch the filesystem
        with (
            patch("os.path.exists", side_effect=AssertionError("should not call")),
        ):
            r2 = detect_container()

        assert r1 == r2


class TestNvidiaVisibleDevices:
    """Test NVIDIA_VISIBLE_DEVICES reading."""

    def test_env_present(self):
        with patch.dict("os.environ", {"NVIDIA_VISIBLE_DEVICES": "0,1"}):
            assert get_nvidia_visible_devices() == "0,1"

    def test_env_absent(self):
        with patch.dict("os.environ", {}, clear=True):
            assert get_nvidia_visible_devices() is None


class TestCgroupV2Parsing:
    """Test cgroupv2 scope-based container detection."""

    def test_docker_v2_scope(self) -> None:
        content = "0::/system.slice/docker-abcdef123456abcdef123456.scope\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "docker"
        assert result.container_id == "abcdef123456"

    def test_podman_v2_scope(self) -> None:
        content = "0::/user.slice/podman-deadbeef1234abcdef.scope\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "podman"
        assert result.container_id == "deadbeef1234"

    def test_containerd_v2_scope(self) -> None:
        content = "0::/cri-containerd-aabbccdd112233445566.scope\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "containerd"
        assert result.container_id == "aabbccdd1122"

    def test_v2_no_container(self) -> None:
        content = "0::/init.scope\n"
        result = _parse_container_from_cgroup(content)
        assert result is None


class TestPerProcessContainerDetection:
    """Test detect_process_container() per-process cgroup detection."""

    def test_docker_process(self) -> None:
        cgroup = "12:devices:/docker/aabbccdd112233445566778899\n"
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", mock_open(read_data=cgroup)),
        ):
            result = detect_process_container(1234)
        assert result is not None
        assert result.runtime == "docker"

    def test_host_process(self) -> None:
        cgroup = "12:cpu:/user.slice/user-1000.slice\n"
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", mock_open(read_data=cgroup)),
        ):
            result = detect_process_container(5678)
        assert result is None

    def test_different_uid_rejected(self) -> None:
        with patch("vramtop.enrichment.container.is_same_user", return_value=False):
            result = detect_process_container(1234)
        assert result is None

    def test_process_gone(self) -> None:
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", side_effect=ProcessLookupError),
        ):
            result = detect_process_container(1234)
        assert result is None
