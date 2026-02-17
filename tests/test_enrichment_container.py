"""Tests for container detection (Docker, Podman, cgroup)."""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from vramtop.enrichment.container import (
    ContainerInfo,
    detect_container,
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
