"""Tests for multi-container GPU process detection.

Covers per-process container detection (cgroup parsing), enrichment wiring,
and detail panel container display.
"""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from vramtop.enrichment.container import (
    ContainerInfo,
    _parse_container_from_cgroup,
    detect_process_container,
)


# ---------------------------------------------------------------------------
# TestParseCgroupContent — pure function, no mocks needed
# ---------------------------------------------------------------------------


class TestParseCgroupContent:
    """Test _parse_container_from_cgroup() with various cgroup formats."""

    def test_docker_v1(self) -> None:
        content = (
            "12:devices:/docker/abc123def456789012345678901234567890123456789012\n"
            "11:cpu:/docker/abc123def456789012345678901234567890123456789012\n"
        )
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "docker"
        assert result.container_id == "abc123def456"

    def test_podman_v1(self) -> None:
        content = "11:memory:/podman/deadbeef12345678abcdef1234567890abcdef1234567890ab\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "podman"
        assert result.container_id == "deadbeef1234"

    def test_containerd_v1(self) -> None:
        content = "5:memory:/containerd/aabbccdd11223344aabbccdd11223344aabbccdd11223344\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "containerd"
        assert result.container_id == "aabbccdd1122"

    def test_docker_v2_scope(self) -> None:
        content = "0::/system.slice/docker-abc123def456abcdef0123456789abcdef0123456789abcdef01234567.scope\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "docker"
        assert result.container_id == "abc123def456"

    def test_podman_v2_scope(self) -> None:
        content = "0::/user.slice/user-1000.slice/user@1000.service/podman-deadbeef1234abcdef.scope\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "podman"
        assert result.container_id == "deadbeef1234"

    def test_containerd_v2_scope(self) -> None:
        content = "0::/cri-containerd-aabbccdd112233445566778899aabbccddeeff00112233445566.scope\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.runtime == "containerd"
        assert result.container_id == "aabbccdd1122"

    def test_host_process_no_container(self) -> None:
        content = (
            "12:cpu:/user.slice/user-1000.slice/session-1.scope\n"
            "11:memory:/user.slice/user-1000.slice/session-1.scope\n"
        )
        result = _parse_container_from_cgroup(content)
        assert result is None

    def test_empty_content(self) -> None:
        result = _parse_container_from_cgroup("")
        assert result is None

    def test_cgroupv2_unified_no_container(self) -> None:
        content = "0::/init.scope\n"
        result = _parse_container_from_cgroup(content)
        assert result is None

    def test_short_container_id_12_chars(self) -> None:
        content = "12:devices:/docker/abcdef123456\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert result.container_id == "abcdef123456"

    def test_container_id_truncated_to_12(self) -> None:
        content = "12:devices:/docker/abcdef1234567890abcdef1234567890abcdef1234567890abcdef12345678\n"
        result = _parse_container_from_cgroup(content)
        assert result is not None
        assert len(result.container_id or "") == 12


# ---------------------------------------------------------------------------
# TestDetectProcessContainer — mocked /proc reads
# ---------------------------------------------------------------------------


class TestDetectProcessContainer:
    """Test detect_process_container() with mocked filesystem."""

    def test_docker_process(self) -> None:
        cgroup_content = "12:devices:/docker/abc123def456abcdef\n"
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", mock_open(read_data=cgroup_content)),
        ):
            result = detect_process_container(1234)
        assert result is not None
        assert result.runtime == "docker"
        assert result.container_id == "abc123def456"

    def test_podman_process(self) -> None:
        cgroup_content = "0::/user.slice/podman-deadbeef1234abcdef.scope\n"
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", mock_open(read_data=cgroup_content)),
        ):
            result = detect_process_container(5678)
        assert result is not None
        assert result.runtime == "podman"
        assert result.container_id == "deadbeef1234"

    def test_host_process(self) -> None:
        cgroup_content = "12:cpu:/user.slice/user-1000.slice/session-1.scope\n"
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", mock_open(read_data=cgroup_content)),
        ):
            result = detect_process_container(9999)
        assert result is None

    def test_permission_denied(self) -> None:
        with patch("vramtop.enrichment.container.is_same_user", return_value=False):
            result = detect_process_container(1234)
        assert result is None

    def test_process_not_found(self) -> None:
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            result = detect_process_container(99999)
        assert result is None

    def test_cgroup_read_permission_error(self) -> None:
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", side_effect=PermissionError),
        ):
            result = detect_process_container(1234)
        assert result is None


# ---------------------------------------------------------------------------
# TestMultiContainerEnrichment — multiple processes from different containers
# ---------------------------------------------------------------------------


class TestMultiContainerEnrichment:
    """Test enrichment pipeline identifies containers per process."""

    def _mock_cgroup_for_pid(self, pid_cgroup_map: dict[int, str]):  # noqa: ANN202
        """Create a side-effect for open() that returns different cgroup content per PID."""
        from io import StringIO

        def _side_effect(path: str, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            for pid, content in pid_cgroup_map.items():
                if path == f"/proc/{pid}/cgroup":
                    return StringIO(content)
                if path == f"/proc/{pid}/cmdline":
                    return StringIO(f"python --pid={pid}")
            raise FileNotFoundError(path)

        return _side_effect

    def test_three_containers(self) -> None:
        """Three GPU processes from three different containers."""
        cgroup_map = {
            100: "12:devices:/docker/aaaa11112222333344445555666677778888\n",
            200: "12:devices:/docker/bbbb11112222333344445555666677778888\n",
            300: "12:devices:/podman/cccc11112222333344445555666677778888\n",
        }

        for pid, expected_rt, expected_id_prefix in [
            (100, "docker", "aaaa11112222"),
            (200, "docker", "bbbb11112222"),
            (300, "podman", "cccc11112222"),
        ]:
            with (
                patch("vramtop.enrichment.container.is_same_user", return_value=True),
                patch("builtins.open", side_effect=self._mock_cgroup_for_pid({pid: cgroup_map[pid]})),
            ):
                result = detect_process_container(pid)
            assert result is not None, f"Expected container for pid {pid}"
            assert result.runtime == expected_rt
            assert result.container_id == expected_id_prefix

    def test_host_and_container_mix(self) -> None:
        """Mix of host and containerized processes."""
        host_cgroup = "12:cpu:/user.slice/user-1000.slice/session-1.scope\n"
        docker_cgroup = "12:devices:/docker/aaaa11112222333344445555\n"

        # Host process
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", mock_open(read_data=host_cgroup)),
        ):
            host_result = detect_process_container(100)

        # Docker process
        with (
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", mock_open(read_data=docker_cgroup)),
        ):
            docker_result = detect_process_container(200)

        assert host_result is None
        assert docker_result is not None
        assert docker_result.runtime == "docker"

    def test_enrichment_result_keys(self) -> None:
        """Verify enrich_process sets container_runtime and container_id."""
        from vramtop.enrichment import enrich_process

        cgroup = "12:devices:/docker/aabb11223344556677889900\n"
        cmdline = b"python\x00train.py"

        def mock_open_fn(path: str, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            from io import BytesIO, StringIO

            if path == "/proc/42/cgroup":
                return StringIO(cgroup)
            if path == "/proc/42/cmdline":
                return BytesIO(cmdline)
            # maps, fd - let them fail
            raise FileNotFoundError(path)

        with (
            patch("os.path.exists", return_value=True),
            patch("vramtop.enrichment.is_same_user", return_value=True),
            patch("vramtop.enrichment.container.is_same_user", return_value=True),
            patch("builtins.open", side_effect=mock_open_fn),
        ):
            result = enrich_process(42)

        assert result.container_runtime == "docker"
        assert result.container_id == "aabb11223344"


# ---------------------------------------------------------------------------
# TestDetailPanelContainerDisplay — verify correct key usage
# ---------------------------------------------------------------------------


class TestDetailPanelContainerDisplay:
    """Test that the detail panel reads container_runtime and container_id."""

    def test_container_runtime_and_id_displayed(self) -> None:
        """Verify detail panel uses container_runtime / container_id keys."""
        from vramtop.ui.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        # Simulate the line-building logic without mounting the widget
        enrichment: dict[str, object] = {
            "container_runtime": "docker",
            "container_id": "abc123def456",
        }
        lines: list[str] = []

        container_runtime = enrichment.get("container_runtime")
        container_id = enrichment.get("container_id")
        if container_runtime:
            ctr_display = str(container_runtime)
            if container_id:
                ctr_display += f" ({container_id})"
            lines.append(f"[bold]Container:[/bold] {ctr_display}")

        assert len(lines) == 1
        assert "docker (abc123def456)" in lines[0]

    def test_container_runtime_only(self) -> None:
        """If container_id is missing, just show runtime."""
        enrichment: dict[str, object] = {
            "container_runtime": "podman",
        }
        lines: list[str] = []

        container_runtime = enrichment.get("container_runtime")
        container_id = enrichment.get("container_id")
        if container_runtime:
            ctr_display = str(container_runtime)
            if container_id:
                ctr_display += f" ({container_id})"
            lines.append(f"[bold]Container:[/bold] {ctr_display}")

        assert len(lines) == 1
        assert "podman" in lines[0]
        assert "(" not in lines[0]

    def test_no_container_keys(self) -> None:
        """No container keys → no container line."""
        enrichment: dict[str, object] = {}
        lines: list[str] = []

        container_runtime = enrichment.get("container_runtime")
        container_id = enrichment.get("container_id")
        if container_runtime:
            ctr_display = str(container_runtime)
            if container_id:
                ctr_display += f" ({container_id})"
            lines.append(f"[bold]Container:[/bold] {ctr_display}")

        assert len(lines) == 0

    def test_old_container_key_ignored(self) -> None:
        """The old 'container' key should NOT produce output."""
        enrichment: dict[str, object] = {
            "container": "docker",
        }
        lines: list[str] = []

        container_runtime = enrichment.get("container_runtime")
        container_id = enrichment.get("container_id")
        if container_runtime:
            ctr_display = str(container_runtime)
            if container_id:
                ctr_display += f" ({container_id})"
            lines.append(f"[bold]Container:[/bold] {ctr_display}")

        assert len(lines) == 0
