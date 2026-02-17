# vramtop — Implementation Progress

## Project Overview
vramtop is "the htop for GPU memory" — an NVIDIA-first GPU memory monitoring TUI.
See `DESIGN_DOC.md` for the full architecture and design specification.
See `COMPLIANCE_REPORT.md` for detailed design-doc compliance audit.

## Build/Test Commands
```bash
pip install -e ".[dev]"          # Install with dev deps
pytest tests/ -m "not gpu" -v    # Unit tests (no GPU needed)
pytest tests/integration/ -v     # GPU integration tests (requires NVIDIA GPU)
pytest tests/test_property_*.py --hypothesis-show-statistics -v  # Property tests
mypy --strict src/vramtop/       # Type checking
ruff check src/                  # Linting
bandit -r src/                   # Security scan
python -m vramtop --help         # CLI help
python -m vramtop --version      # Version check
```

## Quality Gate Status (current)
- **255 unit tests + 20 integration tests = 275 total** (0 failures)
- **mypy --strict**: 0 errors (43 source files)
- **ruff check**: All checks passed
- **bandit**: 0 high/medium
- **CLI**: `vramtop 0.1.0` working

## Implementation Status

### Phase 1: Core — COMPLETE
- [x] Commit 1: Project scaffolding (pyproject.toml, package stubs, CI, LICENSE)
- [x] Commit 2: Core data types (ProcessIdentity, GPUProcess, GPUDevice, MemorySnapshot), backend ABC, exceptions
- [x] Commit 3: Utility modules (sanitize.py, permissions.py, process_identity.py, secrets.py) + tests + Hypothesis property tests
- [x] Commit 4: NVIDIA backend (NVMLClient) with compute+graphics merge, retry logic, fixtures + tests
- [x] Commit 5: Phase detector (variance-threshold + hysteresis), OOM predictor (range-based), trends + tests + Hypothesis
- [x] Commit 6: Pydantic config with SIGHUP reload + tests
- [x] Commit 7: TUI (GPU cards, memory bar, sparkline timeline, process table, phase badges, OOM alerts, CLI) + pilot tests

### Phase 2: Intelligence — COMPLETE
- [x] Commit 8: Enrichment data model (EnrichmentResult, ModelFileInfo) + orchestrator
- [x] Commit 9: Framework detection (cmdline + maps patterns, 30s TTL cache) + model file scanning
- [x] Commit 10: Container detection (Docker/Podman/cgroup) + MPS daemon detection
- [x] Commit 11: Memory breakdown estimation + OOM predictor rate range enhancement
- [x] Commit 12: Detail panel (slide-in process info) + kill dialog (SIGTERM->SIGKILL, audit log)
- [x] Commit 13: Wired enrichment into app poll loop + enrichment tests
- [x] Commit 14: Security boundary tests (UID enforcement, sanitize, audit permissions)

### Phase 3: Beauty — COMPLETE
- [x] Commit 15: Theme system + 6 themes (dark, light, nord, catppuccin, dracula, solarized)
- [x] Commit 16: Layout modes (FULL/COMPACT/MINI) + terminal size auto-detection
- [x] Commit 17: Accessibility (NO_COLOR, --accessible, text fallbacks)
- [x] Commit 18: Visual polish (number formatting, truncation, loading states, footer bar, theme cycling with 't')

### Phase 4: Scraping — COMPLETE (MVP DONE)
- [x] Commit 19: Base HTTP scraper with 5 security rules (localhost-only, port verification, rate limit, no redirects, size limit)
- [x] Commit 20: Framework scrapers (vLLM, Ollama, SGLang, llama.cpp)
- [x] Commit 21: Wire scrapers into enrichment pipeline
- [x] Commit 22: Scraper security tests (25 tests) + per-framework parsing tests (30 tests)
- [x] Commit 23: Final quality gate — all checks clean

### Phase 5: Export & Polish — IN PROGRESS
- [x] CSV export (`--export-csv FILE` CLI flag, `export/csv_logger.py`)
- [x] SVG screenshot (`s` key, saves to `~/.local/share/vramtop/screenshots/`)
- [x] Memory reporting fix: uses process-sum for used (v1 API), shows reserved as separate bar segment
- [x] GPU integration tests (14 tests with real PyTorch + NVML on RTX 2000 Ada)
- [x] CLI smoke tests (6 tests)
- [x] Design doc compliance audit (see COMPLIANCE_REPORT.md)

## Architecture (key files)
```
src/vramtop/
├── backends/base.py              # Data types, ABC, exceptions
├── backends/nvidia.py            # NVMLClient (compute+graphics merge, v1/v2 memory handling)
├── analysis/phase_detector.py    # Variance-threshold phase detection
├── analysis/oom_predictor.py     # Range-based OOM prediction
├── analysis/breakdown.py         # Weight vs dynamic estimation
├── analysis/trends.py            # EMA allocation rate tracker
├── enrichment/__init__.py        # Enrichment orchestrator
├── enrichment/detector.py        # Framework detection from /proc
├── enrichment/model_files.py     # Model file scanning from /proc/fd
├── enrichment/container.py       # Docker/Podman detection
├── enrichment/mps.py             # MPS daemon detection
├── enrichment/scraper.py         # Base HTTP scraper (5 security rules)
├── enrichment/scrapers/          # vLLM, Ollama, SGLang, llama.cpp scrapers
├── ui/app.py                     # Main Textual app (layout modes, theme cycling, screenshot)
├── ui/widgets/memory_bar.py      # 3-segment bar: used (app) | reserved (driver) | free
├── ui/widgets/                   # GPU card, process table, detail panel, kill dialog, etc.
├── ui/themes/                    # 6 theme TCSS files
├── config.py                     # Pydantic config + SIGHUP reload
├── sanitize.py                   # ANSI/control char stripping (idempotent)
├── permissions.py                # UID checks
├── process_identity.py           # (PID, starttime) from /proc/pid/stat
├── secrets.py                    # Env var + 0600 file resolution
├── export/__init__.py            # ExportManager (CSV)
├── export/csv_logger.py          # Thread-safe CSV writer
├── export/screenshot.py          # SVG screenshot via Textual
└── reporter/                     # Deep mode reporter (placeholder)
```

## Key Design Decisions
- Process identity uses `(PID, starttime)` tuples everywhere to prevent PID recycling issues
- NVML backend calls BOTH `GetComputeRunningProcesses` AND `GetGraphicsRunningProcesses` and merges
- **Memory reporting**: v2 API (has `reserved` field) preferred; v1 fallback uses process-sum for `used` and raw `free` for truly allocatable memory. Memory bar shows reserved as dim `▒` segment between used and free.
- Per-process GPU utilization NOT shown (broken for multi-process, confirmed by NVIDIA)
- All /proc reads are same-UID only (security boundary)
- All external strings sanitized (ANSI strip, control char removal, 256-char truncation)
- Kill flow: SIGTERM first -> 5s wait -> offer SIGKILL, with audit logging
- OOM predictions are always ranges, never point estimates
- HTTP scrapers enforce 5 rules: localhost-only, port-owner verification, rate limiting, no redirects, 64KB size limit

## Test File Map
```
tests/
├── conftest.py                   # Mock NVML fixtures
├── fixtures/nvml_responses.py    # Fake NVML response data
├── test_nvidia_backend.py        # 21 tests: backend lifecycle, merge, errors
├── test_phase_detector.py        # 9 tests: phases, hysteresis, confidence
├── test_oom_predictor.py         # 8 tests: prediction rules, suppression
├── test_sanitize.py              # 18 tests: ANSI, control chars, truncation
├── test_permissions.py           # 9 tests: UID checks
├── test_process_identity.py      # 8 tests: /proc/stat parsing
├── test_config.py                # 19 tests: TOML loading, SIGHUP, validation
├── test_tui.py                   # 5 tests: Textual pilot tests
├── test_enrichment_detector.py   # 17 tests: framework detection
├── test_enrichment_model_files.py # 12 tests: model file scanning
├── test_enrichment_container.py  # 8 tests: Docker/Podman
├── test_enrichment_mps.py        # 8 tests: MPS daemon
├── test_analysis_breakdown.py    # 9 tests: memory breakdown
├── test_kill_dialog.py           # 16 tests: kill safety, audit
├── test_security_boundaries.py   # 9 tests: UID enforcement
├── test_scraper_security.py      # 25 tests: 5 scraper security rules
├── test_scrapers_vllm.py         # 9 tests: Prometheus parsing
├── test_scrapers_ollama.py       # 7 tests: JSON /api/ps
├── test_scrapers_sglang.py       # 6 tests: JSON model info
├── test_scrapers_llamacpp.py     # 8 tests: Prometheus parsing
├── test_export_csv.py            # 8 tests: CSV logger
├── test_export_screenshot.py     # 5 tests: SVG screenshot
├── test_property_sanitize.py     # 4 Hypothesis tests
├── test_property_phase.py        # 3 Hypothesis tests
├── test_property_oom.py          # 2 Hypothesis tests
└── integration/
    ├── test_gpu_pytorch.py       # 14 tests: real GPU (NVML, process detection, CSV, phases)
    └── test_cli_smoke.py         # 6 tests: CLI flags, CSV export with real data
```

## Known Issues / Gotchas
- **NVML v1 memory inflation**: `nvmlDeviceGetMemoryInfo` (v1) lumps driver-reserved memory (~300 MB) into `used`. Fixed by using process-sum for `used_memory_bytes` when v2 API unavailable. `free_memory_bytes` always comes from NVML's `free` (truly allocatable).
- **Docker PID namespace**: NVML reports host PIDs inside containers, not container PIDs. Integration tests handle this with fallback matching.
- **No v2 API on driver 550.127.05**: RTX 2000 Ada doesn't expose `nvmlDeviceGetMemoryInfo_v2`. Code falls back gracefully.
