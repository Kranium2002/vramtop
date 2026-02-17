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
- **477 unit tests + 20 integration tests = 497 total** (0 failures)
- **mypy --strict**: 0 errors (51 source files)
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

### Phase 5: Export & Polish — COMPLETE
- [x] CSV export (`--export-csv FILE` CLI flag, `export/csv_logger.py`)
- [x] SVG screenshot (`s` key, saves to `~/.local/share/vramtop/screenshots/`)
- [x] Memory reporting fix: uses process-sum for used (v1 API), shows reserved as separate bar segment
- [x] GPU integration tests (14 tests with real PyTorch + NVML on RTX 2000 Ada)
- [x] CLI smoke tests (6 tests)
- [x] Design doc compliance audit (see COMPLIANCE_REPORT.md)
- [x] Survival predictor with KV cache + scrape-data awareness (`analysis/survival.py`)
- [x] PELT changepoint detection (`analysis/pelt_detector.py`, optional `ruptures` dependency)
- [x] Deep mode Unix socket IPC (`enrichment/deep_mode.py`, `reporter/protocol.py`, `reporter/pytorch.py`)

### Phase 6: Hardening & UI — IN PROGRESS
- [x] Fix survival predictor blind stable→OK (absolute headroom + spike detection)
- [x] Fix scraping never invoked (pass `scraping_config` to `enrich_process()`)
- [x] Live detail panel (auto-refreshes every poll cycle, retains data on process exit with [EXITED] tag)
- [x] Direct deep mode query in detail panel (bypasses enrichment cache for live PyTorch internals)
- [x] Space background (trig-based dot pattern via `SpaceScroll.render()`)
- [x] UI polish: gradient memory bar, color-coded sparkline, safe ASCII GPU header, GitHub-dark theme
- [x] Enrichment cache TTL tuned to 10s (static /proc data) with live bypass for deep mode
- [x] Docker PID namespace resolution (`_resolve_pid()` in enrichment, kill dialog, deep mode)
- [x] PELT wired into app (per-process timeseries accumulation, always-on in detail panel)
- [x] Labeled PELT segments with memory chart (`analysis/segment_labels.py`, `ui/widgets/memory_chart.py`)
- [ ] Pre-launch OOM risk score (predict OOM before memory fills, not just after)

## Architecture (key files)
```
src/vramtop/
├── backends/base.py              # Data types, ABC, exceptions
├── backends/nvidia.py            # NVMLClient (compute+graphics merge, v1/v2 memory handling)
├── analysis/phase_detector.py    # Variance-threshold phase detection
├── analysis/pelt_detector.py     # PELT changepoint detection (optional ruptures dependency)
├── analysis/segment_labels.py   # 18 model-agnostic segment labels, two-pass heuristic labeling
├── analysis/oom_predictor.py     # Range-based OOM prediction (GPU-level)
├── analysis/survival.py          # Per-process survival predictor (OK/TIGHT/OOM verdicts)
├── analysis/breakdown.py         # Weight vs dynamic estimation
├── analysis/trends.py            # EMA allocation rate tracker
├── enrichment/__init__.py        # Enrichment orchestrator (framework, model, container, scraping, deep mode)
├── enrichment/detector.py        # Framework detection from /proc (JAX before PyTorch)
├── enrichment/model_files.py     # Model file scanning from /proc/fd
├── enrichment/container.py       # Docker/Podman detection
├── enrichment/mps.py             # MPS daemon detection
├── enrichment/deep_mode.py       # Unix socket IPC discovery + enrichment
├── enrichment/scraper.py         # Base HTTP scraper (5 security rules)
├── enrichment/scrapers/          # vLLM, Ollama, SGLang, llama.cpp scrapers
├── ui/app.py                     # Main Textual app (layout modes, theme cycling, screenshot, live detail panel)
├── ui/widgets/memory_bar.py      # 3-segment gradient bar: used | reserved | free
├── ui/widgets/timeline.py        # Color-gradient sparkline (green→yellow→red)
├── ui/widgets/gpu_card.py        # GPU card (header, memory bar, timeline, process table, OOM alert)
├── ui/widgets/process_table.py   # Process table with phase badges + survival verdicts
├── ui/widgets/detail_panel.py    # Slide-in panel (live-updating, PELT chart+segments, deep mode, [EXITED] retention)
├── ui/widgets/memory_chart.py   # Sparkline + segment color bar + human-readable summary
├── ui/widgets/kill_dialog.py     # Kill dialog (SIGTERM→SIGKILL, audit logging, Docker PID resolution)
├── ui/widgets/space_bg.py        # SpaceScroll container with trig-based dot background
├── ui/themes/                    # 6 theme TCSS files (dark = GitHub-dark palette)
├── config.py                     # Pydantic config + SIGHUP reload
├── sanitize.py                   # ANSI/control char stripping (idempotent)
├── permissions.py                # UID checks
├── process_identity.py           # (PID, starttime) from /proc/pid/stat
├── secrets.py                    # Env var + 0600 file resolution
├── export/__init__.py            # ExportManager (CSV)
├── export/csv_logger.py          # Thread-safe CSV writer
├── export/screenshot.py          # SVG screenshot via Textual
├── reporter/__init__.py          # Deep mode reporter package
├── reporter/protocol.py          # Wire protocol (HandshakeMsg, MemoryMsg)
└── reporter/pytorch.py           # PyTorch reporter daemon thread
```

## Key Design Decisions
- Process identity uses `(PID, starttime)` tuples everywhere to prevent PID recycling issues
- NVML backend calls BOTH `GetComputeRunningProcesses` AND `GetGraphicsRunningProcesses` and merges
- **Memory reporting**: v2 API (has `reserved` field) preferred; v1 fallback uses process-sum for `used` and raw `free` for truly allocatable memory. Memory bar shows reserved as dim `╌` segment between used and free.
- Per-process GPU utilization NOT shown (broken for multi-process, confirmed by NVIDIA)
- All /proc reads are same-UID only (security boundary)
- All external strings sanitized (ANSI strip, control char removal, 256-char truncation)
- Kill flow: SIGTERM first -> 5s wait -> offer SIGKILL, with audit logging
- OOM predictions are always ranges, never point estimates
- HTTP scrapers enforce 5 rules: localhost-only, port-owner verification, rate limiting, no redirects, 64KB size limit
- **Survival predictor**: Three layers of OOM detection: (1) spike detection — if historical peak-to-trough > free memory → OOM, (2) absolute headroom — <2% free → TIGHT regardless of phase, <5% free in stable → TIGHT, (3) multiplier heuristic — framework-aware peak estimation for growing/volatile phases. Pre-allocating frameworks (vLLM, SGLang, JAX) exempt from headroom checks (they deliberately run at 95%+ utilization).
- **Detail panel**: Live-updating every poll cycle. Queries deep mode socket directly (bypasses enrichment cache). Retains data with [EXITED] tag when process dies.
- **Enrichment cache**: 10s TTL for expensive static /proc reads. Deep mode in detail panel bypasses this for 1s freshness. HTTP scrapers have their own 5s rate limiter.
- **Space background**: `SpaceScroll(VerticalScroll)` overrides `render()` to draw trig-based dot pattern. Child widgets render on top naturally. Pattern cached per (width, height).
- **Docker PID namespace**: NVML reports host PIDs inside containers. `_resolve_pid()` detects phantom PIDs (not in `/proc/`) and scans for GPU-using processes via `/dev/nvidia*` fd references. Applied in enrichment, kill dialog, and deep mode. Cached 30s.
- **PELT segment labels**: 18 model-agnostic labels via two-pass heuristic system. Pass 1: single-segment heuristics (phase, position, magnitude, duration, variance). Pass 2: multi-segment refinement (neighboring context for checkpoint saves, cooldowns, cache filling). Labels: Initialization, Pre-allocation, Warmup, Allocation Event, Memory Growth, Memory Leak, Cache Filling, Gradient Steps, Steady State, Saturation, Plateau, Idle, Batch Processing, Fragmentation, Checkpoint Save, Cleanup, Releasing, Cooldown. Saturation detection uses `gpu_total_mb`.
- **Memory chart**: Compact sparkline (single row, ▁-█ chars) + colored segment bar with numbered phases. Each segment in the summary shows: label, human description, memory delta in GB, and GPU utilization %. Designed to fit cleanly in the 46-char detail panel.
- **Saved analysis**: PELT analysis keyed by process name survives process exit, enabling post-mortem review.

## Test File Map
```
tests/
├── conftest.py                   # Mock NVML fixtures
├── fixtures/nvml_responses.py    # Fake NVML response data
├── test_nvidia_backend.py        # 21 tests: backend lifecycle, merge, errors
├── test_phase_detector.py        # 9 tests: phases, hysteresis, confidence
├── test_pelt_detector.py         # 24 tests: PELT changepoints, penalties, framework mapping, fallback
├── test_oom_predictor.py         # 8 tests: prediction rules, suppression
├── test_survival.py              # 69 tests: survival predictor, headroom, spike detection, scrape-data
├── test_sanitize.py              # 18 tests: ANSI, control chars, truncation
├── test_permissions.py           # 9 tests: UID checks
├── test_process_identity.py      # 8 tests: /proc/stat parsing
├── test_config.py                # 19 tests: TOML loading, SIGHUP, validation
├── test_tui.py                   # 5 tests: Textual pilot tests
├── test_enrichment_detector.py   # 17 tests: framework detection
├── test_enrichment_model_files.py # 12 tests: model file scanning
├── test_enrichment_container.py  # 9 tests: Docker/Podman
├── test_enrichment_mps.py        # 8 tests: MPS daemon
├── test_analysis_breakdown.py    # 9 tests: memory breakdown
├── test_segment_labels.py        # 35 tests: 18 heuristic labels, two-pass refinement, metadata
├── test_memory_chart.py          # 30 tests: sparkline chart, segment bar, summary, formatting
├── test_deep_mode.py             # 18 tests: deep mode IPC, socket discovery
├── test_reporter_pytorch.py      # 6 tests: PyTorch reporter daemon
├── test_pid_namespace.py         # 8 tests: Docker PID namespace resolution, deep mode fallback
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
- **Docker PID namespace (FIXED)**: NVML reports host PIDs inside containers, not container PIDs. `_resolve_pid()` scans `/proc/*/fd/` for `/dev/nvidia*` to find the real container PID. Applied in enrichment, kill dialog, and deep mode socket fallback. Integration tests handle this with fallback matching.
- **No v2 API on driver 550.127.05**: RTX 2000 Ada doesn't expose `nvmlDeviceGetMemoryInfo_v2`. Code falls back gracefully.
- **starttime=0 cache poisoning**: When `/proc/{pid}/stat` is unreadable, `ProcessIdentity` gets `starttime=0`. The enrichment cache now skips caching for `starttime=0` to prevent PID recycling from aliasing different processes. The framework detection cache in `detector.py` also receives starttime to prevent aliasing.
- **NVML compute+graphics double-counting (FIXED)**: A process with both CUDA compute and OpenGL graphics contexts appears in both `GetComputeRunningProcesses` and `GetGraphicsRunningProcesses`. NVML reports the SAME allocation in both lists. `nvidia.py` now uses `max()` (not sum) when merging to avoid 2x inflation.
- **Phase states keyed by ProcessIdentity (FIXED)**: `_phase_states` in `app.py` is now keyed by `(gpu_index, ProcessIdentity)` instead of `(gpu_index, pid)` to prevent PID recycling from inheriting stale phase data.
- **PyTorch caching allocator**: NVML reports the allocator's reserved pool (2-3x actual usage). Memory breakdown is labeled as "estimate" because we can't distinguish cache from active memory without deep mode data.
- **JAX detection (FIXED)**: `detector.py` now checks JAX patterns (`libxla_extension`, `libxla`, `libtpu`) BEFORE PyTorch's `libtorch` — JAX environments often also have libtorch installed.
- **Pre-allocating framework survival (FIXED)**: `estimate_peak()` now uses `process_used * 1.05` for pre-allocating frameworks even when `model_size_bytes` IS known (KV cache pool size depends on GPU, not model).
- **Cache pruning (FIXED)**: `_prune_dead_processes()` in `app.py` removes entries from `_enrichment_cache`, `_phase_detectors`, `_phase_states`, `_peak_memory` for processes no longer in the snapshot.
- **Enrichment event loop (FIXED)**: `_enrich_processes()` now runs in `asyncio.to_thread()` to avoid blocking the Textual event loop with /proc reads.
- **Breakdown confidence (FIXED)**: File-size-based weight estimates capped at 0.5 confidence (down from 0.7). File sizes don't reliably reflect in-memory sizes due to compression and quantization.
- **PELT VOLATILE classification (FIXED)**: `classify_segments()` now detects VOLATILE segments (mixed-sign deltas with high variance).
- **Raw NVML exceptions in snapshot (FIXED)**: `nvmlSystemGetDriverVersion`, `nvmlSystemGetNVMLVersion`, and `nvmlDeviceGetMemoryInfo` (v1 fallback) now wrapped in try/except to translate through `_translate_nvml_error` instead of escaping as raw `NVMLError`.
- **Framework detector starttime=0 caching (FIXED)**: `detector.py` now skips its own `_cache` when `starttime=0`, matching the enrichment cache fix in `app.py`.
- **SIGKILL audit accuracy (FIXED)**: `kill_dialog.py` SIGKILL handler now records `process_gone` or `permission_denied` instead of always `sent`.
- **Scraper rate-limit on failure (FIXED)**: Rate-limit timestamp is cleared on network failure so transient errors don't throttle the next retry.
- **Deep-mode socket read cap (FIXED)**: `read_deep_data()` enforces a 64 KB byte cap to prevent same-UID memory pressure from large payloads.
- **OOM min_rate_mb_per_sec default (FIXED)**: Changed from 5.0 to 1.0 to match design doc specification. The old value of 5.0 suppressed warnings for gradual memory growth.
- **Survival predictor blind stable→OK (FIXED)**: Previously `phase == "stable"` returned OK unconditionally — no headroom check. A process using 97.5% of GPU (405 MiB free on 16 GB) got "OK stable". Fixed with three checks: (1) spike detection if `peak - current > free`, (2) critical headroom if `free < 2%`, (3) stable + low if `free < 5%`. Pre-allocating frameworks exempt.
- **Scraping never invoked (FIXED)**: `enrich_process()` accepted `scraping_config` but `app.py` never passed it. HTTP scrapers were dead code. Now passes `self.config.scraping`.
- **Detail panel was one-shot (FIXED)**: Pressing `d` rendered once and never updated. Now auto-refreshes every poll cycle while visible. Deep mode queried directly (bypasses enrichment cache). Shows `[EXITED]` when process dies instead of vanishing.

## Design Rules (Prevent Logical Errors)

These rules MUST be followed when modifying analysis, enrichment, or survival code:

### 1. Pre-Allocation Awareness
**Rule**: Frameworks that pre-allocate memory pools (vLLM, SGLang, TGI, JAX) report `process_used` that already includes the pool. Applying a multiplier > 1x on already-allocated memory double-counts.
- `_PRE_ALLOCATING_FRAMEWORKS` in `survival.py` tracks these frameworks
- ALWAYS use `process_used * 1.05` for pre-allocating frameworks, even when `model_size_bytes` IS known
- KV cache pool size is proportional to remaining GPU memory, NOT model size — `model_size * 1.8x` is wrong
- When `model_size_bytes` IS known for non-pre-allocating frameworks, use `model_size * multiplier`

### 2. Scrape Data Freshness
**Rule**: Scrape data (HTTP metrics from inference servers) has its own 5s rate limiter. It can be stale. Don't treat it as real-time truth.
- Survival predictor returns scrape-based verdicts early (most accurate when fresh)
- Peak tracking and collective pressure operate independently of scrape data
- Never assume scrape data and NVML data are from the same instant

### 3. PID Identity Safety
**Rule**: `starttime=0` means "identity unknown". Never cache data keyed on `(pid, 0)` because PID recycling will alias different processes.
- Enrichment cache in `app.py` skips caching when `starttime=0`
- Kill dialog MUST re-verify identity before sending signals
- Phase detectors are keyed by full `ProcessIdentity`, not just PID

### 4. Framework-Specific Memory Patterns
**Rule**: Different frameworks have fundamentally different memory patterns. Never apply generic assumptions.
- **PyTorch**: Caching allocator creates sawtooth noise. `reserved >> allocated >> active`. Phase detector may see "volatile" even when training is stable. Training has bursty allocation (forward/backward spikes).
- **JAX**: Pre-allocates ~90% on first computation. Goes 0→90% in one sample. OOM prediction is meaningless for JAX startup.
- **vLLM/SGLang**: Pre-allocate KV cache pool at startup. KV cache usage % from metrics is the real signal, not NVML `process_used`. Exempt from absolute headroom checks.
- **Ollama**: Model loads fail fast (before model is fully in VRAM). Scrape data `/api/ps` only shows already-loaded models — can't predict OOM for new loads.
- **TGI**: Same pre-allocation pattern as vLLM. Uses internal KV cache pool.

### 5. Memory Accounting Consistency
**Rule**: v1 and v2 NVML APIs report different values for `used`. Code MUST be consistent about which one it uses.
- `device.used_memory_bytes`: v2 = app-allocated, v1 = process-sum (excludes driver overhead)
- `device.free_memory_bytes`: Always from NVML's raw `free` (truly allocatable)
- `device.total_memory_bytes`: Always exact
- Survival predictor uses `gpu_free_bytes` for headroom checks — this is correct regardless of v1/v2

### 6. Multi-Process Collective Safety
**Rule**: Individual process predictions can all be "OK" while the collective memory demand exceeds GPU total.
- `check_collective_pressure()` sums estimated peaks and compares to GPU total
- If overcommitted, upgrades OK→TIGHT and TIGHT→OOM
- This catches scenarios where 3 training jobs each think they have enough headroom

### 7. No False Precision
**Rule**: Never show point estimates for predictions. Always ranges or qualitative verdicts.
- OOM predictor: always `seconds_low`/`seconds_high` range
- Survival predictor: qualitative OK/TIGHT/OOM with reason string
- Memory breakdown: labeled as "estimate", never "exact"

### 8. Absolute Headroom Floor
**Rule**: Never return OK for a non-pre-allocating process when GPU free memory is dangerously low, regardless of phase.
- Spike detection: if `peak_used - current_used > gpu_free` → OOM (training loops spike during forward/backward)
- Critical floor: if `free < 2% of total` → TIGHT (any phase)
- Stable floor: if `free < 5% of total` → TIGHT (even stable phase)
- Pre-allocating frameworks are EXEMPT (they deliberately run at 95%+ utilization)
- `gpu_total_bytes` parameter is required for these checks; if 0, checks are skipped for backward compatibility

### 9. Enrichment Cache Architecture
**Rule**: The enrichment cache (10s TTL) protects against expensive /proc reads. Dynamic data sources bypass it.
- Static data (framework, model files, container) → cached for 10s, changes rarely
- Deep mode (PyTorch internals) → detail panel queries socket directly every 1s, bypasses cache
- HTTP scraping → has its own 5s rate limiter in `BaseScraper`, independent of enrichment cache
- Never increase enrichment cache TTL above 10s — delays deep mode socket discovery for new processes

### 10. UI Emoji Safety
**Rule**: Never use multi-byte emoji characters (🌡, ⊞, etc.) in widget rendering — they cause variable-width rendering across terminals.
- GPU card header uses ASCII separators (`|`) not Unicode box-drawing or emoji
- Phase badges use safe Unicode: `●`, `▲`, `▼`, `◆`
- Verdict badges use safe Unicode: `✓`, `⚠`, `✗`
- Memory bar uses safe box-drawing: `━`, `╌`, `─`, `├`, `┤`
