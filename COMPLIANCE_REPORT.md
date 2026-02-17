# vramtop Design Doc Compliance Report

**Audit date:** 2026-02-17
**Auditor:** compliance-auditor (automated)
**Design doc:** `/root/vramtop/DESIGN_DOC.md`
**Codebase root:** `/root/vramtop/src/vramtop/`

---

## Section 2 -- NVML Backend

**Files:** `backends/nvidia.py`, `backends/base.py`

- [x] Abstract `GPUBackend` ABC in `base.py` (~30 lines as specified) -- `base.py:87-111`
- [x] `NVMLClient` class implements `GPUBackend` -- `nvidia.py:106`
- [x] Calls BOTH `GetComputeRunningProcesses` AND `GetGraphicsRunningProcesses` -- `nvidia.py:225-236`
- [x] Merges/deduplicates by PID; process in both lists gets `"compute+graphics"` type -- `nvidia.py:238-273`
- [x] Uses `nvmlDeviceGetMemoryInfo_v2` with v1 fallback -- `nvidia.py:203-212`
- [x] `GPU_IS_LOST` translated to `GPULostError` -- `nvidia.py:75-76`
- [x] `UNKNOWN` retries 3x with exponential backoff, then raises `DriverError` -- `nvidia.py:80-103`
- [x] Process name resolution with same-UID check via `is_same_user()` -- `nvidia.py:30-56`
- [x] `nvmlInit`/`nvmlShutdown` lifecycle managed -- `nvidia.py:112-131`
- [x] Process identity `(PID, starttime)` used via `_get_identity()` -- `nvidia.py:59-69`
- [x] Exception hierarchy: `VramtopError > BackendError > GPULostError / DriverError` -- `base.py:15-28`
- [x] Frozen, slotted data types (`ProcessIdentity`, `GPUProcess`, `GPUDevice`, `MemorySnapshot`) -- `base.py:38-81`

**Verdict: FULLY COMPLIANT**

---

## Section 3 -- Phase Detection

**File:** `analysis/phase_detector.py`

- [x] `Phase` enum with STABLE, GROWING, SHRINKING, VOLATILE -- `phase_detector.py:10-14`
- [x] `PhaseState` dataclass with `phase`, `duration_samples`, `rate_mb_per_sec`, `confidence` -- `phase_detector.py:17-22`
- [x] Sliding window with `deque(maxlen=window_size)` -- `phase_detector.py:48`
- [x] Default `window_size=10`, `hysteresis_samples=3`, `noise_floor_mb=1.0`, `variance_threshold=4.0`, `ema_alpha=0.3` -- `phase_detector.py:41-46`
- [x] Variance-threshold classification logic matches design doc exactly -- `phase_detector.py:73-79`
- [x] Hysteresis: M consecutive samples before transition -- `phase_detector.py:82-94`
- [x] EMA-smoothed rate -- `phase_detector.py:97`
- [x] Confidence formula: `min(1.0, duration/30.0) * (1.0/(1.0 + variance/100.0))` -- `phase_detector.py:99-101`
- [x] Returns `PhaseState(Phase.STABLE, 0, 0.0, 0.0)` when `len(window) < 3` -- `phase_detector.py:64-65`

**Verdict: FULLY COMPLIANT -- implementation matches design doc pseudocode exactly**

---

## Section 4 -- OOM Prediction

**File:** `analysis/oom_predictor.py`

- [x] `OOMPrediction` dataclass with `seconds_low`, `seconds_high`, `confidence`, `display` -- `oom_predictor.py:11-16`
- [x] `rate_range_mb_per_sec` field for min/max EMA rate tracking -- `oom_predictor.py:16`
- [x] Rule 1: Only predicts during GROWING phase -- `oom_predictor.py:45-49`
- [x] Rule 2: Requires `>min_sustained_samples` sustained samples -- `oom_predictor.py:66-67`
- [x] Rule 3: Range estimate, not point estimate -- `oom_predictor.py:83-89`
- [x] Rule 5: Capped at ">1h" -- `oom_predictor.py:93-99`
- [x] Rule 6: Suppressed if rate < `min_rate_mb_per_sec` -- `oom_predictor.py:70-71`
- [x] Rate range tracking (min/max EMA over GROWING phase) -- `oom_predictor.py:52-61`
- [x] Default `min_sustained_samples=10`, `min_rate_mb_per_sec=1.0` -- `oom_predictor.py:32-33`

**Verdict: FULLY COMPLIANT**

---

## Section 5 -- Enrichment

**Files:** `enrichment/__init__.py`, `enrichment/detector.py`, `enrichment/model_files.py`, `enrichment/container.py`, `enrichment/mps.py`

- [x] Framework detection from `/proc/<pid>/cmdline` -- `detector.py:59-72`
- [x] Framework detection from `/proc/<pid>/maps` (loaded libraries) -- `detector.py:75-88`
- [x] `_CMDLINE_PATTERNS`: ollama, vllm, sglang, tgi, llamacpp -- `detector.py:15-21`
- [x] `_MAPS_PATTERNS`: pytorch (libtorch), jax (libjax), tensorflow (libtensorflow) -- `detector.py:24-28`
- [x] Same-UID enforcement on framework detection -- `detector.py:48-49`
- [x] Model file scanning from `/proc/<pid>/fd` via readlink -- `model_files.py:17-64`
- [x] Model file extensions: `.safetensors`, `.gguf`, `.pt`, `.bin`, `.onnx`, `.pth`, `.h5`, `.tflite` -- `model_files.py:10-12`
- [x] Same-UID enforcement on model file scanning -- `model_files.py:23-24`
- [x] Container detection: Docker (`/.dockerenv`) -- `container.py:46-48`
- [x] Container detection: Podman (`/run/.containerenv`) -- `container.py:51-53`
- [x] Container detection: cgroup parsing from `/proc/1/cgroup` -- `container.py:55-58, 68-79`
- [x] `NVIDIA_VISIBLE_DEVICES` env reading (own process only) -- `container.py:38-40`
- [x] MPS detection via process scan for `nvidia-cuda-mps-control` -- `mps.py:44-55`
- [x] MPS badge: `is_mps_client()` checks same-UID -- `mps.py:18-26`
- [x] Enrichment orchestrator: all enrichers in try/except, failure of one doesn't block others -- `enrichment/__init__.py:38-118`
- [x] Same-UID enforcement at top of `enrich_process()` -- `enrichment/__init__.py:48-49`

**Minor deviation:** CDI-based GPU assignment (`/var/run/nvidia-container-devices/`) is not implemented. Design doc mentions it but it's a minor detection path.

**Verdict: SUBSTANTIALLY COMPLIANT (1 minor omission: CDI detection)**

---

## Section 6 -- TUI

**Files:** `ui/app.py`, `ui/themes/__init__.py`, `ui/widgets/kill_dialog.py`, `ui/widgets/phase_badge.py`, `ui/widgets/alerts.py`, `ui/widgets/timeline.py`, `ui/widgets/detail_panel.py`, `ui/widgets/process_table.py`, `ui/widgets/gpu_card.py`, `ui/widgets/memory_bar.py`

### Themes
- [x] 6 themes: dark, light, nord, catppuccin, dracula, solarized -- `ui/themes/__init__.py:16-23` + 6 `.tcss` files confirmed

### Layout Modes
- [x] FULL layout mode -- `ui/app.py:51`
- [x] COMPACT layout mode -- `ui/app.py:52`
- [x] MINI layout mode -- `ui/app.py:53`
- [x] TOO_SMALL mode (terminal < 40x10) -- `ui/app.py:54, 78-79`
- [x] Auto layout resolution based on terminal size -- `ui/app.py:56-79`
- [x] `_apply_layout()` adjusts widget visibility per mode -- `ui/app.py:241-286`

### Accessibility & NO_COLOR
- [x] `NO_COLOR` env var respected -- `ui/app.py:149`
- [x] Accessible mode flag (`--accessible` CLI arg) -- `cli.py:42-47`, `ui/app.py:148`
- [x] Phase badge supports accessible mode with text descriptions -- `ui/widgets/phase_badge.py:25-30, 67-68`
- [x] Phase badge supports NO_COLOR (text-only fallback) -- `ui/widgets/phase_badge.py:55, 69-70`
- [x] `color_disabled` property combines NO_COLOR and accessible -- `ui/app.py:168-170`

### Kill Dialog
- [x] Kill dialog triggered by `[k]` key -- `ui/app.py:110`
- [x] Confirmation dialog with process age -- `ui/widgets/kill_dialog.py:186-213`
- [x] Same-UID check before allowing kill -- `ui/widgets/kill_dialog.py:167-181`
- [x] SIGTERM first -- `ui/widgets/kill_dialog.py:244-308`
- [x] 5-second wait, then offer SIGKILL -- `ui/widgets/kill_dialog.py:29, 308, 310-332`
- [x] Audit log at `~/.local/share/vramtop/audit.log` -- `ui/widgets/kill_dialog.py:26`
- [x] Audit log directory created with `0o700` permissions -- `ui/widgets/kill_dialog.py:33-39`
- [x] `--no-kill` flag / `config.display.enable_kill` support -- `ui/widgets/kill_dialog.py:157-163`
- [x] PID identity re-verified before kill (anti-PID-recycling) -- `ui/widgets/kill_dialog.py:229-242, 248`

### Phase Badges
- [x] Phase symbols: `\u25ac` STABLE, `\u25b2` GROWING, `\u25bc` SHRINKING, `\u301c` VOLATILE -- `ui/widgets/phase_badge.py:11-16`
- [x] Text fallback for NO_COLOR -- `ui/widgets/phase_badge.py:18-23`

### OOM Alerts
- [x] Bold red `!!! OOM` display -- `ui/widgets/alerts.py:38`
- [x] No blinking (confirmed: no blink styles anywhere in the codebase)

### Sparkline Timeline
- [x] Timeline widget with sparkline characters -- `ui/widgets/timeline.py:9, 45-62`
- [x] History deque with configurable maxlen -- `ui/widgets/timeline.py:31`

### Detail Panel
- [x] Detail panel widget showing process info -- `ui/widgets/detail_panel.py:16-160`
- [x] Triggered by `[d]` key -- `ui/app.py:109`
- [x] Shows PID, command, framework, container, MPS, VRAM, phase, rate, OOM -- `ui/widgets/detail_panel.py:84-141`
- [x] Escape to close -- `ui/widgets/detail_panel.py:48-49`

**Verdict: FULLY COMPLIANT**

---

## Section 7 -- Scraping

**Files:** `enrichment/scraper.py`, `enrichment/scrapers/__init__.py`, `enrichment/scrapers/vllm.py`, `enrichment/scrapers/ollama.py`, `enrichment/scrapers/sglang.py`, `enrichment/scrapers/llamacpp.py`

### 5 Security Rules
- [x] Rule 1: Localhost only (hardcoded `http://127.0.0.1:{port}`) -- `scraper.py:108`
- [x] Rule 1 extended: `_is_localhost()` with IP validation -- `scraper.py:126-135`
- [x] Rule 1 extended: `allowed_ports` enforcement -- `scraper.py:114-123`
- [x] Rule 2: Port owner PID verification via `psutil.net_connections()` -- `scraper.py:138-158`
- [x] Rule 3: Per-endpoint, per-PID rate limiting -- `scraper.py:160-172`
- [x] Rule 4: No redirect following (`_NoRedirectHandler`) -- `scraper.py:49-63`
- [x] Rule 5: 500ms timeout default, 64KB response size limit -- `scraper.py:30-31, 205-209`

### 4 Framework Scrapers
- [x] vLLM scraper (`/metrics`, Prometheus text format) -- `scrapers/vllm.py:28-57`
- [x] Ollama scraper (`/api/ps`, JSON + Pydantic validation) -- `scrapers/ollama.py:42-69`
- [x] SGLang scraper (`/get_model_info`, JSON + Pydantic validation) -- `scrapers/sglang.py:25-47`
- [x] llama.cpp scraper (`/metrics`, Prometheus text format) -- `scrapers/llamacpp.py:27-60`

### Port Detection
- [x] Port detection via `psutil.net_connections()` -- `scrapers/__init__.py:51-87`
- [x] Default ports per framework (vllm:8000, ollama:11434, sglang:8000, llamacpp:8080) -- `scrapers/__init__.py:58-63`

**Verdict: FULLY COMPLIANT**

---

## Section 8 -- Config

**File:** `config.py`

### Config Sections with Correct Fields and Defaults
- [x] `[general]` section: `refresh_rate=1.0`, `history_length=300` -- `config.py:39-43`
- [x] `[alerts]` section: `oom_warning_seconds=300`, `temp_warning_celsius=85`, `oom_min_confidence=0.3` -- `config.py:54-59`
- [x] `[display]` section: `theme="dark"`, `layout="auto"`, `show_other_users=false`, `enable_kill=true`, `phase_indicator=true` -- `config.py:62-69`
- [x] `[scraping]` section: `enable=true`, `rate_limit_seconds=5`, `timeout_ms=500`, `allowed_ports=[8000,8080,11434,3000]` -- `config.py:72-78`
- [x] `[oom_prediction]` section: `algorithm="variance"`, `min_sustained_samples=10`, `min_rate_mb_per_sec=1.0` -- `config.py:81-86`
- [x] `[export]` section: `prometheus_bind="127.0.0.1"`, `prometheus_port=0` -- `config.py:89-93`

### Validation
- [x] Unknown keys rejected (`extra="forbid"` on all models) -- `config.py:40, 55, 63, 73, 82, 90, 97`
- [x] Out-of-range validation: `refresh_rate` must be 0.5-10 -- `config.py:45-51`
- [x] Pydantic `BaseModel` for all config sections -- confirmed
- [x] TOML loading with `tomllib` (3.11+) / `tomli` fallback -- `config.py:20-26`

### Hot Reload
- [x] SIGHUP-triggered reload via `ConfigHolder` -- `config.py:154-199`
- [x] Signal handler sets flag, polled from event loop -- `config.py:184-194`
- [x] Reload failure keeps old config -- `config.py:176-182`

**Verdict: FULLY COMPLIANT**

---

## Section 9 -- Error Handling (10 Error Modes)

| # | Failure Mode | Status | Evidence |
|---|-------------|--------|----------|
| 1 | NVML init fails | [x] | `nvidia.py:117-123`, `cli.py:78-87`: clean error message and exit |
| 2 | GPU falls off bus (`GPU_IS_LOST`) | [x] | `nvidia.py:75-76, 88-89`: translated to `GPULostError`; `app.py:298-300, 542-548`: marks GPU as LOST |
| 3 | Driver crash (`UNKNOWN`) | [x] | `nvidia.py:80-103`: 3x retry with exponential backoff, then `DriverError` |
| 4 | Process vanishes (PID gone) | [x] | `process_identity.py:21`: returns None; `nvidia.py:64`: fallback to starttime=0 |
| 5 | PID recycled (starttime mismatch) | [x] | `kill_dialog.py:229-242`: identity re-verified before kill; `app.py:383-392`: keyed by `ProcessIdentity` |
| 6 | HTTP scrape fails | [x] | `scraper.py:183-211`: `ScrapeFailedError`; `enrichment/__init__.py:110-116`: caught, logged, continues |
| 7 | Permission denied /proc | [x] | `detector.py:48-49, 65-66, 81-82`: falls through gracefully; `model_files.py:23-24, 29`: returns empty |
| 8 | Config corrupt | [x] | `config.py:139-146`: raises `ConfigError`; `config.py:176-182`: reload keeps old config |
| 9 | Terminal too small | [x] | `app.py:56-79`: auto-switches to `TOO_SMALL` mode; `app.py:241-258`: displays message |
| 10 | Socket stale | [ ] | Not implemented -- `enrichment/deep_mode.py` and `reporter/` are stubs |

**Verdict: 9/10 IMPLEMENTED (missing: stale socket handling, blocked by deep mode not being implemented)**

---

## Section 10 -- Security

- [x] UID boundary enforcement via `permissions.is_same_user()` -- `permissions.py:8-13`
- [x] UID check used in: process name resolution (`nvidia.py:43`), framework detection (`detector.py:48`), model file scan (`model_files.py:23`), MPS detection (`mps.py:24`), kill dialog (`kill_dialog.py:167`), enrichment orchestrator (`enrichment/__init__.py:48`)
- [x] Input sanitization: ANSI stripping + control char removal -- `sanitize.py:17-24`
- [x] Sanitization is idempotent (as designed) -- `sanitize.py:20-21` (documented)
- [x] Sanitization truncates to 256 chars -- `sanitize.py:7, 24`
- [x] Audit logging for kill actions at `~/.local/share/vramtop/audit.log` -- `kill_dialog.py:26, 42-48`
- [x] Audit log directory `0o700` permissions -- `kill_dialog.py:33-39`
- [x] Secrets file refused if group/other perms set -- `secrets.py:30-31`
- [ ] Socket file permissions (`0o700` directory for IPC) -- Not implemented (deep mode is a stub)

**Verdict: SUBSTANTIALLY COMPLIANT (socket permissions N/A since deep mode not yet implemented)**

---

## File Structure Compliance

### Implemented Files (present and functional)
- [x] `__init__.py`, `__main__.py`, `cli.py` -- entry points
- [x] `config.py` -- Pydantic-validated, SIGHUP-reloadable
- [x] `secrets.py` -- env var + 0600 file
- [x] `sanitize.py` -- ANSI strip, control char removal
- [x] `permissions.py` -- UID checks
- [x] `process_identity.py` -- (PID, starttime) management
- [x] `backends/__init__.py`, `backends/base.py`, `backends/nvidia.py` -- backend layer
- [x] `enrichment/__init__.py`, `enrichment/detector.py`, `enrichment/model_files.py`, `enrichment/scraper.py` -- enrichment
- [x] `enrichment/scrapers/vllm.py`, `enrichment/scrapers/ollama.py`, `enrichment/scrapers/sglang.py`, `enrichment/scrapers/llamacpp.py` -- 4 scrapers
- [x] `enrichment/container.py`, `enrichment/mps.py` -- container/MPS detection
- [x] `analysis/__init__.py`, `analysis/phase_detector.py`, `analysis/oom_predictor.py`, `analysis/trends.py`, `analysis/breakdown.py` -- analysis
- [x] `ui/app.py`, `ui/themes/`, `ui/widgets/gpu_card.py`, `ui/widgets/memory_bar.py`, `ui/widgets/timeline.py`, `ui/widgets/process_table.py`, `ui/widgets/detail_panel.py`, `ui/widgets/alerts.py`, `ui/widgets/kill_dialog.py`, `ui/widgets/phase_badge.py` -- TUI
- [x] `export/__init__.py`, `export/csv_logger.py` -- CSV export
- [x] `reporter/__init__.py` -- placeholder

### Files from Design Doc Not Yet Implemented (Phase 5)
- [ ] `enrichment/deep_mode.py` -- Unix socket IPC (file missing entirely)
- [ ] `analysis/pelt_detector.py` -- Optional PELT (file missing)
- [ ] `reporter/pytorch.py` -- vramtop.report() for PyTorch (file missing)
- [ ] `reporter/protocol.py` -- Wire protocol definitions (file missing)
- [ ] `export/prometheus.py` -- Prometheus exporter (file missing)
- [ ] `export/json_stream.py` -- JSON stream exporter (file missing)
- [ ] `export/screenshot.py` -- Screenshot exporter (file missing)

### Test Files
- [x] `tests/conftest.py`
- [x] `tests/test_nvidia_backend.py`
- [x] `tests/test_phase_detector.py`
- [x] `tests/test_oom_predictor.py`
- [x] `tests/test_process_identity.py`
- [x] `tests/test_sanitize.py`
- [x] `tests/test_permissions.py`
- [x] `tests/test_scraper_security.py`
- [x] `tests/test_config.py`
- [x] `tests/test_tui.py`
- [x] `tests/test_property_phase.py` (Hypothesis)
- [x] `tests/test_property_oom.py` (Hypothesis)
- [x] `tests/test_property_sanitize.py` (Hypothesis)
- [x] `tests/test_enrichment_model_files.py` (extra)
- [x] `tests/test_enrichment_container.py` (extra)
- [x] `tests/test_enrichment_mps.py` (extra)
- [x] `tests/test_analysis_breakdown.py` (extra)
- [x] `tests/test_enrichment_detector.py` (extra)
- [x] `tests/test_kill_dialog.py` (extra)
- [x] `tests/test_security_boundaries.py` (extra)
- [x] `tests/test_scrapers_vllm.py` (extra)
- [x] `tests/test_scrapers_llamacpp.py` (extra)
- [x] `tests/test_scrapers_sglang.py` (extra)
- [x] `tests/test_scrapers_ollama.py` (extra)
- [x] `tests/test_export_csv.py` (extra)
- [x] `tests/integration/test_gpu_pytorch.py` (extra)
- [ ] `tests/test_ipc_protocol.py` (missing -- deep mode not implemented)
- [ ] `tests/fixtures/` -- only `nvml_responses.py` exists (partial)

---

## Summary

### Overall Compliance: Phases 1-4 FULLY IMPLEMENTED

The codebase implements all requirements from the design doc for **Phases 1 through 4** of the build sequence:

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Core | **COMPLETE** | NVML backend, process identity, memory bar, sparkline, PhaseDetector, basic OOM, sanitization, permission boundaries, error handling, Hypothesis tests |
| Phase 2: Intelligence | **COMPLETE** | Layer 1 detection, OOMPredictor with ranges, weight estimation, container detection, detail panel, kill dialog |
| Phase 3: Beauty | **COMPLETE** | 6 themes, 3+1 layout modes, accessible mode, NO_COLOR |
| Phase 4: Scraping | **COMPLETE** | vLLM/SGLang/Ollama/llama.cpp scrapers with all 5 security rules |
| Phase 5: Depth | **NOT STARTED** | Deep mode (IPC), PELT, Prometheus, JSON stream, webhooks, SSH remote, historical playback |

### What Remains (Phase 5 - Deferred / Ongoing)
1. **Deep mode** (`enrichment/deep_mode.py`, `reporter/pytorch.py`, `reporter/protocol.py`) -- Unix socket IPC not implemented
2. **PELT detector** (`analysis/pelt_detector.py`) -- optional `ruptures` dependency
3. **Export backends**: Prometheus (`export/prometheus.py`), JSON stream (`export/json_stream.py`), screenshot (`export/screenshot.py`)
4. **Stale socket handling** (error mode #10) -- depends on deep mode
5. **CDI-based GPU assignment** (`/var/run/nvidia-container-devices/`) -- minor container detection path
6. **IPC protocol tests** (`tests/test_ipc_protocol.py`)

### Deviations from Design Doc
- None for Phases 1-4. Implementation matches design doc pseudocode precisely (especially phase detector and OOM predictor).
- The `reporter/` package exists as a placeholder (`__init__.py` only) with no functionality yet.
- Test coverage exceeds design doc requirements (many extra test files for enrichment, scrapers, breakdown, kill dialog, security boundaries, and CSV export).
