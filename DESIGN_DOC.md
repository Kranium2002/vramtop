# vramtop v3 — NVIDIA-Only Architecture & Implementation Plan

> The htop for GPU memory. Beautiful. Zero-friction. NVIDIA-first.
> **Revision 3: NVIDIA-only, ship-ready.**

---

## Scope Decision

**NVIDIA only for v1.0.** AMD, Apple Silicon, and Intel Arc are deferred to a future major version. The backend abstraction layer remains in the codebase (it's just good architecture), but only the NVIDIA backend is implemented, tested, and documented.

**Rationale:** NVIDIA is >95% of the ML training/inference GPU market. Shipping one excellent backend beats shipping four mediocre ones. The abstraction layer means adding vendors later is additive, not a rewrite.

---

## Changelog from Original Plan

| Area | Change | Rationale |
|------|--------|-----------|
| Scope | Multi-vendor → NVIDIA only | Ship faster, validate core, add vendors later |
| NVML binding | `pynvml` → `nvidia-ml-py` | `pynvml` officially deprecated (Sep 2025). `nvidia-ml-py` is NVIDIA's canonical package (v13.590+). |
| Per-process GPU util | Removed entirely | `nvmlDeviceGetProcessUtilization` is broken for multi-process (confirmed by NVIDIA engineer). Only accurate for single-process-per-GPU. Showing wrong numbers is worse than not showing them. |
| OOM prediction | PELT → Two-stage: variance-threshold + optional PELT | PELT requires penalty tuning per-framework. Variance threshold is simpler, faster, sufficient for MVP. |
| Process identity | PID → `(PID, starttime)` tuple | PID recycling on loaded servers happens in <5s. `/proc/<pid>/stat` field 22 provides unique identity. |
| IPC protocol | Unspecified → Line-delimited JSON over Unix domain socket | Fault isolation, cleanup on crash, protocol versioning. |
| Secrets storage | `keyring` → Env vars + 0600 file | `keyring` fails on headless GPU servers (`NoKeyringError`). |
| Config reload | Startup-only → SIGHUP + watchdog file watcher | Long-running TUI needs hot-reload. |
| Testing | Unit + fixtures → Add Hypothesis property-based testing | Edge cases (NaN, zero-length, negative deltas) that fixtures miss. |

---

## 1. Core Philosophy

1. **Process-agnostic** — Works on any NVIDIA GPU process (PyTorch, JAX, vLLM, Ollama, Blender, games, anything).
2. **Zero-instrumentation by default** — `pip install vramtop && vramtop`.
3. **Secure by default** — No privilege escalation. No data leakage. No destructive actions without confirmation.
4. **Honestly documented** — Every metric labeled as exact vs estimate. No marketing claims that can't be verified in 10 minutes.

---

## 2. Data Source Architecture

### 2.1 Architecture

```
┌─────────────────────────────────────────────────┐
│                  vramtop TUI                    │
├─────────────────────────────────────────────────┤
│              Unified Data Model                 │
│  GPUDevice / GPUProcess / MemorySnapshot        │
├─────────────────────────────────────────────────┤
│            NVIDIA Backend (NVML)                │
│  via nvidia-ml-py (official, v13.590+)          │
├─────────────────────────────────────────────────┤
│         Abstract GPUBackend interface           │
│    (for future AMD/Apple/Intel — not now)        │
└─────────────────────────────────────────────────┘
```

The abstract `GPUBackend` interface exists in `base.py` as a ~30 line ABC. It costs nothing and means adding AMD later is a new file, not a refactor.

### 2.2 What NVML Gives Us

| Metric | API | Accuracy | Notes |
|--------|-----|----------|-------|
| **Total VRAM** | `nvmlDeviceGetMemoryInfo_v2` | ✅ Exact | |
| **Used VRAM (device)** | `nvmlDeviceGetMemoryInfo_v2` | ✅ Exact | |
| **Per-process VRAM** | `GetComputeRunningProcesses` + `GetGraphicsRunningProcesses` | ✅ Exact | Must call BOTH and merge/dedup by PID |
| **Device GPU util %** | `nvmlDeviceGetUtilizationRates` | ⚠️ Time-based | "% of time at least one kernel running" — not SM occupancy. Labeled honestly. |
| **Memory BW util %** | `nvmlDeviceGetUtilizationRates` | ⚠️ Time-based | Same caveat |
| **Temperature** | `nvmlDeviceGetTemperature` | ✅ Exact | |
| **Power draw** | `nvmlDeviceGetPowerUsage` | ✅ Exact | milliwatts |
| **Clock speeds** | `nvmlDeviceGetClockInfo` | ✅ Exact | |
| **PCIe throughput** | `nvmlDeviceGetPcieThroughput` | ✅ Exact | |
| **Per-process GPU util** | `nvmlDeviceGetProcessUtilization` | ❌ **Broken** | Only works single-process-per-GPU. Confirmed by NVIDIA. **Not shown.** |
| **MPS client detection** | MPS control daemon query | ⚠️ Best-effort | |

### 2.3 NVML Binding Strategy

**Use `nvidia-ml-py` (v13+) directly.**

```python
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo, nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetGraphicsRunningProcesses, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature, nvmlDeviceGetPowerUsage,
    nvmlDeviceGetName, nvmlDeviceGetUUID,
    NVML_TEMPERATURE_GPU,
)
```

Wrapped in an `NVMLClient` class that:
- Manages `nvmlInit`/`nvmlShutdown` lifecycle
- Translates NVML error codes → vramtop exceptions
- Merges compute + graphics process lists with dedup
- Isolates the dependency to one file for fixture-based testing

### 2.4 Process Identity — PID Recycling Defense

On loaded GPU servers, PIDs recycle in <5 seconds. Use `(PID, starttime)` as the unique process key.

```python
def get_process_identity(pid: int) -> tuple[int, int] | None:
    """Return (pid, starttime) or None if process doesn't exist."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            stat = f.read()
        end_comm = stat.rfind(')')
        fields = stat[end_comm + 2:].split()
        starttime = int(fields[19])  # field 22, index 19 after comm
        return (pid, starttime)
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None
```

Applied everywhere:
- Process table keys are `(pid, starttime)` not just `pid`
- VRAM attribution validated: if starttime changes, old data discarded
- Kill confirmation shows process age as additional safety check

### 2.5 Why We Don't Show Per-Process GPU Utilization

`nvmlDeviceGetProcessUtilization` exists but is broken for multi-process. When two processes share a GPU, it assigns all utilization to one and reports 0% for the other. Confirmed by NVIDIA engineer on their forums: "Process utilization is calculated only for a single running process in GPU. Currently it is not supported for concurrent running processes."

Since the whole point of vramtop is shared GPU servers, showing this would be actively misleading. Instead we show:

- **Device-wide GPU util %** — accurate, honestly labeled
- **Per-process VRAM** — accurate, the real differentiator
- **Per-process allocation rate** — phase-aware EMA, unique to vramtop
- **Memory-bound vs compute-bound** — device-wide ratio, labeled as such

The README says this explicitly. Honesty builds trust.

---

## 3. Phase-Aware OOM Prediction (Complete Algorithm)

### 3.1 GPU Memory Allocation Patterns by Framework

| Framework | Pattern | Challenge |
|-----------|---------|-----------|
| **PyTorch** | Step-function: model load → flat → optimizer step-up → flat → per-batch sawtooth (caching allocator) | Caching allocator creates sawtooth noise on top of step functions |
| **JAX** | Near-instant pre-allocation to ~90% on first computation | Goes 0→90% in one sample. OOM prediction irrelevant. |
| **vLLM** | Model load → KV cache → flat with prefix cache eviction/reload cycles | Eviction dips → naive regression misinterprets as "shrinking" |
| **Ollama/llama.cpp** | Load model → flat. New model loads cause step increase. | Staircase pattern |
| **Training runs** | Model load → optimizer init (2x-3x jump) → per-epoch gradient accumulation | The 2-3x optimizer jump is the critical event |

### 3.2 Two-Stage Detection Algorithm

**Stage 1: Variance-threshold phase detector (MVP — zero dependencies)**

```python
from collections import deque
from dataclasses import dataclass
from enum import Enum

class Phase(Enum):
    STABLE = "stable"       # variance < threshold, |mean| < noise floor
    GROWING = "growing"     # sustained positive trend
    SHRINKING = "shrinking" # sustained negative trend
    VOLATILE = "volatile"   # high variance, no clear trend (e.g., sawtooth)

@dataclass
class PhaseState:
    phase: Phase
    duration_samples: int
    rate_mb_per_sec: float  # EMA-smoothed within this phase
    confidence: float       # 0.0 - 1.0

class PhaseDetector:
    """
    Sliding-window phase detection for GPU memory timeseries.
    
    Algorithm:
    1. Maintain rolling window of memory deltas (size=W, default 10)
    2. Compute rolling variance and mean of deltas
    3. Classify:
       - variance < threshold AND |mean| < noise_floor → STABLE
       - mean > noise_floor AND >60% deltas positive → GROWING
       - mean < -noise_floor AND >60% deltas negative → SHRINKING
       - else → VOLATILE
    4. Hysteresis: require M consecutive samples (default 3)
       before transitioning, preventing flicker.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        hysteresis_samples: int = 3,
        noise_floor_mb: float = 1.0,
        variance_threshold: float = 4.0,
        ema_alpha: float = 0.3,
    ):
        self.window = deque(maxlen=window_size)
        self.hysteresis = hysteresis_samples
        self.noise_floor = noise_floor_mb
        self.var_threshold = variance_threshold
        self.ema_alpha = ema_alpha
        
        self._current_phase = Phase.STABLE
        self._candidate_phase = Phase.STABLE
        self._candidate_count = 0
        self._phase_duration = 0
        self._ema_rate = 0.0
    
    def update(self, delta_mb: float, dt_seconds: float) -> PhaseState:
        rate = delta_mb / max(dt_seconds, 0.001)
        self.window.append(rate)
        
        if len(self.window) < 3:
            return PhaseState(Phase.STABLE, 0, 0.0, 0.0)
        
        rates = list(self.window)
        mean_rate = sum(rates) / len(rates)
        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        positive_frac = sum(1 for r in rates if r > self.noise_floor) / len(rates)
        negative_frac = sum(1 for r in rates if r < -self.noise_floor) / len(rates)
        
        if variance < self.var_threshold and abs(mean_rate) < self.noise_floor:
            detected = Phase.STABLE
        elif mean_rate > self.noise_floor and positive_frac > 0.6:
            detected = Phase.GROWING
        elif mean_rate < -self.noise_floor and negative_frac > 0.6:
            detected = Phase.SHRINKING
        else:
            detected = Phase.VOLATILE
        
        if detected == self._candidate_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = detected
            self._candidate_count = 1
        
        if (self._candidate_count >= self.hysteresis 
                and self._current_phase != self._candidate_phase):
            self._current_phase = self._candidate_phase
            self._phase_duration = 0
            self._ema_rate = mean_rate
        
        self._phase_duration += 1
        self._ema_rate = self.ema_alpha * rate + (1 - self.ema_alpha) * self._ema_rate
        
        confidence = min(1.0, self._phase_duration / 30.0) * (
            1.0 / (1.0 + variance / 100.0)
        )
        
        return PhaseState(
            phase=self._current_phase,
            duration_samples=self._phase_duration,
            rate_mb_per_sec=self._ema_rate,
            confidence=confidence,
        )
```

**Stage 2: PELT refinement (Phase 5 — optional `ruptures` dependency) -- IMPLEMENTED**

For post-mortem analysis and historical playback. Uses `ruptures` with `l2` cost model (piecewise constant, O(n) average). Framework-specific penalty presets. Gracefully degrades when `ruptures` is not installed (returns empty changepoint list). Implemented in `analysis/pelt_detector.py` with `detect_changepoints()` and `classify_segments()` functions.

```python
PENALTY_PRESETS = {
    "pytorch_training": 10,  # Many small steps, need sensitivity
    "inference_server": 50,  # Few big steps, need stability
    "unknown": 20,           # Balanced default
}
```

### 3.3 Per-Process Survival Predictor -- IMPLEMENTED

Complements the GPU-level OOM predictor with per-process "will this OOM?" verdicts. Implemented in `analysis/survival.py`.

**Verdicts:** `OK` (green), `TIGHT` (yellow), `OOM` (red) — shown as badges in the process table.

**Algorithm (stateless):**

1. **Scrape data first** — most accurate when available:
   - vLLM: uses KV cache utilization (`gpu_cache_usage_perc` v0 / `kv_cache_usage_perc` v1)
   - SGLang: uses `max_total_num_tokens` pool info
   - Ollama: excluded (loads fail before model is in VRAM — use NVML-only approach)

2. **Phase shortcut** — if `STABLE` or `SHRINKING`, return `OK`

3. **Multiplier heuristic** (for `GROWING`/`VOLATILE` phases):

```
Framework/mode       | Multiplier (peak / weights)
-------------------- | --------------------------
Inference only       | 1.2x
Inference long-ctx   | 1.5x
LoRA / PEFT          | 1.5x
Full training (SGD)  | 3.0x
Full training (Adam) | 4.0x
vLLM/SGLang server   | 1.8x
Unknown pytorch      | 2.5x (conservative)
Unknown              | 1.5x (generic)
```

4. **Pre-allocation awareness** — vLLM, SGLang, TGI, and JAX pre-allocate memory pools at startup. When `model_size_bytes` is unknown and we fall back to `process_used_bytes`, these frameworks use 1.05x multiplier (not the full multiplier) to avoid double-counting the already-allocated pool.

5. **Peak tracking** — historical peak memory per process (`max(process_used)` over time) overrides the multiplier estimate when observed usage exceeds the estimate.

6. **Collective pressure** — if the sum of all processes' estimated peak memory exceeds GPU total, individual OK verdicts are upgraded to TIGHT (and TIGHT to OOM).

### 3.4 OOM Prediction

```python
@dataclass
class OOMPrediction:
    seconds_low: float | None   # Optimistic (fastest growth in phase)
    seconds_high: float | None  # Pessimistic (slowest growth in phase)
    confidence: float           # 0.0 - 1.0
    display: str                # "OOM in ~60-120s" or "Stable"

class OOMPredictor:
    """
    Rules:
    1. Only predict during GROWING phase
    2. Only predict if growth sustained > 10 samples
    3. Use rate range (min/max EMA over phase) for confidence interval
    4. Never show point estimate — always a range
    5. Cap at "OOM in >1h"
    6. If rate < 1 MB/s sustained, suppress (noise)
    """
    
    def update(self, phase: PhaseState, free_mb: float) -> OOMPrediction:
        if phase.phase != Phase.GROWING:
            return OOMPrediction(None, None, 0.0, "Stable")
        
        # ... (rate tracking, range computation, display formatting)
        # Full implementation in v2 plan section 3.3
```

---

## 4. Unix Socket IPC Protocol (Deep Mode) -- IMPLEMENTED

Deep mode IPC is implemented across three modules: `reporter/protocol.py` (message types: HandshakeMsg, MemoryMsg), `reporter/pytorch.py` (PyTorch reporter daemon thread with socket server), and `enrichment/deep_mode.py` (socket discovery, stale cleanup, enrichment integration). See also the survival predictor (`analysis/survival.py`) which provides KV-cache-aware and scrape-data-aware OOM survival predictions.

### 4.1 Wire Protocol

Line-delimited JSON over Unix domain socket at `$XDG_RUNTIME_DIR/vramtop/<pid>.sock`.

```json
{"v":1,"pid":12345,"framework":"pytorch","cuda_device":0}
{"ts":1708000000.123,"allocated_mb":4096,"reserved_mb":6144,"active_mb":3800,"num_allocs":12345,"segments":42}
```

### 4.2 Reporter Module

```python
# User-facing: import vramtop; vramtop.report()
# OR: vramtop wrap -- python train.py
```

Key safety guarantees:
- **Daemon thread** — never blocks `sys.exit()` or training shutdown
- **Top-level try/except** on entire run loop — reporter crash ≠ training crash
- **atexit + SIGTERM handler** for socket cleanup
- **XDG_RUNTIME_DIR** paths (tmpfs, auto-cleaned on logout)
- Socket file permissions: `0o700` directory, only accessible by owner

### 4.3 Discovery

vramtop scans `$XDG_RUNTIME_DIR/vramtop/*.sock` for active reporters, same-UID only.

---

## 5. Progressive Enrichment Layers

### Layer 0: Universal (Any NVIDIA GPU Process) — No Special Permissions

- PID, process name (sanitized), uptime
- Process identity: `(PID, starttime)` validated every poll
- VRAM used (% of GPU total)
- Device-wide GPU utilization (labeled "device-wide")
- Allocation rate (EMA-smoothed via PhaseDetector)
- OOM risk indicator (phase-aware, range-based, confidence-gated)

### Layer 1: Smart Detection (Same-User Only)

**Security:** Only reads `/proc/<pid>/` for same-UID processes.

| Method | What | Security | Fallback |
|--------|------|----------|----------|
| `/proc/<pid>/cmdline` | Process command | Same-UID | Raw name from NVML |
| `/proc/<pid>/maps` | Loaded libraries | Same-UID | Skip framework detection |
| `/proc/<pid>/fd` → readlink | Open files | Same-UID | Skip model detection |
| `/proc/<pid>/stat` field 22 | Process starttime | Same-UID | PID-only identity (degraded) |
| Listening ports | Inference servers | psutil | Skip server detection |

**Container awareness:**
- Detect Docker/Podman via `/proc/1/cgroup` or `/.dockerenv`
- Parse `NVIDIA_VISIBLE_DEVICES` env (own process only)
- CDI-based GPU assignment via `/var/run/nvidia-container-devices/`
- `--pid=host` detection for full process discovery

**MPS awareness:**
- Detect MPS daemon via process scan for `nvidia-cuda-mps-control`
- Badge MPS clients with `[MPS]` tag

### Layer 2: Deep Mode (Opt-In)

**No gdb. No ptrace. No injection.**

| Framework | Method |
|-----------|--------|
| PyTorch | `vramtop wrap -- python train.py` (Unix socket IPC) |
| PyTorch | `import vramtop; vramtop.report()` (background daemon thread) |
| vLLM/SGLang/TGI | HTTP scrape (same-UID, rate-limited, schema-validated) |
| Ollama | HTTP scrape `/api/ps` |
| llama.cpp | HTTP scrape `/metrics` |

Scraping rules: localhost only, verify port owner PID, 500ms timeout, 5s rate limit, no redirects.

---

## 6. TUI Design

### 6.1 Framework: Textual

| Metric | Value |
|--------|-------|
| Render rate | ~10-25 redraws/s (terminal-limited). Poll GPU at 1-2 Hz. |
| CPU overhead | 1-2% one core |
| RAM usage | 40-60 MB |

Config hot-reload via SIGHUP handler + optional `watchdog` file watcher.

### 6.2 Visual Indicators

- OOM risk: **Bold red `!!! OOM ~60-120s`** (no blinking — seizure risk)
- Phase: `▬` STABLE, `▲` GROWING, `▼` SHRINKING, `〜` VOLATILE
- Confidence: `[■■■□□]` for 60%
- `NO_COLOR` env var respected
- All color-coded info also has symbol/text fallback

### 6.3 Process Kill Safety

1. `[k]` → confirmation dialog with process age
2. Same-UID only
3. SIGTERM first → 5s wait → offer SIGKILL
4. Audit log at `~/.local/share/vramtop/audit.log`
5. `--no-kill` flag for shared servers

### 6.4 Input Sanitization

Strip ANSI escape sequences + control chars from all `/proc/` and external API strings. Truncate to 256 chars. Idempotent. Tested with Hypothesis.

---

## 7. Secrets Management

**No `keyring`.** Fails on headless GPU servers.

Resolution order:
1. Environment variable: `VRAMTOP_WEBHOOK_URL`, etc.
2. `~/.config/vramtop/secrets.toml` with `0600` permissions (refused if group/other have access)
3. Interactive prompt for initial setup

---

## 8. Config

```toml
# ~/.config/vramtop/config.toml

[general]
refresh_rate = 1.0           # seconds, min 0.5, max 10
history_length = 300

[alerts]
oom_warning_seconds = 300
temp_warning_celsius = 85
oom_min_confidence = 0.3

[display]
theme = "dark"               # dark|light|nord|catppuccin|dracula|solarized
layout = "auto"              # auto|full|compact|mini
show_other_users = false
enable_kill = true
phase_indicator = true

[scraping]
enable = true
rate_limit_seconds = 5
timeout_ms = 500
allowed_ports = [8000, 8080, 11434, 3000]

[oom_prediction]
algorithm = "variance"       # "variance" (default) | "pelt" (requires ruptures)
min_sustained_samples = 10
min_rate_mb_per_sec = 1.0

[export]
prometheus_bind = "127.0.0.1"
prometheus_port = 0          # 0 = disabled
```

Schema-validated with Pydantic. Unknown keys rejected. Reloadable via SIGHUP.

---

## 9. Error Handling

| Failure Mode | Detection | Response |
|-------------|-----------|----------|
| NVML init fails | `nvmlInit()` error | "No NVIDIA GPU detected. vramtop requires an NVIDIA GPU with drivers installed." Exit cleanly. |
| GPU falls off bus | `NVML_ERROR_GPU_IS_LOST` | Mark "LOST — recovery needed", keep other GPUs live |
| Driver crash | `NVML_ERROR_UNKNOWN` | Retry 3x with exponential backoff, then degrade |
| Process vanishes | PID gone or starttime changed | Silently remove, log at debug |
| PID recycled | starttime mismatch | Discard old data, treat as new process |
| HTTP scrape fails | Timeout/refused/bad JSON | "metrics unavailable" badge, continue |
| Permission denied /proc | EACCES | Show VRAM only (from NVML), skip enrichment |
| Config corrupt | TOML parse error | Use defaults, show notification |
| Terminal too small | < 40 cols or < 10 rows | Auto-switch to minimal mode |
| Socket stale | File exists, no listener | Unlink and skip |

**Principle: Never crash. Degrade gracefully. Always show what we can.**

---

## 10. Testing Strategy

### 10.1 Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st

@given(deltas=st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=0, max_size=1000
))
def test_phase_detector_never_crashes(deltas):
    detector = PhaseDetector()
    for delta in deltas:
        result = detector.update(delta, 1.0)
        assert isinstance(result.phase, Phase)
        assert 0.0 <= result.confidence <= 1.0

@given(text=st.text(min_size=0, max_size=500))
def test_sanitize_idempotent(text):
    assert sanitize_process_name(text) == sanitize_process_name(sanitize_process_name(text))
```

### 10.2 Test Matrix

| Test Type | CI Without GPU? |
|-----------|-----------------|
| Unit: phase detection, OOM prediction | ✅ |
| Property: Hypothesis (phase, sanitizer, OOM) | ✅ |
| Unit: process identity, config validation | ✅ |
| Integration: NVML with recorded fixtures | ✅ |
| Integration: TUI with Textual `pilot` | ✅ |
| Integration: Unix socket IPC | ✅ |
| System: real GPU vs nvidia-smi | ❌ (GPU CI, weekly) |
| Security: UID boundary | ✅ (two test users) |

### 10.3 CI

```yaml
jobs:
  lint: ruff check --strict && mypy --strict src/
  test-unit: pytest tests/ -m "not gpu"
  test-property: pytest tests/test_property_*.py --hypothesis-show-statistics
  test-tui: pytest tests/test_tui.py
  security: bandit -r src/ && pip-audit
  test-gpu: pytest tests/ -m "gpu"  # self-hosted, weekly
```

---

## 11. File Structure

```
vramtop/
├── pyproject.toml
├── README.md
├── LICENSE (MIT)
├── src/
│   └── vramtop/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py                # Pydantic-validated, SIGHUP-reloadable
│       ├── secrets.py               # Env var + 0600 file
│       ├── sanitize.py              # ANSI strip, control char removal
│       ├── permissions.py           # UID checks
│       ├── process_identity.py      # (PID, starttime) management
│       │
│       ├── backends/
│       │   ├── __init__.py          # Auto-detection & factory
│       │   ├── base.py              # Abstract GPUBackend (~30 lines)
│       │   └── nvidia.py            # nvidia-ml-py wrapper (NVMLClient)
│       │
│       ├── enrichment/
│       │   ├── __init__.py
│       │   ├── detector.py          # Framework detection (same-UID)
│       │   ├── model_files.py       # Model file size estimation
│       │   ├── scraper.py           # HTTP scraper + security rules
│       │   ├── scrapers/
│       │   │   ├── vllm.py
│       │   │   ├── ollama.py
│       │   │   ├── sglang.py
│       │   │   └── llamacpp.py
│       │   ├── container.py         # Docker/Podman/CDI detection
│       │   ├── mps.py               # NVIDIA MPS awareness
│       │   └── deep_mode.py         # Unix socket IPC
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── phase_detector.py    # Variance-threshold (Stage 1)
│       │   ├── pelt_detector.py     # Optional PELT (Stage 2, ruptures)
│       │   ├── oom_predictor.py     # Range-based, confidence-gated
│       │   ├── survival.py         # Per-process survival predictor
│       │   ├── trends.py            # EMA, allocation rate
│       │   └── breakdown.py         # Weight vs dynamic (labeled estimate)
│       │
│       ├── reporter/
│       │   ├── __init__.py
│       │   ├── pytorch.py           # vramtop.report() for PyTorch
│       │   └── protocol.py          # Wire protocol definitions
│       │
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── app.py               # Textual app + SIGHUP handler
│       │   ├── widgets/
│       │   │   ├── gpu_card.py
│       │   │   ├── memory_bar.py
│       │   │   ├── timeline.py
│       │   │   ├── process_table.py
│       │   │   ├── detail_panel.py
│       │   │   ├── alerts.py
│       │   │   ├── kill_dialog.py
│       │   │   └── phase_badge.py   # ▬▲▼〜
│       │   ├── themes/
│       │   │   └── *.tcss
│       │   └── styles.tcss
│       │
│       └── export/
│           ├── __init__.py
│           ├── prometheus.py        # Binds 127.0.0.1 by default
│           ├── json_stream.py
│           ├── csv_logger.py
│           └── screenshot.py
│
├── tests/
│   ├── fixtures/                    # Recorded NVML responses
│   ├── conftest.py
│   ├── test_nvidia_backend.py
│   ├── test_phase_detector.py
│   ├── test_oom_predictor.py
│   ├── test_process_identity.py
│   ├── test_sanitize.py
│   ├── test_permissions.py
│   ├── test_scraper_security.py
│   ├── test_ipc_protocol.py
│   ├── test_config.py
│   ├── test_tui.py
│   ├── test_property_phase.py       # Hypothesis
│   ├── test_property_oom.py         # Hypothesis
│   └── test_property_sanitize.py    # Hypothesis
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 12. Dependencies

**Required:**
- Python ≥ 3.10
- `nvidia-ml-py` ≥ 13.0
- `textual` ≥ 1.0
- `psutil` ≥ 5.9
- `tomli` (Python < 3.11) / `tomllib` (3.11+)
- `pydantic` ≥ 2.0

**Optional:**
- `ruptures` — PELT algorithm for OOM prediction Stage 2
- `watchdog` — config file hot-reload
- `httpx` — HTTP scraping (fallback to `urllib.request`)

**Dev/Test:**
- `pytest`, `hypothesis`, `ruff`, `mypy`, `bandit`, `pip-audit`

**Explicitly NOT dependencies:**
- ~~`keyring`~~ — fails headless
- ~~`pynvml`~~ — deprecated
- ~~`requests`~~ — too heavy

---

## 13. Build Sequence

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1: Core** | 3 weeks | NVIDIA backend (compute+graphics+MPS), process identity (PID+starttime), memory bar, sparkline, PhaseDetector, basic OOM, sanitization, permission boundaries, error handling, Hypothesis tests |
| **Phase 2: Intelligence** | 2 weeks | Layer 1 detection (same-UID), OOMPredictor with confidence ranges, weight estimation, container detection, process detail panel, kill dialog |
| **Phase 3: Beauty** | 1 week | 6 themes, compact/full/mini modes, accessible mode, `NO_COLOR`, polish |
| **Phase 4: Scraping** | 1 week | vLLM/SGLang/Ollama/llama.cpp metrics (rate-limited, schema-validated, same-UID) |
| **Phase 5: Depth** | Complete (core) | Per-process survival predictor, deep mode (Unix socket IPC), PELT refinement, CSV export, SVG screenshot. Deferred: Prometheus, JSON stream, webhooks, SSH remote, historical playback |

**MVP = Phase 1 + 2 + 3 = ~6 weeks.**

---

## 14. Day 1 Priorities

Build these five files first, in this order:

1. **`nvidia.py`** — NVMLClient with both process calls + dedup. The foundation.
2. **`process_identity.py`** — `(PID, starttime)` from day 1. Retrofitting is painful.
3. **`phase_detector.py`** — Variance-threshold detector + Hypothesis tests immediately.
4. **`sanitize.py`** — Tiny, write early, use everywhere, test with Hypothesis.
5. **`app.py`** — Minimal Textual app: one GPU card, one process table. Something on screen.

---

## 15. What We Tell Users

### README positioning

> **nvidia-smi tells you how much. vramtop tells you why.**
>
> vramtop is a GPU memory monitoring tool for NVIDIA GPUs. It shows per-process VRAM usage, memory allocation trends, and predicts OOM events before they happen.

### Honest limitations section

> **What vramtop does NOT show:**
> - Per-process GPU compute utilization (NVML's API for this is unreliable with multiple processes — we won't show you wrong numbers)
> - Exact memory breakdowns (weight vs activations estimates are labeled as estimates)
> - Non-NVIDIA GPUs (v1.0 is NVIDIA-only; multi-vendor support is planned)

### Hook for future vendors

> **Want AMD/Apple/Intel support?** Star the repo and open an issue. The backend abstraction is already in place — contributions welcome.