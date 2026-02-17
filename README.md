# vramtop

**The htop for GPU memory.** Beautiful. Zero-friction. NVIDIA-first.

A terminal UI for monitoring GPU memory usage across all your NVIDIA GPUs. See per-process VRAM consumption, memory phase detection, OOM prediction, and framework-aware enrichment -- all without instrumenting your code.

## Install

```bash
pip install vramtop
```

Requires Python 3.10+ and NVIDIA drivers (make sure `nvidia-smi` works).

### Optional extras

```bash
pip install vramtop[pelt]    # PELT changepoint detection (requires ruptures + numpy)
```

## Quick start

```bash
vramtop
```

That's it. vramtop discovers all NVIDIA GPUs and shows live memory usage.

## Features

- **Per-process VRAM tracking** -- see exactly which processes use how much GPU memory
- **Memory phase detection** -- classifies each process as stable, growing, shrinking, or volatile
- **OOM prediction** -- warns before your GPU runs out of memory, with time-to-OOM estimates
- **Survival verdicts** -- per-process OK / TIGHT / OOM risk assessment with spike detection
- **Framework detection** -- automatically identifies PyTorch, JAX, vLLM, Ollama, SGLang, llama.cpp, TGI
- **Model file scanning** -- detects which model files are loaded from /proc/fd
- **Container awareness** -- shows Docker/Podman container IDs per process
- **PELT changepoint analysis** -- optional segment-level memory phase analysis with labeled segments
- **Deep mode** -- get PyTorch-internal memory stats (allocated vs reserved vs active) via IPC
- **HTTP scraping** -- pulls KV cache usage from vLLM, Ollama, SGLang, llama.cpp metrics endpoints
- **6 themes** -- dark (GitHub-dark), light, nord, catppuccin, dracula, solarized
- **CSV export** -- log GPU and process data to CSV for offline analysis
- **SVG screenshots** -- save terminal screenshots as SVG
- **Accessibility** -- respects `NO_COLOR`, `--accessible` mode with text labels

## Usage

### Monitor (default)

```bash
vramtop                          # Launch the TUI
vramtop --refresh-rate 2         # Poll every 2 seconds (default: 1)
vramtop --no-kill                # Disable process kill functionality
vramtop --accessible             # Text labels instead of Unicode symbols
vramtop --export-csv gpu.csv     # Log data to CSV while monitoring
vramtop --config path/to/config.toml
```

### Wrap (deep mode)

Run any command with deep mode reporting enabled:

```bash
vramtop wrap -- python train.py
```

This sets `VRAMTOP_REPORT=1` in the subprocess environment. If your training script calls `vramtop.report()`, it will send PyTorch memory internals to vramtop over a Unix socket.

### Deep mode (in your code)

Add two lines to your PyTorch script to report allocator-level memory stats:

```python
import vramtop
vramtop.report()

# ... your training code ...
```

vramtop's TUI will automatically discover the reporter socket and show allocated, reserved, and active memory breakdowns in the detail panel.

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Open detail panel for selected process |
| `k` | Kill selected process (SIGTERM, then SIGKILL) |
| `t` | Cycle through themes |
| `s` | Save SVG screenshot |
| `?` | Show help |
| Up/Down | Navigate process list |

## Configuration

Config file location: `~/.config/vramtop/config.toml`

Reload without restarting: send `SIGHUP` (`kill -HUP $(pgrep vramtop)`).

```toml
[general]
refresh_rate = 1.0       # Poll interval in seconds (0.5-10)
history_length = 300     # Samples to keep for sparkline/analysis

[display]
theme = "dark"           # dark, light, nord, catppuccin, dracula, solarized
layout = "auto"          # auto, full, compact, mini
show_other_users = false
enable_kill = true
phase_indicator = true

[alerts]
oom_warning_seconds = 300
temp_warning_celsius = 85
oom_min_confidence = 0.3

[scraping]
enable = true
rate_limit_seconds = 5
timeout_ms = 500
allowed_ports = [8000, 8080, 11434, 3000]

[oom_prediction]
algorithm = "variance"   # variance or pelt
min_sustained_samples = 10
min_rate_mb_per_sec = 1.0
```

## Layout modes

vramtop auto-detects terminal size and picks the best layout:

| Mode | Min size | Shows |
|------|----------|-------|
| **Full** | 120x30 | GPU cards, memory bars, sparklines, process tables, phase badges |
| **Compact** | 80x20 | GPU cards, memory bars, process tables |
| **Mini** | 40x10 | Minimal GPU summary |

Override with `layout = "full"` in config or let `"auto"` handle it.

## How it works

vramtop reads GPU data through NVML (`nvidia-ml-py`) -- the same library that powers `nvidia-smi`. No root access needed, no kernel modules, no LD_PRELOAD hacks.

**Data sources by tier:**

1. **NVML** (always) -- total/used/free VRAM, per-process memory, temperature, power, clocks
2. **`/proc` enrichment** (always) -- framework detection from cmdline/maps, model files from fd, container from cgroup
3. **HTTP scraping** (opt-in) -- KV cache metrics from inference server endpoints (localhost only, port-owner verified)
4. **Deep mode IPC** (opt-in) -- PyTorch allocator stats via Unix domain socket

**Security model:**

- All `/proc` reads are same-UID only
- HTTP scraping enforces: localhost-only, port-owner verification, rate limiting, no redirects, 64KB size cap
- Kill requires confirmation (SIGTERM first, SIGKILL only after 5s timeout)
- All external strings sanitized (ANSI/control char stripping, 256-char truncation)

## Requirements

- Python 3.10+
- NVIDIA GPU with drivers installed
- Linux (uses `/proc` for enrichment)

## License

MIT
