"""Per-process survival predictor: will this process OOM?

Stateless heuristic using framework-aware memory multipliers and
scrape-data-aware overrides for inference servers.

Produces a verdict per process: OK (green), TIGHT (yellow), OOM (red).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Verdict(Enum):
    """Survival verdict for a single process."""

    OK = "ok"
    TIGHT = "tight"
    OOM = "oom"


@dataclass(frozen=True, slots=True)
class SurvivalPrediction:
    """Result of survival prediction for a process."""

    verdict: Verdict
    reason: str


# --- Multiplier heuristic ---
# Peak memory / model weights ratio by framework and mode.
# Components:
#   Inference only:       weights + KV cache overhead
#   Inference long-ctx:   weights + large KV cache (long sequences)
#   LoRA/PEFT:            weights + adapter + small optimizer
#   Full training (SGD):  weights + gradients + activations
#   Full training (Adam): weights + gradients + optimizer(2x for m+v) + activations
#   vLLM/SGLang server:   weights + KV cache pool (pre-allocated)
#   Unknown pytorch:      conservative (could be training)
#   Unknown:              generic

_MULTIPLIERS: dict[str, float] = {
    "inference": 1.2,
    "inference_long_ctx": 1.5,
    "lora": 1.5,
    "peft": 1.5,
    "training_sgd": 3.0,
    "training_adam": 4.0,
    "vllm_server": 1.8,
    "sglang_server": 1.8,
    "pytorch": 2.5,
    "generic": 1.5,
}

# Frameworks that pre-allocate a large memory pool at startup (KV cache, JAX XLA).
# When model_size_bytes is unknown and we fall back to process_used_bytes as base,
# the multiplier must be ~1.0x because process_used already includes the pool.
# Applying 1.8x on top of an already-pre-allocated 12GB → false 21.6GB estimate.
_PRE_ALLOCATING_FRAMEWORKS: frozenset[str] = frozenset({
    "vllm", "sglang", "tgi", "jax",
})


def get_multiplier(
    framework: str | None, cmdline: str | None
) -> float:
    """Determine peak-memory multiplier from framework and command line.

    Args:
        framework: Detected framework name (e.g. "pytorch", "vllm").
        cmdline: Process command line string.

    Returns:
        Multiplier (peak / model weights).
    """
    cmd = (cmdline or "").lower()

    # Check for LoRA/PEFT indicators
    if "lora" in cmd or "peft" in cmd:
        return _MULTIPLIERS["lora"]

    # Check for optimizer hints (Adam variants)
    if any(kw in cmd for kw in ("adam", "adamw", "adafactor")):
        return _MULTIPLIERS["training_adam"]

    # Check for training indicators with SGD
    if "sgd" in cmd:
        return _MULTIPLIERS["training_sgd"]

    # Check for generic training indicators — assume Adam (conservative)
    if any(kw in cmd for kw in ("train", "finetune", "fine-tune", "fine_tune")):
        return _MULTIPLIERS["training_adam"]

    # vLLM / SGLang inference servers have pre-allocated KV cache pools
    if framework == "vllm":
        return _MULTIPLIERS["vllm_server"]
    if framework == "sglang":
        return _MULTIPLIERS["sglang_server"]

    # TGI also pre-allocates KV cache pool
    if framework == "tgi":
        return _MULTIPLIERS["vllm_server"]

    # Inference servers with lower overhead
    if framework in ("ollama", "llamacpp"):
        return _MULTIPLIERS["inference"]

    # JAX pre-allocates ~90% on first computation — already at peak
    if framework == "jax":
        return _MULTIPLIERS["inference"]

    # Long-context inference detection from cmdline
    if any(kw in cmd for kw in ("--max-model-len", "--context-length", "--ctx-size")):
        return _MULTIPLIERS["inference_long_ctx"]

    # PyTorch without training indicators — unknown usage (conservative)
    if framework == "pytorch":
        return _MULTIPLIERS["pytorch"]

    # Generic unknown
    return _MULTIPLIERS["generic"]


def _check_scrape_data(
    scrape_data: dict[str, Any],
    framework: str | None,
    process_used_bytes: int,
    gpu_free_bytes: int,
) -> SurvivalPrediction | None:
    """Check scrape data for framework-specific survival signals.

    Returns a prediction if scrape data gives a definitive answer, else None
    to fall through to the multiplier heuristic.
    """
    if framework == "vllm":
        # Support both v0 (gpu_cache_usage_perc) and v1 (kv_cache_usage_perc) metric names
        kv_cache_usage = scrape_data.get("gpu_cache_usage_perc")
        if kv_cache_usage is None:
            kv_cache_usage = scrape_data.get("kv_cache_usage_perc")
        if kv_cache_usage is not None:
            try:
                usage = float(kv_cache_usage)
            except (TypeError, ValueError):
                return None

            # At KV cache limit with very little free memory -> OOM risk
            if usage >= 0.99 and gpu_free_bytes < 100 * 1024 * 1024:
                return SurvivalPrediction(
                    Verdict.OOM,
                    f"KV cache {usage:.0%}, {gpu_free_bytes // (1024 * 1024)} MB free",
                )

            if usage > 0.9:
                return SurvivalPrediction(
                    Verdict.TIGHT,
                    f"KV cache {usage:.0%} utilized",
                )

            return SurvivalPrediction(
                Verdict.OK,
                f"KV cache {usage:.0%}",
            )

    # Ollama: loads fail fast (before model is fully in VRAM), so scrape data
    # only tells us what's already loaded — use NVML multiplier approach instead.

    if framework == "sglang":
        max_tokens = scrape_data.get("max_total_num_tokens")
        if max_tokens is not None and isinstance(max_tokens, int) and max_tokens > 0:
            # SGLang pre-allocates memory for max_total_num_tokens
            if gpu_free_bytes < 200 * 1024 * 1024:
                return SurvivalPrediction(
                    Verdict.TIGHT,
                    f"SGLang pool active, {gpu_free_bytes // (1024 * 1024)} MB free",
                )
            return SurvivalPrediction(
                Verdict.OK,
                f"SGLang pool: {max_tokens} tokens capacity",
            )

    return None


def estimate_peak(
    *,
    framework: str | None,
    cmdline: str | None,
    model_size_bytes: int | None,
    process_used_bytes: int,
    peak_used_bytes: int | None = None,
) -> int:
    """Estimate peak memory for a process.

    Uses multiplier heuristic, but prefers historical peak when it exceeds
    the multiplier estimate (the process has already proven it needs more).

    Important: for pre-allocating frameworks (vLLM, SGLang, TGI, JAX),
    when model_size_bytes is unknown and we fall back to process_used_bytes,
    we use a ~1.05x multiplier. These frameworks pre-allocate their memory
    pool at startup, so process_used already includes the full allocation.
    Applying the normal multiplier would double-count.

    Returns estimated peak in bytes.
    """
    multiplier = get_multiplier(framework, cmdline)

    if model_size_bytes is not None and framework not in _PRE_ALLOCATING_FRAMEWORKS:
        # Non-pre-allocating: model_size * multiplier is a good peak estimate
        base = model_size_bytes
    elif framework in _PRE_ALLOCATING_FRAMEWORKS:
        # Pre-allocating framework without model size info: process_used
        # already includes the pre-allocated pool. Use 1.05x (small buffer
        # for runtime overhead) instead of the full multiplier.
        base = process_used_bytes
        multiplier = 1.05
    else:
        base = process_used_bytes

    estimated = int(base * multiplier)

    # If we've observed a higher peak, trust the observation
    if peak_used_bytes is not None and peak_used_bytes > estimated:
        estimated = peak_used_bytes

    return estimated


# Absolute headroom thresholds (fraction of total GPU memory).
# Training workloads burst-allocate during forward/backward passes.  Even a
# "stable" process can OOM on the next spike if free memory is too low.
_TIGHT_FREE_PCT = 5.0   # <5% free → TIGHT regardless of phase
_CRITICAL_FREE_PCT = 2.0  # <2% free → on the edge


def predict_survival(
    *,
    phase: str,
    framework: str | None,
    cmdline: str | None,
    model_size_bytes: int | None,
    process_used_bytes: int,
    gpu_free_bytes: int,
    gpu_total_bytes: int = 0,
    scrape_data: dict[str, Any] | None = None,
    peak_used_bytes: int | None = None,
) -> SurvivalPrediction:
    """Predict whether a process will survive without OOM.

    Stateless function — takes current state, returns verdict.

    Args:
        phase: Current phase string ("stable", "growing", etc.).
        framework: Detected framework or None.
        cmdline: Process command line or None.
        model_size_bytes: Estimated model file size or None.
        process_used_bytes: Current process VRAM usage in bytes.
        gpu_free_bytes: Currently free GPU memory in bytes.
        gpu_total_bytes: Total GPU memory in bytes (for % calculations).
        scrape_data: Optional dict from HTTP scraper (vLLM metrics, etc.).
        peak_used_bytes: Historical peak memory for this process, or None.

    Returns:
        SurvivalPrediction with verdict and reason.
    """
    # Check scrape data first — most accurate signal when available
    if scrape_data is not None and framework is not None:
        scrape_result = _check_scrape_data(
            scrape_data, framework, process_used_bytes, gpu_free_bytes
        )
        if scrape_result is not None:
            return scrape_result

    is_pre_alloc = framework in _PRE_ALLOCATING_FRAMEWORKS
    free_mb = gpu_free_bytes / (1024 * 1024)

    # --- Spike detection ---
    # Training loops have sawtooth memory patterns: forward pass allocates
    # activations (spike), backward pass peaks, then memory drops between
    # steps.  If we've observed a spike larger than available free memory,
    # the next forward pass will OOM.
    spike_bytes = 0
    if peak_used_bytes is not None and peak_used_bytes > process_used_bytes:
        spike_bytes = peak_used_bytes - process_used_bytes

    if not is_pre_alloc and spike_bytes > 0 and spike_bytes > gpu_free_bytes:
        spike_mb = spike_bytes // (1024 * 1024)
        return SurvivalPrediction(
            Verdict.OOM,
            f"spike ~{spike_mb} MB > {int(free_mb)} MB free",
        )

    # --- Absolute headroom checks (any phase) ---
    # Pre-allocating frameworks (vLLM, SGLang, JAX) manage their own memory
    # pools and deliberately run at high utilization — skip this for them.
    total = max(gpu_total_bytes, 1)
    free_pct = gpu_free_bytes / total * 100 if gpu_total_bytes > 0 else 100.0

    if not is_pre_alloc and gpu_total_bytes > 0 and free_pct < _CRITICAL_FREE_PCT:
            return SurvivalPrediction(
                Verdict.TIGHT,
                f"{int(free_mb)} MB free ({free_pct:.0f}%)",
            )

    # --- Phase-based logic ---

    if phase == "stable":
        # Stable but low headroom → warn (training can spike any time)
        if not is_pre_alloc and gpu_total_bytes > 0 and free_pct < _TIGHT_FREE_PCT:
            return SurvivalPrediction(
                Verdict.TIGHT,
                f"stable but {int(free_mb)} MB free ({free_pct:.0f}%)",
            )
        return SurvivalPrediction(Verdict.OK, "stable")

    if phase == "shrinking":
        return SurvivalPrediction(Verdict.OK, "past peak")

    # Growing or volatile: use multiplier + historical peak heuristic
    estimated = estimate_peak(
        framework=framework,
        cmdline=cmdline,
        model_size_bytes=model_size_bytes,
        process_used_bytes=process_used_bytes,
        peak_used_bytes=peak_used_bytes,
    )
    headroom_needed = estimated - process_used_bytes

    if headroom_needed <= 0:
        return SurvivalPrediction(Verdict.OK, "past peak")

    if gpu_free_bytes >= int(headroom_needed * 1.5):
        return SurvivalPrediction(Verdict.OK, "sufficient headroom")

    if gpu_free_bytes >= headroom_needed:
        return SurvivalPrediction(
            Verdict.TIGHT,
            f"~{headroom_needed // (1024 * 1024)} MB needed, "
            f"{gpu_free_bytes // (1024 * 1024)} MB free",
        )

    return SurvivalPrediction(
        Verdict.OOM,
        f"~{headroom_needed // (1024 * 1024)} MB needed, "
        f"only {gpu_free_bytes // (1024 * 1024)} MB free",
    )


def check_collective_pressure(
    predictions: dict[int, SurvivalPrediction],
    estimated_peaks: dict[int, int],
    gpu_total_bytes: int,
) -> dict[int, SurvivalPrediction]:
    """Upgrade verdicts based on collective pressure across all processes.

    If the sum of all processes' estimated peak memory exceeds GPU total,
    individual OK verdicts are upgraded to TIGHT (or TIGHT to OOM).

    Args:
        predictions: Per-PID survival predictions.
        estimated_peaks: Per-PID estimated peak memory in bytes.
        gpu_total_bytes: Total GPU memory in bytes.

    Returns:
        New predictions dict with upgraded verdicts where needed.
    """
    if not estimated_peaks:
        return predictions

    total_peak = sum(estimated_peaks.values())
    if total_peak <= gpu_total_bytes:
        return predictions

    overcommit_ratio = total_peak / max(gpu_total_bytes, 1)
    reason_suffix = f" ({overcommit_ratio:.1f}x overcommit)"

    result: dict[int, SurvivalPrediction] = {}
    for pid, pred in predictions.items():
        if pred.verdict == Verdict.OK and pid in estimated_peaks:
            result[pid] = SurvivalPrediction(
                Verdict.TIGHT,
                pred.reason + reason_suffix,
            )
        elif pred.verdict == Verdict.TIGHT and pid in estimated_peaks:
            result[pid] = SurvivalPrediction(
                Verdict.OOM,
                pred.reason + reason_suffix,
            )
        else:
            result[pid] = pred

    return result
