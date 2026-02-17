"""Tests for the per-process survival predictor."""

from __future__ import annotations

from vramtop.analysis.survival import (
    SurvivalPrediction,
    Verdict,
    check_collective_pressure,
    estimate_peak,
    get_multiplier,
    predict_survival,
)


def _gb(n: float) -> int:
    """Convert GB to bytes."""
    return int(n * 1024 * 1024 * 1024)


def _mb(n: float) -> int:
    """Convert MB to bytes."""
    return int(n * 1024 * 1024)


class TestGetMultiplier:
    """Test multiplier selection for different framework/mode combos."""

    def test_lora_from_cmdline(self) -> None:
        """LoRA in cmdline should return 1.5x multiplier."""
        assert get_multiplier("pytorch", "python train_lora.py") == 1.5

    def test_peft_from_cmdline(self) -> None:
        """PEFT in cmdline should return 1.5x multiplier."""
        assert get_multiplier("pytorch", "python run_peft_model.py") == 1.5

    def test_adam_training(self) -> None:
        """Adam optimizer keyword should return 4.0x multiplier."""
        assert get_multiplier("pytorch", "python train.py --optimizer adam") == 4.0

    def test_adamw_training(self) -> None:
        """AdamW optimizer keyword should return 4.0x multiplier."""
        assert get_multiplier("pytorch", "python train.py --optimizer adamw") == 4.0

    def test_sgd_training(self) -> None:
        """SGD optimizer keyword should return 3.0x multiplier."""
        assert get_multiplier("pytorch", "python train.py --optimizer sgd") == 3.0

    def test_generic_training(self) -> None:
        """Generic 'train' keyword should assume Adam (4.0x)."""
        assert get_multiplier("pytorch", "python train.py") == 4.0

    def test_finetune(self) -> None:
        """Finetune keyword should assume Adam (4.0x)."""
        assert get_multiplier("pytorch", "python finetune_model.py") == 4.0

    def test_vllm_server(self) -> None:
        """vLLM framework should return 1.8x (pre-allocated KV cache pool)."""
        assert get_multiplier("vllm", "python -m vllm.entrypoints.openai.api_server") == 1.8

    def test_sglang_server(self) -> None:
        """SGLang framework should return 1.8x (pre-allocated KV cache pool)."""
        assert get_multiplier("sglang", "python -m sglang.launch_server") == 1.8

    def test_ollama_inference(self) -> None:
        """Ollama framework should return 1.2x (inference-only)."""
        assert get_multiplier("ollama", "ollama serve") == 1.2

    def test_llamacpp_inference(self) -> None:
        """llama.cpp framework should return 1.2x (inference-only)."""
        assert get_multiplier("llamacpp", "./server --model model.gguf") == 1.2

    def test_pytorch_unknown(self) -> None:
        """Unknown pytorch usage should return 2.5x (conservative)."""
        assert get_multiplier("pytorch", "python my_script.py") == 2.5

    def test_generic_unknown(self) -> None:
        """Completely unknown framework should return 1.5x (generic)."""
        assert get_multiplier(None, None) == 1.5

    def test_none_cmdline(self) -> None:
        """None cmdline should not crash and fall through to framework check."""
        assert get_multiplier("vllm", None) == 1.8

    def test_long_context_inference(self) -> None:
        """Long-context flags in cmdline should return 1.5x."""
        assert get_multiplier(None, "python serve.py --max-model-len 32768") == 1.5

    def test_lora_overrides_framework(self) -> None:
        """LoRA keyword in cmdline should override vllm framework multiplier."""
        assert get_multiplier("vllm", "python -m vllm lora_adapter") == 1.5

    def test_tgi_server(self) -> None:
        """TGI framework should return 1.8x (pre-allocated KV cache pool)."""
        assert get_multiplier("tgi", "text-generation-launcher") == 1.8

    def test_jax_inference(self) -> None:
        """JAX framework should return 1.2x (pre-allocates on first call)."""
        assert get_multiplier("jax", "python jax_serve.py") == 1.2


class TestPreAllocationFix:
    """Test that pre-allocating frameworks don't double-count when model_size unknown."""

    def test_vllm_no_model_size_uses_low_multiplier(self) -> None:
        """vLLM without model_size_bytes should not apply 1.8x to process_used."""
        # vLLM already pre-allocated 12GB. Without model_size, base = 12GB.
        # With old code: 12 * 1.8 = 21.6GB → false OOM
        # With fix: 12 * 1.05 = 12.6GB → headroom = 0.6GB, free = 2GB → OK
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=None,
            process_used_bytes=_gb(12),
            gpu_free_bytes=_gb(2),
        )
        assert result.verdict == Verdict.OK

    def test_vllm_with_model_size_still_uses_process_used(self) -> None:
        """vLLM WITH model_size_bytes still uses process_used*1.05.

        Pre-allocating frameworks (vLLM, SGLang, TGI, JAX) allocate a memory
        pool at startup proportional to remaining GPU memory, NOT to model size.
        So model_size * 1.8x is wrong — the KV cache pool can be much larger
        than model weights.  process_used already reflects the true allocation.
        """
        # process_used=4GB, multiplier=1.05 (pre-alloc), peak=4.2GB, needed=0.2GB
        # free=2GB → OK with sufficient headroom
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
        )
        assert result.verdict == Verdict.OK

    def test_sglang_no_model_size(self) -> None:
        """SGLang without model_size should use 1.05x on process_used."""
        result = predict_survival(
            phase="growing",
            framework="sglang",
            cmdline="python -m sglang.launch_server",
            model_size_bytes=None,
            process_used_bytes=_gb(10),
            gpu_free_bytes=_gb(2),
        )
        # 10 * 1.05 = 10.5GB, needed = 0.5GB, free = 2GB → OK
        assert result.verdict == Verdict.OK

    def test_jax_no_model_size(self) -> None:
        """JAX without model_size should use 1.05x (pre-allocates ~90%)."""
        result = predict_survival(
            phase="growing",
            framework="jax",
            cmdline="python jax_train.py",
            model_size_bytes=None,
            process_used_bytes=_gb(14),
            gpu_free_bytes=_gb(1),
        )
        # 14 * 1.05 = 14.7GB, needed = 0.7GB, free = 1GB → OK (1 >= 0.7*1.5=1.05? just barely)
        assert result.verdict in (Verdict.OK, Verdict.TIGHT)

    def test_pytorch_no_model_size_uses_full_multiplier(self) -> None:
        """PyTorch is NOT pre-allocating — should use full 2.5x multiplier."""
        result = predict_survival(
            phase="growing",
            framework="pytorch",
            cmdline="python my_script.py",
            model_size_bytes=None,
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
        )
        # 4 * 2.5 = 10GB, needed = 6GB, free = 2GB → OOM
        assert result.verdict == Verdict.OOM


class TestPredictSurvival:
    """Test survival prediction verdicts."""

    def test_stable_phase_is_ok(self) -> None:
        """Stable phase should always return OK."""
        result = predict_survival(
            phase="stable",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(2),
            process_used_bytes=_gb(8),
            gpu_free_bytes=_gb(2),
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "stable"

    def test_shrinking_phase_is_ok(self) -> None:
        """Shrinking phase should return OK (past peak)."""
        result = predict_survival(
            phase="shrinking",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(2),
            process_used_bytes=_gb(8),
            gpu_free_bytes=_gb(1),
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "past peak"

    def test_past_peak_when_used_exceeds_estimate(self) -> None:
        """If current usage exceeds estimated peak, verdict is OK (past peak)."""
        result = predict_survival(
            phase="growing",
            framework="ollama",
            cmdline="ollama serve",
            model_size_bytes=_gb(2),
            process_used_bytes=_gb(4),  # Already 4 GB, peak = 2*1.2 = 2.4 GB
            gpu_free_bytes=_gb(1),
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "past peak"

    def test_sufficient_headroom(self) -> None:
        """Should return OK when free memory is >= 1.5x headroom needed."""
        # model=2GB, multiplier=1.2 (ollama), peak=2.4GB, used=2GB, needed=0.4GB
        # Need free >= 0.4*1.5 = 0.6GB → 1GB free is enough
        result = predict_survival(
            phase="growing",
            framework="ollama",
            cmdline="ollama serve",
            model_size_bytes=_gb(2),
            process_used_bytes=_gb(2),
            gpu_free_bytes=_gb(1),
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "sufficient headroom"

    def test_tight_when_marginal_headroom(self) -> None:
        """Should return TIGHT when free >= needed but < 1.5x needed."""
        # model=4GB, multiplier=4.0 (adam training pytorch), peak=16GB, used=4GB, needed=12GB
        # Need free >= 12*1.5=18GB for OK.  free=14GB >= 12GB but < 18GB → TIGHT
        result = predict_survival(
            phase="growing",
            framework="pytorch",
            cmdline="python train.py --optimizer adam",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(14),
        )
        assert result.verdict == Verdict.TIGHT

    def test_oom_when_insufficient_free(self) -> None:
        """Should return OOM when free memory < headroom needed."""
        # model=4GB, multiplier=4.0 (adam training), peak=16GB, used=4GB, needed=12GB
        # free=2GB < 12GB → OOM
        result = predict_survival(
            phase="growing",
            framework="pytorch",
            cmdline="python train.py --optimizer adam",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
        )
        assert result.verdict == Verdict.OOM
        assert "needed" in result.reason
        assert "free" in result.reason

    def test_no_model_size_falls_back_to_process_used(self) -> None:
        """When model_size_bytes is None, should use process_used_bytes as base."""
        # process_used=2GB, multiplier=1.5 (generic), peak=3GB, needed=1GB
        # free=2GB >= 1.5*1GB → OK
        result = predict_survival(
            phase="growing",
            framework=None,
            cmdline=None,
            model_size_bytes=None,
            process_used_bytes=_gb(2),
            gpu_free_bytes=_gb(2),
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "sufficient headroom"

    def test_volatile_phase_uses_multiplier(self) -> None:
        """Volatile phase should use the multiplier heuristic (not skip)."""
        # model=4GB, multiplier=2.5 (pytorch unknown), peak=10GB, used=4GB, needed=6GB
        # free=1GB < 6GB → OOM
        result = predict_survival(
            phase="volatile",
            framework="pytorch",
            cmdline="python my_script.py",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(1),
        )
        assert result.verdict == Verdict.OOM


class TestScrapeDataAware:
    """Test scrape-data-aware prediction overrides."""

    def test_vllm_gpu_cache_usage_perc_low_is_ok(self) -> None:
        """vLLM v0 metric gpu_cache_usage_perc with low usage should return OK."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
            scrape_data={"gpu_cache_usage_perc": 0.5},
        )
        assert result.verdict == Verdict.OK
        assert "KV cache" in result.reason

    def test_vllm_kv_cache_usage_perc_low_is_ok(self) -> None:
        """vLLM v1 metric kv_cache_usage_perc with low usage should return OK."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
            scrape_data={"kv_cache_usage_perc": 0.5},
        )
        assert result.verdict == Verdict.OK
        assert "KV cache" in result.reason

    def test_vllm_gpu_cache_usage_perc_high_is_tight(self) -> None:
        """vLLM v0 metric gpu_cache_usage_perc >90% should return TIGHT."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
            scrape_data={"gpu_cache_usage_perc": 0.95},
        )
        assert result.verdict == Verdict.TIGHT
        assert "KV cache" in result.reason

    def test_vllm_kv_cache_usage_perc_high_is_tight(self) -> None:
        """vLLM v1 metric kv_cache_usage_perc >90% should return TIGHT."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
            scrape_data={"kv_cache_usage_perc": 0.95},
        )
        assert result.verdict == Verdict.TIGHT
        assert "KV cache" in result.reason

    def test_vllm_kv_cache_full_with_low_free_is_oom(self) -> None:
        """vLLM with KV cache >=99% and <100MB free should return OOM."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(14),
            gpu_free_bytes=_mb(50),  # Only 50 MB free
            scrape_data={"gpu_cache_usage_perc": 0.99},
        )
        assert result.verdict == Verdict.OOM
        assert "KV cache" in result.reason

    def test_vllm_v1_kv_cache_full_with_low_free_is_oom(self) -> None:
        """vLLM v1 with kv_cache_usage_perc >=99% and <100MB free should return OOM."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(14),
            gpu_free_bytes=_mb(50),
            scrape_data={"kv_cache_usage_perc": 0.995},
        )
        assert result.verdict == Verdict.OOM
        assert "KV cache" in result.reason

    def test_vllm_v0_preferred_over_v1(self) -> None:
        """When both v0 and v1 keys are present, v0 (gpu_cache_usage_perc) wins."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
            scrape_data={
                "gpu_cache_usage_perc": 0.5,  # v0 → OK
                "kv_cache_usage_perc": 0.95,  # v1 → TIGHT (ignored)
            },
        )
        # v0 key is checked first, so 0.5 → OK
        assert result.verdict == Verdict.OK
        assert "KV cache" in result.reason

    def test_vllm_invalid_scrape_falls_through(self) -> None:
        """Invalid vLLM scrape data should fall through to multiplier logic."""
        result = predict_survival(
            phase="stable",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
            scrape_data={"gpu_cache_usage_perc": "invalid"},
        )
        # Falls through to multiplier logic, phase=stable → OK
        assert result.verdict == Verdict.OK
        assert result.reason == "stable"

    def test_vllm_with_request_counts(self) -> None:
        """vLLM scrape data with request metrics alongside KV cache should work."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
            scrape_data={
                "gpu_cache_usage_perc": 0.3,
                "num_requests_running": 5,
                "num_requests_waiting": 2,
            },
        )
        assert result.verdict == Verdict.OK
        assert "KV cache" in result.reason

    def test_ollama_scrape_data_falls_through_to_multiplier(self) -> None:
        """Ollama scrape data is not used for survival — falls through to multiplier.

        Ollama loads fail fast (before model is fully in VRAM), so scrape_data
        only tells us what's already loaded, not what memory will be needed.
        """
        # model=4GB, multiplier=1.2 (ollama inference), peak=4.8GB, used=4GB, needed=0.8GB
        # free=4GB >= 0.8*1.5=1.2GB → OK with sufficient headroom
        result = predict_survival(
            phase="growing",
            framework="ollama",
            cmdline="ollama serve",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(4),
            scrape_data={
                "models": [
                    {
                        "name": "llama2:7b",
                        "size": _gb(4),
                        "size_vram": _gb(4),
                        "family": "llama",
                        "parameter_size": "7B",
                        "quantization_level": "Q4_0",
                    },
                ]
            },
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "sufficient headroom"

    def test_ollama_multiple_models_falls_through(self) -> None:
        """Ollama with multiple models also falls through to multiplier heuristic."""
        # model=8GB, multiplier=1.2 (ollama), peak=9.6GB, used=8GB, needed=1.6GB
        # free=4GB >= 1.6*1.5=2.4GB → OK
        result = predict_survival(
            phase="growing",
            framework="ollama",
            cmdline="ollama serve",
            model_size_bytes=_gb(8),
            process_used_bytes=_gb(8),
            gpu_free_bytes=_gb(4),
            scrape_data={
                "models": [
                    {"name": "llama2:7b", "size": _gb(4), "size_vram": _gb(4),
                     "family": "llama", "parameter_size": "7B", "quantization_level": "Q4_0"},
                    {"name": "mistral:7b", "size": _gb(4), "size_vram": _gb(4),
                     "family": "mistral", "parameter_size": "7B", "quantization_level": "Q4_0"},
                ]
            },
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "sufficient headroom"

    def test_sglang_pool_ok(self) -> None:
        """SGLang with token pool and sufficient free memory should return OK."""
        result = predict_survival(
            phase="growing",
            framework="sglang",
            cmdline="python -m sglang.launch_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(2),
            scrape_data={
                "model_path": "/models/llama-7b",
                "max_total_num_tokens": 32768,
                "context_length": 4096,
                "is_generation": True,
            },
        )
        assert result.verdict == Verdict.OK
        assert "SGLang pool" in result.reason

    def test_sglang_pool_tight(self) -> None:
        """SGLang with token pool and low free memory should return TIGHT."""
        result = predict_survival(
            phase="growing",
            framework="sglang",
            cmdline="python -m sglang.launch_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(14),
            gpu_free_bytes=_mb(100),
            scrape_data={
                "model_path": "/models/llama-7b",
                "max_total_num_tokens": 32768,
                "context_length": 4096,
                "is_generation": True,
            },
        )
        assert result.verdict == Verdict.TIGHT
        assert "SGLang pool" in result.reason

    def test_no_scrape_data_falls_through(self) -> None:
        """No scrape data should fall through to multiplier heuristic."""
        result = predict_survival(
            phase="growing",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(10),
            scrape_data=None,
        )
        # Falls through to multiplier: vllm=1.8x, peak=7.2GB, needed=3.2GB
        # free=10GB >= 3.2*1.5=4.8GB → OK with headroom
        assert result.verdict == Verdict.OK
        assert result.reason == "sufficient headroom"

    def test_unknown_framework_scrape_data_ignored(self) -> None:
        """Scrape data for unknown framework should be ignored."""
        result = predict_survival(
            phase="stable",
            framework=None,
            cmdline="python my_script.py",
            model_size_bytes=_gb(2),
            process_used_bytes=_gb(2),
            gpu_free_bytes=_gb(4),
            scrape_data={"gpu_cache_usage_perc": 0.95},
        )
        # framework=None, so scrape_data is skipped; stable → OK
        assert result.verdict == Verdict.OK
        assert result.reason == "stable"


class TestPeakTracking:
    """Test historical peak memory tracking in survival prediction."""

    def test_peak_overrides_multiplier_when_higher(self) -> None:
        """If historical peak exceeds multiplier estimate, use the peak."""
        # model=2GB, multiplier=1.2 (ollama), estimate=2.4GB
        # But peak was 5GB → use 5GB as estimate
        # used=3GB, needed=5-3=2GB, free=1GB < 2GB → OOM
        result = predict_survival(
            phase="growing",
            framework="ollama",
            cmdline="ollama serve",
            model_size_bytes=_gb(2),
            process_used_bytes=_gb(3),
            gpu_free_bytes=_gb(1),
            peak_used_bytes=_gb(5),
        )
        assert result.verdict == Verdict.OOM

    def test_peak_ignored_when_lower_than_multiplier(self) -> None:
        """If historical peak is below multiplier estimate, use multiplier."""
        # model=4GB, multiplier=4.0 (adam), estimate=16GB
        # peak was 5GB (lower) → still use 16GB
        # used=4GB, needed=12GB, free=1GB → OOM
        result = predict_survival(
            phase="growing",
            framework="pytorch",
            cmdline="python train.py --optimizer adam",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(4),
            gpu_free_bytes=_gb(1),
            peak_used_bytes=_gb(5),
        )
        assert result.verdict == Verdict.OOM

    def test_no_peak_data_uses_multiplier(self) -> None:
        """Without peak data, falls back to multiplier estimate."""
        result = predict_survival(
            phase="growing",
            framework="ollama",
            cmdline="ollama serve",
            model_size_bytes=_gb(2),
            process_used_bytes=_gb(2),
            gpu_free_bytes=_gb(5),
            peak_used_bytes=None,
        )
        # multiplier=1.2, peak=2.4GB, needed=0.4GB, free=5GB → OK
        assert result.verdict == Verdict.OK


class TestEstimatePeak:
    """Test the estimate_peak helper function."""

    def test_basic_multiplier(self) -> None:
        """Should return base * multiplier."""
        peak = estimate_peak(
            framework="ollama", cmdline="ollama serve",
            model_size_bytes=_gb(4), process_used_bytes=_gb(2),
        )
        assert peak == int(_gb(4) * 1.2)

    def test_peak_overrides(self) -> None:
        """Historical peak overrides when higher."""
        peak = estimate_peak(
            framework="ollama", cmdline="ollama serve",
            model_size_bytes=_gb(4), process_used_bytes=_gb(2),
            peak_used_bytes=_gb(10),
        )
        assert peak == _gb(10)

    def test_no_model_size_uses_process_used(self) -> None:
        """Without model_size_bytes, uses process_used_bytes as base."""
        peak = estimate_peak(
            framework=None, cmdline=None,
            model_size_bytes=None, process_used_bytes=_gb(2),
        )
        assert peak == int(_gb(2) * 1.5)  # generic multiplier


class TestCollectivePressure:
    """Test collective pressure check across multiple processes."""

    def test_no_pressure_when_under_total(self) -> None:
        """Should not upgrade when sum of peaks <= GPU total."""
        preds = {
            1: SurvivalPrediction(Verdict.OK, "stable"),
            2: SurvivalPrediction(Verdict.OK, "stable"),
        }
        peaks = {1: _gb(4), 2: _gb(4)}
        result = check_collective_pressure(preds, peaks, _gb(16))
        assert result[1].verdict == Verdict.OK
        assert result[2].verdict == Verdict.OK

    def test_ok_upgraded_to_tight(self) -> None:
        """OK processes should be upgraded to TIGHT when collective overcommit."""
        preds = {
            1: SurvivalPrediction(Verdict.OK, "stable"),
            2: SurvivalPrediction(Verdict.OK, "stable"),
        }
        # peaks sum = 12GB > 8GB total
        peaks = {1: _gb(6), 2: _gb(6)}
        result = check_collective_pressure(preds, peaks, _gb(8))
        assert result[1].verdict == Verdict.TIGHT
        assert "overcommit" in result[1].reason

    def test_tight_upgraded_to_oom(self) -> None:
        """TIGHT processes should be upgraded to OOM when collective overcommit."""
        preds = {
            1: SurvivalPrediction(Verdict.TIGHT, "marginal"),
            2: SurvivalPrediction(Verdict.TIGHT, "marginal"),
        }
        peaks = {1: _gb(6), 2: _gb(6)}
        result = check_collective_pressure(preds, peaks, _gb(8))
        assert result[1].verdict == Verdict.OOM
        assert result[2].verdict == Verdict.OOM

    def test_oom_stays_oom(self) -> None:
        """OOM processes should stay OOM (no further upgrade)."""
        preds = {
            1: SurvivalPrediction(Verdict.OOM, "already doomed"),
        }
        peaks = {1: _gb(20)}
        result = check_collective_pressure(preds, peaks, _gb(8))
        assert result[1].verdict == Verdict.OOM
        assert result[1].reason == "already doomed"

    def test_empty_peaks(self) -> None:
        """Should return predictions unchanged when no peaks provided."""
        preds = {1: SurvivalPrediction(Verdict.OK, "stable")}
        result = check_collective_pressure(preds, {}, _gb(16))
        assert result[1].verdict == Verdict.OK


class TestAbsoluteHeadroom:
    """Test absolute headroom checks — the fixes for missed OOM detection.

    These test the scenario where a process is STABLE (between training steps)
    but free memory is critically low.  Previously, stable → OK unconditionally.
    """

    def test_user_oom_scenario(self) -> None:
        """Reproduce the exact scenario that caused undetected OOM.

        GPU: 15.70 GiB total, process using 15.30 GiB, 405 MiB free.
        Phase: stable (between training steps).
        Previous behavior: OK "stable" — wrong!
        Fixed behavior: TIGHT (only 2.5% free).
        """
        result = predict_survival(
            phase="stable",
            framework="pytorch",
            cmdline="python scripts/train_test.py",
            model_size_bytes=_mb(500),  # GPT-2 weights
            process_used_bytes=_gb(15.3),
            gpu_free_bytes=_mb(405),
            gpu_total_bytes=int(15.70 * 1024 * 1024 * 1024),
        )
        assert result.verdict == Verdict.TIGHT
        assert "free" in result.reason

    def test_stable_with_plenty_of_room_still_ok(self) -> None:
        """Stable phase with >5% free should still return OK."""
        result = predict_survival(
            phase="stable",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(2),
            process_used_bytes=_gb(8),
            gpu_free_bytes=_gb(6),
            gpu_total_bytes=_gb(16),
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "stable"

    def test_stable_low_free_pct_triggers_tight(self) -> None:
        """Stable phase with <5% free should return TIGHT."""
        # 14.5 GB used, 0.5 GB free on 16 GB GPU = 3.1% free
        result = predict_survival(
            phase="stable",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(1),
            process_used_bytes=_gb(14.5),
            gpu_free_bytes=_mb(500),
            gpu_total_bytes=_gb(16),
        )
        assert result.verdict == Verdict.TIGHT
        assert "free" in result.reason

    def test_critical_free_pct_triggers_tight_any_phase(self) -> None:
        """<2% free should return TIGHT even during growing phase.

        This catches the case before the multiplier heuristic runs.
        """
        result = predict_survival(
            phase="growing",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(1),
            process_used_bytes=_gb(15.5),
            gpu_free_bytes=_mb(200),
            gpu_total_bytes=_gb(16),
        )
        assert result.verdict in (Verdict.TIGHT, Verdict.OOM)

    def test_pre_alloc_exempt_from_headroom_check(self) -> None:
        """Pre-allocating frameworks (vLLM) should not trigger low-free warnings.

        These frameworks deliberately run at high utilization and manage
        their own memory pools.
        """
        result = predict_survival(
            phase="stable",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(14),
            gpu_free_bytes=_mb(300),
            gpu_total_bytes=_gb(16),
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "stable"

    def test_no_gpu_total_skips_headroom_check(self) -> None:
        """Without gpu_total_bytes (=0), headroom check is skipped for compat."""
        result = predict_survival(
            phase="stable",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(1),
            process_used_bytes=_gb(15),
            gpu_free_bytes=_mb(200),
            gpu_total_bytes=0,
        )
        # Without total, can't compute %, falls through to stable → OK
        assert result.verdict == Verdict.OK
        assert result.reason == "stable"


class TestSpikeDetection:
    """Test memory spike detection from peak tracking."""

    def test_spike_exceeds_free_triggers_oom(self) -> None:
        """If observed spike > free memory, predict OOM.

        Training loops have sawtooth memory: forward pass allocates activations,
        backward pass peaks, then memory drops.  If the spike size exceeds
        available free memory, the next forward pass will OOM.
        """
        result = predict_survival(
            phase="stable",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(1),
            process_used_bytes=_gb(10),
            gpu_free_bytes=_gb(1),
            gpu_total_bytes=_gb(16),
            peak_used_bytes=_gb(12),  # Spike of 2 GB > 1 GB free
        )
        assert result.verdict == Verdict.OOM
        assert "spike" in result.reason

    def test_spike_within_free_no_trigger(self) -> None:
        """If observed spike < free memory, spike check doesn't trigger."""
        result = predict_survival(
            phase="stable",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(1),
            process_used_bytes=_gb(8),
            gpu_free_bytes=_gb(4),
            gpu_total_bytes=_gb(16),
            peak_used_bytes=_gb(10),  # Spike of 2 GB < 4 GB free
        )
        assert result.verdict == Verdict.OK
        assert result.reason == "stable"

    def test_spike_pre_alloc_exempt(self) -> None:
        """Pre-allocating frameworks should not trigger spike detection."""
        result = predict_survival(
            phase="stable",
            framework="vllm",
            cmdline="python -m vllm.entrypoints.openai.api_server",
            model_size_bytes=_gb(4),
            process_used_bytes=_gb(10),
            gpu_free_bytes=_gb(1),
            gpu_total_bytes=_gb(16),
            peak_used_bytes=_gb(14),  # Spike of 4 GB > 1 GB free
        )
        # vLLM is pre-allocating → exempt from spike detection
        assert result.verdict == Verdict.OK

    def test_no_peak_no_spike(self) -> None:
        """Without peak data, spike detection doesn't trigger."""
        result = predict_survival(
            phase="stable",
            framework="pytorch",
            cmdline="python train.py",
            model_size_bytes=_gb(1),
            process_used_bytes=_gb(14),
            gpu_free_bytes=_gb(1.5),
            gpu_total_bytes=_gb(16),
            peak_used_bytes=None,
        )
        # No peak → no spike → stable + 9.4% free → OK
        assert result.verdict == Verdict.OK


class TestSurvivalPrediction:
    """Test SurvivalPrediction dataclass."""

    def test_frozen(self) -> None:
        """SurvivalPrediction should be immutable."""
        pred = SurvivalPrediction(Verdict.OK, "stable")
        try:
            pred.verdict = Verdict.OOM  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_equality(self) -> None:
        """Two identical predictions should be equal."""
        a = SurvivalPrediction(Verdict.TIGHT, "low headroom")
        b = SurvivalPrediction(Verdict.TIGHT, "low headroom")
        assert a == b
