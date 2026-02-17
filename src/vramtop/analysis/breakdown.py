"""Memory breakdown estimation: weights vs dynamic (labeled as estimate)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MemoryBreakdown:
    """Estimated memory breakdown for a GPU process."""

    total_bytes: int
    estimated_weights_bytes: int | None
    estimated_dynamic_bytes: int | None
    confidence: float
    label: str  # Always contains "estimate"


def estimate_breakdown(
    total_used: int, model_file_sizes: list[int]
) -> MemoryBreakdown:
    """Estimate weight vs dynamic memory split.

    Uses sum of model file sizes as the weight estimate, with the
    remainder attributed to dynamic allocations (activations, KV cache,
    optimizer state, etc.).

    This is always an estimate -- model files on disk may be
    compressed, quantized differently, or only partially loaded.
    """
    if not model_file_sizes or total_used <= 0:
        return MemoryBreakdown(
            total_bytes=max(total_used, 0),
            estimated_weights_bytes=None,
            estimated_dynamic_bytes=None,
            confidence=0.0,
            label="estimate: insufficient data",
        )

    weights_sum = sum(model_file_sizes)
    # Clamp: weights cannot exceed total usage
    weights_est = min(weights_sum, total_used)
    dynamic_est = total_used - weights_est

    # Confidence based on how close file sizes are to total
    # High confidence when weights are 30-90% of total
    ratio = weights_est / total_used if total_used > 0 else 0.0
    if 0.3 <= ratio <= 0.9:
        confidence = 0.7
    elif 0.1 <= ratio <= 0.95:
        confidence = 0.4
    else:
        confidence = 0.2

    return MemoryBreakdown(
        total_bytes=total_used,
        estimated_weights_bytes=weights_est,
        estimated_dynamic_bytes=dynamic_est,
        confidence=confidence,
        label="estimate: weights vs dynamic",
    )
