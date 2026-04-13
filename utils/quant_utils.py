"""Quantization utilities for BitNet models.

Provides helper functions for quantization type detection, conversion,
and validation specific to BitNet's 1-bit and 1.58-bit weight formats.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Supported quantization types and their descriptions
QUANT_TYPES = {
    "i2_s": "2-bit signed integer (BitNet b1.58)",
    "tl1": "Ternary lookup table type 1",
    "tl2": "Ternary lookup table type 2",
    "q4_0": "4-bit quantization (baseline)",
    "q8_0": "8-bit quantization (baseline)",
}

# Mapping from model architecture to recommended quant type
ARCH_QUANT_MAP = {
    "bitnet": "i2_s",
    "bitnet_b1_58": "tl1",
    "llama": "tl2",
    "falcon": "q4_0",
}


def get_recommended_quant_type(arch: str) -> str:
    """Return the recommended quantization type for a given architecture.

    Args:
        arch: Model architecture string (e.g. 'bitnet', 'llama').

    Returns:
        Recommended quantization type string.
    """
    arch_lower = arch.lower()
    for key, quant in ARCH_QUANT_MAP.items():
        if key in arch_lower:
            return quant
    logger.warning(
        "No recommended quant type found for architecture '%s'. Defaulting to 'i2_s'.",
        arch,
    )
    return "i2_s"


def is_bitnet_quantization(quant_type: str) -> bool:
    """Check whether the given quantization type is a native BitNet format.

    Args:
        quant_type: Quantization type string.

    Returns:
        True if the quant type is a BitNet-native format, False otherwise.
    """
    return quant_type in ("i2_s", "tl1", "tl2")


def get_quant_description(quant_type: str) -> str:
    """Return a human-readable description for a quantization type.

    Args:
        quant_type: Quantization type string.

    Returns:
        Description string, or a generic message if type is unknown.
    """
    return QUANT_TYPES.get(quant_type, f"Unknown quantization type: {quant_type}")


def estimate_quantized_size_gb(
    original_size_gb: float, quant_type: str
) -> Tuple[float, float]:
    """Estimate the size of a quantized model and the compression ratio.

    Args:
        original_size_gb: Original model size in gigabytes (float32 baseline).
        quant_type: Target quantization type.

    Returns:
        Tuple of (estimated_size_gb, compression_ratio).
    """
    # Approximate bits-per-weight for each quant type relative to fp32 (32 bits)
    bpw_map = {
        "i2_s": 2.0,
        "tl1": 1.58,
        "tl2": 1.58,
        "q4_0": 4.0,
        "q8_0": 8.0,
    }
    bpw = bpw_map.get(quant_type, 8.0)
    compression_ratio = 32.0 / bpw
    estimated_size_gb = original_size_gb / compression_ratio
    return round(estimated_size_gb, 3), round(compression_ratio, 2)


def load_quant_config(model_dir: str) -> Optional[dict]:
    """Attempt to load quantization config from a model directory.

    Looks for a 'quant_config.json' or 'config.json' file and extracts
    quantization-related fields.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Dictionary with quant config fields, or None if not found.
    """
    model_path = Path(model_dir)
    for config_name in ("quant_config.json", "config.json"):
        config_file = model_path / config_name
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                quant_fields = {
                    k: v
                    for k, v in config.items()
                    if any(
                        kw in k.lower()
                        for kw in ("quant", "bits", "weight_bits", "activation_bits")
                    )
                }
                if quant_fields:
                    logger.debug(
                        "Loaded quant config from '%s': %s", config_file, quant_fields
                    )
                    return quant_fields
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read config file '%s': %s", config_file, e)
    return None
