"""Utility functions for BitNet model management and conversion.

Provides helpers for loading, converting, and managing BitNet models,
including quantization utilities and model inspection tools.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

# Supported model architectures
SUPPORTED_ARCHITECTURES = [
    "bitnet",
    "llama",
    "mistral",
    "falcon",
    "phi",  # added phi support - useful for smaller experiments
]

# Default quantization configurations
QUANT_CONFIGS = {
    "i2_s": {"bits": 2, "group_size": -1, "desc": "2-bit symmetric quantization"},
    "tl1": {"bits": 2, "group_size": 8, "desc": "Ternary lookup table (1.58-bit)"},
    "tl2": {"bits": 2, "group_size": 16, "desc": "Ternary lookup table extended"},
}


def get_model_config(model_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load and return the model configuration from a directory.

    Args:
        model_dir: Path to the directory containing config.json.

    Returns:
        Dictionary containing the model configuration.

    Raises:
        FileNotFoundError: If config.json is not found in model_dir.
        ValueError: If the config file is malformed.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed config.json in {model_dir}: {e}") from e

    logger.debug("Loaded model config from %s", config_path)
    return config


def detect_model_architecture(config: Dict[str, Any]) -> Optional[str]:
    """Detect the model architecture from its configuration.

    Args:
        config: Model configuration dictionary.

    Returns:
        Architecture name string, or None if not recognized.
    """
    arch = config.get("model_type", "").lower()
    if arch in SUPPORTED_ARCHITECTURES:
        return arch

    # Fallback: check architectures list
    architectures = config.get("architectures", [])
    for a in architectures:
        a_lower = a.lower()
        for supported in SUPPORTED_ARCHITECTURES:
            if supported in a_lower:
                return supported

    logger.warning("Could not detect architecture from config: %s", config)
    return None


def validate_quant_type(quant_type: str) -> bool:
    """Validate that a quantization type is supported.

    Args:
        quant_type: Quantization type string (e.g., 'i2_s', 'tl1').

    Returns:
        True if the quantization type is supported, False otherwise.
    """
    return quant_type in QUANT_CONFIGS


def get_model_size_gb(model_dir: Union[str, Path]) -> float:
    """Estimate the total size of model files in gigabytes.

    Args:
        model_dir: Path to the directory containing model files.

    Returns:
        Total size of .bin and .safetensors files in gigabytes.
    """
    model_dir = Path(model_dir)
    # Include .safetensors in addition to .bin - most modern models use this format
    total_bytes = sum(
        f.stat().st_size
        for f in model_dir.iterdir()
        if f.suffix in (".bin", ".safetensors")
    )
    return total_bytes / (1024 ** 3)
