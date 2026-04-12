"""Setup script for BitNet - A fork of microsoft/BitNet.

This package provides tools for running inference with BitNet models,
including support for 1-bit and 1.58-bit quantized language models.
"""

from setuptools import setup, find_packages
import os

# Read the README for the long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies required for running BitNet inference
INSTALL_REQUIRES = [
    "numpy>=1.24.0",
    "torch>=2.0.0",
    "transformers>=4.36.0",
    "huggingface_hub>=0.20.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "tqdm>=4.65.0",
    "requests>=2.28.0",
]

# Optional dependencies for development and testing
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ],
    "convert": [
        "gguf>=0.6.0",
        "safetensors>=0.4.0",
    ],
    "server": [
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
}

# Combine all optional deps under 'all'
EXTRAS_REQUIRE["all"] = [
    dep for group in EXTRAS_REQUIRE.values() for dep in group
]

setup(
    name="bitnet",
    version="0.1.0",
    description="Efficient inference framework for 1-bit and 1.58-bit quantized language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="BitNet Contributors",
    license="MIT",
    url="https://github.com/microsoft/BitNet",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence],
    keywords bitnet 1-bit transformer",
    packagesexclude=["tests*", "INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "bitnet-run=bitnet.run_inference:main",
            "bitnet-convert=bitnet.utils.convert:main",
            "bitnet-setup=bitnet.setup_env:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bitnet": ["*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
)
