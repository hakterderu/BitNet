# bitnet.cpp
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-1.0-blue)

[<img src="./assets/header_model_release.png" alt="BitNet Model on Hugging Face" width="800"/>](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)

Try it out via this [demo](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/), or build and run it on your own [CPU](https://github.com/microsoft/BitNet?tab=readme-ov-file#build-from-source) or [GPU](https://github.com/microsoft/BitNet/blob/main/gpu/README.md).

bitnet.cpp is the official inference framework for 1-bit LLMs (e.g., BitNet b1.58). It offers a suite of optimized kernels, that support **fast** and **lossless** inference of 1.58-bit models on CPU and GPU (NPU support will coming next).

The first release of bitnet.cpp is to support inference on CPUs. bitnet.cpp achieves speedups of **1.37x** to **5.07x** on ARM CPUs, with larger models experiencing greater performance gains. Additionally, it reduces energy consumption by **55.4%** to **70.0%**, further boosting overall efficiency. On x86 CPUs, speedups range from **2.37x** to **6.17x** with energy reductions between **71.9%** to **82.2%**. Furthermore, bitnet.cpp can run a 100B BitNet b1.58 model on a single CPU, achieving speeds comparable to human reading (5-7 tokens per second), significantly enhancing the potential for running LLMs on local devices. Please refer to the [technical report](https://arxiv.org/abs/2410.16144) for more details.

**Latest optimization** introduces parallel kernel implementations with configurable tiling and embedding quantization support, achieving **1.15x to 2.1x** additional speedup over the original implementation across different hardware platforms and workloads. For detailed technical information, see the [optimization guide](src/README.md).

<img src="./assets/performance.png" alt="performance_comparison" width="800"/>


## Demo

A demo of bitnet.cpp running a BitNet b1.58 3B model on Apple M2:

https://github.com/user-attachments/assets/7f46b736-edec-4828-b809-4be780a3e5b1

## Personal Notes

> **Fork notes (for my own reference):** I'm using this primarily to experiment with the 2B model on my Linux x86 machine. The build instructions in the section below worked out of the box for me on Ubuntu 22.04 with GCC 12. Main area of interest: comparing throughput with different thread counts via `-t` flag.
>
> **Thread count findings (Ubuntu 22.04, Ryzen 7 5800X, 8 cores / 16 threads):**
> - `-t 4`: ~9.2 tok/s
> - `-t 8`: ~14.7 tok/s  ← sweet spot for me
> - `-t 16`: ~13.1 tok/s (hyperthreading hurts slightly here)
> Best results with `-t 8 -ngl 0` on the 2B model.
>
> **Context length findings:** Tested `-c` values on the 2B model. Going above 2048 noticeably tanks throughput (~11.3 tok/s at `-c 4096`). Sticking with `-c 2048` as my default for interactive use.
>
> **Prompt length findings:** Short prompts (<50 tokens) show noticeably faster time-to-first-token than longer ones (200+ tokens). For interactive/chat use, keeping system prompts concise makes a real difference in perceived responsiveness. Prefill seems to be the bottleneck more than generation on this hardware.
>
> **My go-to launch command:**
> ```
> python run_inference.py -m models/bitnet-2b -p "Your prompt here" -t 8 -c 2048 -n 200
> ```

## What's New:
- 01/15/2026 [BitNet CPU Inferenc
