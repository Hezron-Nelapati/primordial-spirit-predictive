"""
gpu_utils.py — single source of truth for device selection.

Priority order:
  1. CUDA  — NVIDIA GPUs (Linux, Windows)
  2. MPS   — Apple Silicon M-series (macOS 12.3+, PyTorch ≥ 1.12)
  3. XPU   — Intel Arc / Xe GPUs (Linux/Windows, PyTorch ≥ 2.4)
  4. CPU   — universal fallback

Usage:
    from gpu_utils import get_device
    model = SentenceTransformer("...", device=get_device())
"""
from __future__ import annotations


def get_device() -> str:
    """Return the best available compute device as a PyTorch device string."""
    try:
        import torch
    except ImportError:
        return "cpu"

    # 1. NVIDIA CUDA
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  [gpu_utils] CUDA available — {name}")
        return "cuda"

    # 2. Apple Silicon MPS (Metal Performance Shaders)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("  [gpu_utils] MPS available — Apple Silicon GPU")
        return "mps"

    # 3. Intel XPU (Arc / Xe GPUs, requires intel-extension-for-pytorch)
    try:
        if torch.xpu.is_available():
            name = torch.xpu.get_device_name(0)
            print(f"  [gpu_utils] XPU available — {name}")
            return "xpu"
    except AttributeError:
        pass  # torch.xpu not present in this PyTorch build

    print("  [gpu_utils] No GPU found — using CPU")
    return "cpu"
