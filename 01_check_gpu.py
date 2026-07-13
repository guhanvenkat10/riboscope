"""
01_check_gpu.py — RIBOSCOPE GPU verification

Purpose
-------
Confirms PyTorch is installed correctly and can see your RTX 5060 Ti
through WSL2's NVIDIA driver passthrough. Run this BEFORE the full smoke
test (02_smoke_test_rnafm.py), because if this fails, nothing else will work.

What it does
------------
1. Prints PyTorch version and the CUDA version it was built against.
2. Checks `torch.cuda.is_available()`.
3. Prints details for every visible GPU.
4. Allocates a small tensor on the GPU and runs a trivial computation
   (this is the real test — `is_available()` can lie if the wrong CUDA
   version is installed).
5. Reports VRAM usage.

Run with
--------
    uv run python 01_check_gpu.py

Expected output
---------------
PyTorch version: 2.6.x or newer
CUDA available: True
CUDA version (compiled): 12.8
Number of GPUs: 1
GPU 0: NVIDIA GeForce RTX 5060 Ti
GPU 0 memory: 16.0 GB
Test tensor on GPU: tensor([1., 2., 3.], device='cuda:0')
Sum on GPU: 6.0
✅ GPU is working!

If anything fails, see the troubleshooting section in
RIBOSCOPE_GETTING_STARTED.md, Phase 3.
"""

import sys

# We import inside try/except so the user gets a clear error if PyTorch
# isn't installed yet, rather than a cryptic ImportError stack trace.
try:
    import torch
except ImportError:
    print("❌ PyTorch is not installed in this environment.")
    print("   Run (from inside ~/projects/riboscope):")
    print("     uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)


def main() -> None:
    """Run all GPU diagnostic checks."""
    print("=" * 60)
    print("RIBOSCOPE — GPU verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Versions
    # ------------------------------------------------------------------
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version (compiled): {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # ------------------------------------------------------------------
    # 2. Bail early if no CUDA
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print()
        print("❌ CUDA is NOT available to PyTorch.")
        print("   Likely causes:")
        print("   - You installed a CPU-only PyTorch wheel.")
        print("     Fix: uv pip install torch torchvision \\")
        print("            --index-url https://download.pytorch.org/whl/cu128")
        print("   - Your Windows NVIDIA driver is too old for WSL passthrough.")
        print("     Fix: update via the NVIDIA App in Windows, restart Windows.")
        print("   - You're running this outside WSL on a system with no GPU.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Per-GPU info
    # ------------------------------------------------------------------
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        # `total_memory` is in bytes; convert to GB for readability.
        total_mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        # Capability (e.g. (8, 9) for Ada, (12, 0) for Blackwell)
        cap = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {name}")
        print(f"GPU {i} memory: {total_mem_gb:.1f} GB")
        print(f"GPU {i} compute capability: sm_{cap[0]}{cap[1]}")

    # ------------------------------------------------------------------
    # 4. The real test: allocate + compute on GPU
    # ------------------------------------------------------------------
    # `is_available()` can return True even when the actual CUDA kernels
    # are missing for your GPU's architecture. The only way to be sure
    # is to actually run something on it.
    try:
        device = torch.device("cuda:0")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        s = x.sum().item()
        print(f"Test tensor on GPU: {x}")
        print(f"Sum on GPU: {s}")
    except RuntimeError as e:
        print()
        print("❌ GPU is visible but compute failed.")
        print(f"   Error: {e}")
        print("   Likely cause: your PyTorch build doesn't have kernels for Blackwell (sm_120).")
        print("   Fix: reinstall PyTorch with the cu128 index URL (see Phase 3.1).")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. VRAM usage report
    # ------------------------------------------------------------------
    used_gb = torch.cuda.memory_allocated() / 1e9
    cached_gb = torch.cuda.memory_reserved() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM allocated: {used_gb:.3f} GB / {total_gb:.1f} GB")
    print(f"VRAM cached:    {cached_gb:.3f} GB")

    print()
    print("✅ GPU is working! You're ready for the next step (02_smoke_test_rnafm.py).")


if __name__ == "__main__":
    main()
