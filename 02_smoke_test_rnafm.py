"""
02_smoke_test_rnafm.py — RIBOSCOPE end-to-end smoke test

Purpose
-------
This is the BIG checkpoint of Phase 5. It downloads the RNA-FM model from
Hugging Face, moves it to your GPU, and runs a forward pass on a real,
disease-relevant RNA sequence (HIV-1 TAR — a well-studied stem-loop in
the HIV genome). If this script finishes without errors, your local
environment is fully ready for the real research work.

Why this specific test
----------------------
- HIV-1 TAR is small (~30 nucleotides), so the test is fast.
- It has a known, well-characterized hairpin secondary structure, so we
  know what biology *should* be in the activations.
- It's disease-relevant, which is the kind of sequence we'll target later.
- It's appeared in 100+ RNA-structure papers, so reproducing a forward
  pass on it is a known-good calibration.

What the script does
--------------------
1. Loads the RNA-FM tokenizer and model from HuggingFace
   (downloads ~500MB the first time; cached after that).
2. Moves the model to your RTX 5060 Ti.
3. Tokenizes the HIV-1 TAR sequence.
4. Runs a forward pass and reports the embedding shape.
5. Reports VRAM usage and timing.

Run with
--------
    uv run python 02_smoke_test_rnafm.py

Expected output (numbers will vary slightly)
--------------------------------------------
=== RIBOSCOPE Smoke Test: RNA-FM ===
[1/5] Loading tokenizer + model from HuggingFace...
[2/5] Moving model to GPU...
   Using device: cuda
   GPU: NVIDIA GeForce RTX 5060 Ti
[3/5] Tokenizing test sequence (HIV-1 TAR, 29 nt)...
   Sequence length: 29
   Tokenized shape: torch.Size([1, 31])  # +2 for special tokens
[4/5] Running forward pass...
   Forward pass time: 0.0XX seconds
   Output embeddings shape: torch.Size([1, 31, 640])
   First 5 values of position 0: tensor([...])
[5/5] Memory check...
   GPU VRAM used: ~1.4 GB / 16.0 GB

✅ SMOKE TEST PASSED! RNA-FM forward pass works on your GPU.
"""

import sys
import time

# --------------------------------------------------------------------- #
# Imports — wrapped in try/except so the user gets a friendly error
# rather than a stack trace if a dependency is missing.
# --------------------------------------------------------------------- #
try:
    import torch
except ImportError:
    print("❌ PyTorch not installed. Run step 3.1 in GETTING_STARTED.md.")
    sys.exit(1)

try:
    from multimolecule import RnaTokenizer, RnaFmModel
except ImportError:
    print("❌ multimolecule not installed. Run step 3.3 in GETTING_STARTED.md:")
    print("   uv pip install transformers multimolecule huggingface_hub accelerate datasets")
    sys.exit(1)

# --------------------------------------------------------------------- #
# Test sequence — HIV-1 TAR (Trans-Activation Response element)
# This is the RNA hairpin at the 5' end of all HIV-1 transcripts.
# Sequence from the HIV-1 NL4-3 reference, positions ~454–482.
# Has a well-characterized stem-loop with a UCU bulge near the apex.
# --------------------------------------------------------------------- #
HIV1_TAR = "GGCAGAUCUGAGCCUGGGAGCUCUCUGCC"
TEST_SEQUENCE_NAME = "HIV-1 TAR (Trans-Activation Response element)"


def main() -> None:
    """Run the end-to-end RNA-FM forward pass smoke test."""
    print("=" * 60)
    print("RIBOSCOPE Smoke Test: RNA-FM")
    print("=" * 60)

    # ------------------------------------------------------------------
    # [1/5] Load the model
    # ------------------------------------------------------------------
    print("[1/5] Loading tokenizer + model from HuggingFace...")
    print("      (First run downloads ~500MB — may take 1-3 minutes)")

    # `from_pretrained()` downloads weights to the HF cache (~/.cache/huggingface)
    # and reuses them on subsequent runs.
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model = RnaFmModel.from_pretrained("multimolecule/rnafm")

    # eval() disables dropout etc. — we only do inference, not training.
    model.eval()

    # ------------------------------------------------------------------
    # [2/5] Move model to GPU
    # ------------------------------------------------------------------
    print("[2/5] Moving model to GPU...")

    if not torch.cuda.is_available():
        print("   ⚠ CUDA not available, falling back to CPU (will be slow).")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   Using device: cuda")
        print(f"   GPU: {gpu_name}")

    model = model.to(device)

    # ------------------------------------------------------------------
    # [3/5] Tokenize the test sequence
    # ------------------------------------------------------------------
    print(f"[3/5] Tokenizing test sequence ({TEST_SEQUENCE_NAME})...")
    print(f"      Sequence: {HIV1_TAR}")
    print(f"      Sequence length: {len(HIV1_TAR)} nucleotides")

    # `return_tensors="pt"` gives us PyTorch tensors. The tokenizer adds
    # special tokens (CLS at start, EOS at end), so the output is +2 longer.
    inputs = tokenizer(HIV1_TAR, return_tensors="pt")
    print(f"      Tokenized shape: {inputs['input_ids'].shape}")
    print(f"      (Length is +2 vs. raw — the tokenizer adds [CLS] and [EOS] tokens.)")

    # Move the input tensors to the same device as the model.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # [4/5] Forward pass
    # ------------------------------------------------------------------
    print("[4/5] Running forward pass...")

    # `torch.no_grad()` tells PyTorch not to track gradients — saves VRAM
    # and is the right thing to do for inference-only.
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.perf_counter() - start

    # `last_hidden_state` is the embedding tensor at the final layer.
    # Shape is [batch_size, seq_len, hidden_dim]. For RNA-FM, hidden_dim=640.
    embeddings = outputs.last_hidden_state

    print(f"      Forward pass time: {elapsed:.3f} seconds")
    print(f"      Output embeddings shape: {embeddings.shape}")
    print(f"      First 5 values at position 0: {embeddings[0, 0, :5].cpu().tolist()}")

    # Sanity check on the shape — should be [1, 31, 640].
    expected_seq_len = len(HIV1_TAR) + 2  # +2 for [CLS] and [EOS]
    expected_hidden = 640                  # RNA-FM hidden dim
    assert embeddings.shape == (1, expected_seq_len, expected_hidden), (
        f"Unexpected output shape! Got {embeddings.shape}, "
        f"expected (1, {expected_seq_len}, {expected_hidden})."
    )

    # ------------------------------------------------------------------
    # [5/5] Memory report
    # ------------------------------------------------------------------
    print("[5/5] Memory check...")
    if device.type == "cuda":
        used_gb = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"      GPU VRAM used: {used_gb:.2f} GB / {total_gb:.1f} GB")
        # Flag if we're using > 4 GB on a tiny sequence (something would be wrong).
        if used_gb > 4.0:
            print("      ⚠  Higher than expected. Should be ~1-2 GB for this small input.")

    print()
    print("✅ SMOKE TEST PASSED! RNA-FM forward pass works on your GPU.")
    print("   You are now ready to start the real project work.")
    print()
    print("   Next steps (Phase 8+, future doc):")
    print("   - Install sae_lens and transformer_lens")
    print("   - Build the activation extraction pipeline")
    print("   - Train your first BatchTopK SAE on layer 6 of RNA-FM")


if __name__ == "__main__":
    main()
