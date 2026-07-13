"""
03_extract_activations.py — RIBOSCOPE Phase 8: Activation extraction from RNA-FM

Purpose
-------
Take a list of RNA sequences (in FASTA format), run each through RNA-FM,
capture the hidden-state activations at several layers, and save them to
a single safetensors file. These activations are the substrate that
sparse autoencoders (Phase 9) will be trained on.

Why we do it this way
---------------------
- We use PyTorch forward hooks to "tap" specific layers of the model.
  When the model runs a forward pass, every hook fires and stores the
  layer's output. This is non-invasive — the model itself is untouched.

- We hook layers 1, 6, and 11 (early / mid / late). Different layers
  encode different abstractions. SAEs trained on different layers will
  reveal different features:
    * Early layers (1): nucleotide-level patterns, local k-mers
    * Middle layers (6): structural motifs (stem-loops, bulges)
    * Late layers (11): functional / family-level features

- We save in safetensors format because it's fast, type-safe, and the
  default for the Hugging Face / SAE ecosystem.

- We cast to float16 before saving — halves the disk footprint with
  negligible quality loss for downstream SAE training.

What this script does
---------------------
1. Loads RNA-FM and moves it to GPU.
2. Reads the FASTA file at sequences/test_rnas.fasta.
3. Registers forward hooks on layers 1, 6, 11 of the encoder.
4. Runs each sequence through; hooks capture activations.
5. Saves the activations as a single safetensors file at
   outputs/activations_rnafm_test.safetensors.

Run with
--------
    cd ~/projects/riboscope
    uv run python 03_extract_activations.py

Expected output
---------------
A `outputs/` folder with `activations_rnafm_test.safetensors` (a few MB),
plus a console log showing each sequence processed and the final tensor
count + file size.
"""

import sys
import time
from pathlib import Path

# --------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------- #
try:
    import torch
    from safetensors.torch import save_file
    from multimolecule import RnaTokenizer, RnaFmModel
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Make sure you ran the Phase 3.3 install step:")
    print("     uv pip install transformers multimolecule huggingface_hub accelerate datasets")
    sys.exit(1)


# --------------------------------------------------------------------- #
# Config — edit these to change behavior
# --------------------------------------------------------------------- #
LAYERS_TO_HOOK = [1, 6, 11]   # which encoder layers to tap (0-indexed)
FASTA_FILE = Path("sequences/test_rnas.fasta")
OUTPUT_FILE = Path("outputs/activations_rnafm_test.safetensors")
MODEL_NAME = "multimolecule/rnafm"


# --------------------------------------------------------------------- #
# A simple FASTA parser — no external dependency needed
# --------------------------------------------------------------------- #
def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """
    Read a FASTA file and return a list of (name, sequence) tuples.

    The 'name' is taken as the part of the header line *before* the
    first '|' delimiter — so '>HIV1_TAR | HIV-1 ... | 29 nt' becomes
    just 'HIV1_TAR'. We use this short name as the safetensors key.

    All T's are converted to U's (some sources use DNA notation even
    for RNA sequences; RNA-FM was trained on RNA so we standardize).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"FASTA file not found at {path}. "
            f"Did you copy sequences/test_rnas.fasta from the workspace? "
            f"See PHASE8 doc step 1."
        )

    sequences: list[tuple[str, str]] = []
    current_name: str | None = None
    current_seq: list[str] = []

    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save the previous record (if any)
                if current_name is not None:
                    sequences.append((current_name, "".join(current_seq)))
                # Parse the new header
                header = line[1:].strip()
                # Take the part before the first '|', strip whitespace
                current_name = header.split("|")[0].strip()
                current_seq = []
            else:
                # Accumulate sequence lines, converting any DNA T's to RNA U's
                current_seq.append(line.replace("T", "U").replace("t", "u"))

        # Don't forget the last record
        if current_name is not None:
            sequences.append((current_name, "".join(current_seq)))

    return sequences


# --------------------------------------------------------------------- #
# Hook factory — creates a closure that captures activations to a dict
# --------------------------------------------------------------------- #
def make_hook(captured: dict, layer_idx: int):
    """
    Return a forward hook function that stores the layer's output in
    `captured` under the key `layer_idx`. Cast to float16 + move to CPU
    so we don't accumulate VRAM as we process more sequences.
    """
    def hook(module, inputs, output):
        # Some HF transformer layers return a tuple; the first element
        # is always the hidden states tensor of shape [batch, seq, hidden].
        if isinstance(output, tuple):
            output = output[0]
        captured[layer_idx] = output.detach().to(torch.float16).cpu()
    return hook


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main() -> None:
    print("=" * 60)
    print("RIBOSCOPE Phase 8: Activation Extraction")
    print("=" * 60)

    # Make sure the output dir exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"[1/4] Loading {MODEL_NAME}...")
    tokenizer = RnaTokenizer.from_pretrained(MODEL_NAME)
    model = RnaFmModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"      Device: {device}")

    n_layers = len(model.encoder.layer)
    hidden_dim = model.config.hidden_size
    print(f"      RNA-FM has {n_layers} encoder layers, hidden_dim={hidden_dim}")

    # Validate the requested layer indices
    for li in LAYERS_TO_HOOK:
        if li < 0 or li >= n_layers:
            raise ValueError(
                f"Requested layer {li} is out of range [0, {n_layers - 1}]."
            )

    # ------------------------------------------------------------------
    # 2. Register hooks
    # ------------------------------------------------------------------
    print(f"[2/4] Registering forward hooks on layers {LAYERS_TO_HOOK}...")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in LAYERS_TO_HOOK:
        layer = model.encoder.layer[layer_idx]
        handles.append(layer.register_forward_hook(make_hook(captured, layer_idx)))

    # ------------------------------------------------------------------
    # 3. Read sequences
    # ------------------------------------------------------------------
    print(f"[3/4] Reading sequences from {FASTA_FILE}...")
    sequences = parse_fasta(FASTA_FILE)
    print(f"      Found {len(sequences)} sequences:")
    for name, seq in sequences:
        preview = seq[:25] + ("..." if len(seq) > 25 else "")
        print(f"        - {name:25s} len={len(seq):4d}  {preview}")

    # ------------------------------------------------------------------
    # 4. Extract
    # ------------------------------------------------------------------
    print(f"[4/4] Extracting activations...")
    all_activations: dict[str, torch.Tensor] = {}
    total_tokens = 0
    start = time.perf_counter()

    with torch.no_grad():
        for seq_idx, (name, seq) in enumerate(sequences):
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)

            # `captured` now has activations for each hooked layer.
            # Save with descriptive keys: "{name}__layer{N}".
            seq_len = inputs["input_ids"].shape[1]
            total_tokens += seq_len

            for layer_idx in LAYERS_TO_HOOK:
                # Squeeze the batch dimension (shape becomes [seq_len, hidden_dim])
                act = captured[layer_idx][0].clone()
                key = f"{name}__layer{layer_idx}"
                all_activations[key] = act

            print(
                f"      [{seq_idx + 1:2d}/{len(sequences)}] "
                f"{name:25s} → {seq_len} tokens, {len(LAYERS_TO_HOOK)} layers captured"
            )

    elapsed = time.perf_counter() - start

    # Clean up hooks (always do this, otherwise they persist across runs in the same process)
    for h in handles:
        h.remove()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_file(all_activations, str(OUTPUT_FILE))

    # Report
    n_keys = len(all_activations)
    file_size_mb = OUTPUT_FILE.stat().st_size / 1e6
    sample_key = next(iter(all_activations.keys()))
    sample_shape = tuple(all_activations[sample_key].shape)
    sample_dtype = all_activations[sample_key].dtype

    print()
    print("=" * 60)
    print("✅ Activation extraction complete!")
    print("=" * 60)
    print(f"   Total sequences:    {len(sequences)}")
    print(f"   Total tokens:       {total_tokens}")
    print(f"   Hooks per sequence: {len(LAYERS_TO_HOOK)} (layers {LAYERS_TO_HOOK})")
    print(f"   Total tensors:      {n_keys}")
    print(f"   Sample tensor:      key={sample_key!r}, shape={sample_shape}, dtype={sample_dtype}")
    print(f"   Wall time:          {elapsed:.2f} s")
    print(f"   Output file:        {OUTPUT_FILE} ({file_size_mb:.2f} MB)")

    # Quick sanity check on the first activation
    sample = all_activations[sample_key]
    if torch.isnan(sample).any():
        print("   ⚠  WARNING: NaNs detected in activations. Something is wrong.")
    elif sample.abs().mean().item() < 1e-6:
        print("   ⚠  WARNING: activations are near-zero everywhere. Something is wrong.")
    else:
        print(f"   Sanity stats:       mean={sample.mean():.4f}, "
              f"std={sample.std():.4f}, abs_max={sample.abs().max():.4f}")

    print()
    print("   Next step: Phase 9 — train a sparse autoencoder on these activations.")


if __name__ == "__main__":
    main()
