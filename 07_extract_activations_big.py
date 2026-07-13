"""
07_extract_activations_big.py — RIBOSCOPE Phase 11: Scaled activation extraction.

Same logic as Phase 8's 03_extract_activations.py, but adapted for ~10k
sequences instead of 12. Specifically:
  - Reads from a (configurable) FASTA file
  - Hooks ONLY layer 6 by default (layer 1 + 11 doubles disk usage and
    we don't need them for the first scaled run)
  - Adds a tqdm progress bar so you can see how the overnight run is going
  - Saves to a single safetensors file at outputs/activations_big_layer6.safetensors

Estimated runtime on RTX 5060 Ti: ~30–90 minutes for 10k sequences.
Estimated output size: ~2–4 GB depending on average sequence length.

If the 5060 Ti runs out of VRAM (unlikely at sequence-length 510), reduce
the BATCH_SIZE config below from 16 to 8 or 4.

Run with
--------
    cd ~/projects/riboscope
    uv run python 07_extract_activations_big.py
"""

import sys
import time
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
    from multimolecule import RnaTokenizer, RnaFmModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


# ============================ CONFIG ============================
LAYERS_TO_HOOK = [6]   # which layers to extract (just layer 6 for first big run)
BATCH_SIZE = 16        # number of sequences processed per forward pass

FASTA_FILE = Path("sequences/rfam_30k.fasta")  # Phase 12: was rfam_10k.fasta
OUTPUT_FILE = Path("outputs/activations_big_layer6_v2.safetensors")  # Phase 12: was _v1
MODEL_NAME = "multimolecule/rnafm"
# ================================================================


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Same parser as Phase 8."""
    sequences: list[tuple[str, str]] = []
    name: str | None = None
    seq: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    sequences.append((name, "".join(seq)))
                name = line[1:].split("|")[0].strip()
                seq = []
            else:
                seq.append(line.replace("T", "U").replace("t", "u"))
        if name is not None:
            sequences.append((name, "".join(seq)))
    return sequences


def make_hook(captured: dict, layer_idx: int):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured[layer_idx] = output.detach().to(torch.float16).cpu()
    return hook


def main() -> None:
    print("=" * 70)
    print("RIBOSCOPE Phase 11: Scaled Activation Extraction (RNA-FM layer 6)")
    print("=" * 70)

    if not FASTA_FILE.exists():
        print(f"❌ FASTA file not found: {FASTA_FILE}")
        print(f"   Run Phase 11 step 1 first: uv run python 06_fetch_rfam_sequences.py")
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- #
    # 1. Load model
    # --------------------------------------------------------------- #
    print(f"[1/4] Loading {MODEL_NAME}...")
    tokenizer = RnaTokenizer.from_pretrained(MODEL_NAME)
    model = RnaFmModel.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"      Device: {device}")

    n_layers = len(model.encoder.layer)
    hidden_dim = model.config.hidden_size
    print(f"      Layers: {n_layers}, hidden_dim: {hidden_dim}")

    # --------------------------------------------------------------- #
    # 2. Hook layers
    # --------------------------------------------------------------- #
    print(f"[2/4] Registering hooks on layers {LAYERS_TO_HOOK}...")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in LAYERS_TO_HOOK:
        layer = model.encoder.layer[layer_idx]
        handles.append(layer.register_forward_hook(make_hook(captured, layer_idx)))

    # --------------------------------------------------------------- #
    # 3. Read sequences
    # --------------------------------------------------------------- #
    print(f"[3/4] Reading sequences from {FASTA_FILE}...")
    sequences = parse_fasta(FASTA_FILE)
    print(f"      Found {len(sequences)} sequences.")

    # --------------------------------------------------------------- #
    # 4. Extract (with progress bar)
    # --------------------------------------------------------------- #
    print(f"[4/4] Extracting layer-{LAYERS_TO_HOOK[0]} activations...")
    all_activations: dict[str, torch.Tensor] = {}
    total_tokens = 0
    start = time.perf_counter()

    pbar = tqdm(sequences, desc="extracting", unit="seq")
    with torch.no_grad():
        for name, seq in pbar:
            try:
                inputs = tokenizer(seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
            except Exception as e:
                # Skip sequences that fail (rare, usually too-long edge cases)
                pbar.write(f"⚠ Skipped {name}: {e}")
                continue

            seq_len = inputs["input_ids"].shape[1]
            total_tokens += seq_len

            for layer_idx in LAYERS_TO_HOOK:
                # Squeeze batch dim → [seq_len, hidden_dim], in fp16
                act = captured[layer_idx][0].clone()
                key = f"{name}__layer{layer_idx}"
                all_activations[key] = act

    elapsed = time.perf_counter() - start

    # Clean up hooks
    for h in handles:
        h.remove()

    # --------------------------------------------------------------- #
    # Save
    # --------------------------------------------------------------- #
    print(f"\nSaving {len(all_activations)} tensors to {OUTPUT_FILE}...")
    save_file(all_activations, str(OUTPUT_FILE))

    # Report
    n_keys = len(all_activations)
    file_size_gb = OUTPUT_FILE.stat().st_size / 1e9

    print()
    print("=" * 70)
    print("✅ Scaled extraction complete!")
    print("=" * 70)
    print(f"   Sequences processed:  {len(sequences)}")
    print(f"   Total tokens:         {total_tokens:,}")
    print(f"   Tensors saved:        {n_keys}")
    print(f"   Wall time:            {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"   Output file:          {OUTPUT_FILE} ({file_size_gb:.2f} GB)")
    print()
    print("   Next step: Phase 11 — train a production SAE on these activations:")
    print("     uv run python 08_train_sae_big.py")


if __name__ == "__main__":
    main()
