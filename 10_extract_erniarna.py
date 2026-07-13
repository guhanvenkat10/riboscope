"""
10_extract_erniarna.py — RIBOSCOPE Phase 13: Extract activations from ErnieRNA.

Why ErnieRNA (not OmniGenome)
-----------------------------
Originally planned: OmniGenome (Yang et al. 2024). But on a 2026-vintage
transformers + multimolecule install, both OmniGenome checkpoints fail:
  - yangheng/OmniGenome-186M requires the ViennaRNA library as a hard dep
  - yangheng/OmniGenome-52M uses a removed transformers internal API

ErnieRNA (Yin et al. 2024) is in the same scientific category — a
structure-aware RNA foundation model that incorporates RNA secondary
structure during pretraining — and it ships cleanly via the multimolecule
package, with no extra dependencies and no API compatibility issues.

The "second RNA FM" role is to provide a structure-aware contrast against
RNA-FM (which is structure-naive). ErnieRNA fills that role at least as
well as OmniGenome would.

Run with
--------
    cd ~/projects/riboscope
    uv run python 10_extract_erniarna.py

Output
------
    outputs/activations_erniarna_layer6.safetensors
"""

import shutil
import sys
import time
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
    from multimolecule import RnaTokenizer, ErnieRnaModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   If ErnieRnaModel is missing, your multimolecule version is unusual.")
    print("   Run the diagnostic from earlier to confirm what's available.")
    sys.exit(1)


def estimate_output_bytes(activations: dict) -> int:
    """Estimate raw byte size of the saved safetensors file (metadata is small overhead)."""
    return sum(t.numel() * t.element_size() for t in activations.values())


def verify_saved_file(path: Path, expected_count: int) -> None:
    """Reload the file and confirm tensor count matches. Raise on failure."""
    loaded = load_file(str(path))
    if len(loaded) != expected_count:
        raise RuntimeError(
            f"Verification failed: saved {expected_count} tensors but reload sees {len(loaded)}"
        )


# ============================ CONFIG ============================
LAYERS_TO_HOOK = [6]
BATCH_SIZE = 16

FASTA_FILE = Path("sequences/rfam_30k.fasta")
OUTPUT_FILE = Path("outputs/activations_erniarna_layer6.safetensors")
MODEL_NAME = "multimolecule/ernierna"
# ================================================================


def parse_fasta(path: Path) -> list[tuple[str, str]]:
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


def find_encoder_layers(model):
    """
    Find the encoder layers regardless of model wrapping.
    multimolecule models typically use model.encoder.layer, but we try several.
    """
    candidates = [
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "ernie.encoder.layer",
        "model.encoder.layer",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            _ = len(obj)
            return obj, path
        except (AttributeError, TypeError):
            continue
    print("❌ Could not find encoder.layer. Model top-level modules:")
    for name, _ in model.named_modules():
        if name.count(".") <= 2:
            print(f"   {name}")
    raise AttributeError("encoder.layer not found")


def main() -> None:
    print("=" * 70)
    print(f"RIBOSCOPE Phase 13: ErnieRNA activation extraction (layer 6)")
    print("=" * 70)

    if not FASTA_FILE.exists():
        print(f"❌ FASTA not found: {FASTA_FILE}.  Run Phase 12 step 2 first.")
        sys.exit(1)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {MODEL_NAME}...")
    print(f"      (first run downloads model; cached afterward)")
    try:
        tokenizer = RnaTokenizer.from_pretrained(MODEL_NAME)
        model = ErnieRnaModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"❌ Failed to load: {type(e).__name__}: {e}")
        print(f"   If the repo name {MODEL_NAME!r} is wrong, browse")
        print(f"   https://huggingface.co/multimolecule and update MODEL_NAME.")
        sys.exit(1)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"      Device: {device}")

    # Find encoder layers
    print(f"[2/4] Locating encoder layers + hooking layer(s) {LAYERS_TO_HOOK}...")
    layers, path = find_encoder_layers(model)
    n_layers = len(layers)
    hidden_dim = getattr(model.config, "hidden_size", None)
    print(f"      Layers found at: model.{path}")
    print(f"      n_layers={n_layers}, hidden_dim={hidden_dim}")

    for layer_idx in LAYERS_TO_HOOK:
        if layer_idx >= n_layers:
            print(f"❌ Requested layer {layer_idx} is out of range (0–{n_layers - 1}).")
            sys.exit(1)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in LAYERS_TO_HOOK:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(captured, layer_idx)))

    # Read sequences
    print(f"[3/4] Reading {FASTA_FILE}...")
    sequences = parse_fasta(FASTA_FILE)
    print(f"      Found {len(sequences)} sequences.")

    # Extract
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
                pbar.write(f"⚠ Skipped {name}: {e}")
                continue

            seq_len = inputs["input_ids"].shape[1]
            total_tokens += seq_len
            for layer_idx in LAYERS_TO_HOOK:
                act = captured[layer_idx][0].clone()
                key = f"{name}__layer{layer_idx}"
                all_activations[key] = act

    elapsed = time.perf_counter() - start
    for h in handles:
        h.remove()

    # Pre-save: check disk space
    estimated_bytes = estimate_output_bytes(all_activations)
    estimated_gb = estimated_bytes / 1e9
    total, used, free = shutil.disk_usage(OUTPUT_FILE.parent)
    free_gb = free / 1e9
    print(f"\nEstimated output size: {estimated_gb:.2f} GB")
    print(f"Available disk space:  {free_gb:.2f} GB")
    if free < estimated_bytes * 1.5:
        print(f"❌ Insufficient disk space. Need at least {estimated_gb * 1.5:.2f} GB free "
              f"({free_gb:.2f} GB available). Free up space and re-run.")
        sys.exit(1)

    print(f"Saving {len(all_activations)} tensors to {OUTPUT_FILE}...")
    save_file(all_activations, str(OUTPUT_FILE))

    # Post-save: verify the file is readable and complete
    print(f"Verifying saved file...")
    try:
        verify_saved_file(OUTPUT_FILE, expected_count=len(all_activations))
        print(f"   ✓ Verified: {len(all_activations)} tensors load cleanly.")
    except Exception as e:
        print(f"❌ Save verification failed: {e}")
        print(f"   The file at {OUTPUT_FILE} is corrupt or incomplete.")
        print(f"   Delete it and re-run. Likely cause: disk space or WSL filesystem hiccup.")
        sys.exit(1)

    file_size_gb = OUTPUT_FILE.stat().st_size / 1e9
    print()
    print("=" * 70)
    print("✅ ErnieRNA extraction complete!")
    print("=" * 70)
    print(f"   Sequences processed:  {len(sequences)}")
    print(f"   Total tokens:         {total_tokens:,}")
    print(f"   Tensors saved:        {len(all_activations)}")
    print(f"   Wall time:            {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"   Output file:          {OUTPUT_FILE} ({file_size_gb:.2f} GB)")
    print(f"   Hidden dim:           {hidden_dim}  (will be d_input for ErnieRNA SAE)")
    print()
    print("   Next: train SAE on ErnieRNA by editing 08_train_sae_big.py:")
    print(f'     ACT_FILE = Path("{OUTPUT_FILE}")')
    print(f'     SAE_FINAL = Path("outputs/sae_erniarna_layer6_v1.safetensors")')
    print(f'     HISTORY_OUT = Path("outputs/sae_erniarna_layer6_v1_history.json")')
    print(f'     CHECKPOINT_DIR = Path("outputs/checkpoints_erniarna")')


if __name__ == "__main__":
    main()
