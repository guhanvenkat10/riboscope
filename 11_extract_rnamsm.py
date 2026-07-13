"""
11_extract_rnamsm.py — RIBOSCOPE Phase 13: Extract activations from RNA-MSM.

Why RNA-MSM (substituted for RiNALMo)
-------------------------------------
Original plan was to use RiNALMo (Penić et al. 2024) as the third model.
But the multimolecule package has the class without a corresponding HF
checkpoint — RiNALMo weights are only distributed via Zenodo from the
lbcb-sci GitHub repo, requiring custom download/loading.

Pivoting to RNA-MSM (Zhang et al. 2024) is actually a cleaner cross-model
comparison. The three-way contrast becomes:

  RNA-FM     (~100M)  structure-naive
  ErnieRNA   (~85M)   structure-aware via direct secondary-structure pretraining
  RNA-MSM    (~100M)  structure-aware via evolutionary multiple sequence alignments

Model size is held roughly constant; only the inductive bias for structural
information varies. Three flavors: none / direct / evolutionary.

Same defensive pattern as the other extractors: disk-space pre-check,
post-save verification, encoder-layer auto-detection, graceful per-sequence
error handling.

Run with
--------
    cd ~/projects/riboscope
    uv run python 11_extract_rnamsm.py

Output
------
    outputs/activations_rnamsm_layer6.safetensors
"""

import shutil
import sys
import time
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
    from multimolecule import RnaTokenizer, RnaMsmModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   If RnaMsmModel is missing, your multimolecule version is unusual.")
    print("   Confirm with:")
    print("     uv run python -c 'import multimolecule; print(dir(multimolecule))'")
    sys.exit(1)


# ============================ CONFIG ============================
LAYERS_TO_HOOK = [6]
BATCH_SIZE = 16

FASTA_FILE = Path("sequences/rfam_30k.fasta")
OUTPUT_FILE = Path("outputs/activations_rnamsm_layer6.safetensors")
MODEL_NAME = "multimolecule/rnamsm"
FALLBACK_MODELS = [
    "multimolecule/RNA-MSM",
    "multimolecule/RNAMsm",
    "multimolecule/rna-msm",
    "yikun-zhang/RNA-MSM",
]
# ================================================================


def estimate_output_bytes(activations: dict) -> int:
    return sum(t.numel() * t.element_size() for t in activations.values())


def verify_saved_file(path: Path, expected_count: int) -> None:
    loaded = load_file(str(path))
    if len(loaded) != expected_count:
        raise RuntimeError(
            f"Verification failed: saved {expected_count} tensors but reload sees {len(loaded)}"
        )


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
    candidates = [
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "rnamsm.encoder.layer",
        "msm.encoder.layer",
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
    print("❌ Could not find encoder.layer. Top-level modules:")
    for name, _ in model.named_modules():
        if name.count(".") <= 2:
            print(f"   {name}")
    raise AttributeError("encoder.layer not found")


def main() -> None:
    print("=" * 70)
    print(f"RIBOSCOPE Phase 13: RNA-MSM activation extraction (layer 6)")
    print("=" * 70)

    if not FASTA_FILE.exists():
        print(f"❌ FASTA not found: {FASTA_FILE}")
        sys.exit(1)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load with fallback chain
    print(f"[1/4] Loading RNA-MSM...")
    tokenizer = None
    model = None
    tried = []
    for candidate in [MODEL_NAME] + FALLBACK_MODELS:
        tried.append(candidate)
        try:
            print(f"      Trying {candidate}...")
            tokenizer = RnaTokenizer.from_pretrained(candidate)
            model = RnaMsmModel.from_pretrained(candidate)
            print(f"      ✓ Loaded {candidate}")
            break
        except Exception as e:
            print(f"      ✗ Failed: {type(e).__name__}: {str(e)[:200]}")

    if model is None:
        print(f"\n❌ Could not load any RNA-MSM checkpoint.")
        print(f"   Tried: {tried}")
        print(f"   Browse https://huggingface.co/multimolecule for current names.")
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
            print(f"❌ Layer {layer_idx} out of range (0–{n_layers - 1}).")
            sys.exit(1)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_idx in LAYERS_TO_HOOK:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(captured, layer_idx)))

    print(f"[3/4] Reading {FASTA_FILE}...")
    sequences = parse_fasta(FASTA_FILE)
    print(f"      Found {len(sequences)} sequences.")

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
            except torch.cuda.OutOfMemoryError:
                pbar.write(f"⚠ OOM on {name} (len {len(seq)}); skipped.")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                pbar.write(f"⚠ Skipped {name}: {e}")
                continue

            seq_len = inputs["input_ids"].shape[1]
            total_tokens += seq_len
            for layer_idx in LAYERS_TO_HOOK:
                act = captured[layer_idx][0]
                # RNA-MSM emits [seq_len, msa_depth, hidden] from each encoder
                # layer (transformer-with-MSA convention). With single-sequence
                # input msa_depth=1, so squeeze the middle dim to get the
                # [seq_len, hidden] shape the downstream SAE training expects.
                if act.ndim == 3 and act.shape[1] == 1:
                    act = act.squeeze(1)
                act = act.clone()
                key = f"{name}__layer{layer_idx}"
                all_activations[key] = act

    elapsed = time.perf_counter() - start
    for h in handles:
        h.remove()

    # Disk-space check before save
    estimated_bytes = estimate_output_bytes(all_activations)
    estimated_gb = estimated_bytes / 1e9
    total, used, free = shutil.disk_usage(OUTPUT_FILE.parent)
    free_gb = free / 1e9
    print(f"\nEstimated output size: {estimated_gb:.2f} GB")
    print(f"Available disk space:  {free_gb:.2f} GB")
    if free < estimated_bytes * 1.5:
        print(f"❌ Insufficient disk space. Need at least {estimated_gb * 1.5:.2f} GB free.")
        sys.exit(1)

    print(f"Saving {len(all_activations)} tensors to {OUTPUT_FILE}...")
    save_file(all_activations, str(OUTPUT_FILE))

    # Post-save verification
    print(f"Verifying saved file...")
    try:
        verify_saved_file(OUTPUT_FILE, expected_count=len(all_activations))
        print(f"   ✓ Verified: {len(all_activations)} tensors load cleanly.")
    except Exception as e:
        print(f"❌ Save verification failed: {e}")
        sys.exit(1)

    file_size_gb = OUTPUT_FILE.stat().st_size / 1e9
    print()
    print("=" * 70)
    print("✅ RNA-MSM extraction complete!")
    print("=" * 70)
    print(f"   Sequences processed:  {len(sequences)}")
    print(f"   Total tokens:         {total_tokens:,}")
    print(f"   Tensors saved:        {len(all_activations)}")
    print(f"   Wall time:            {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"   Output file:          {OUTPUT_FILE} ({file_size_gb:.2f} GB)")
    print(f"   Hidden dim:           {hidden_dim}")
    print()
    print("   Next steps:")
    print("     uv run python set_model.py rnamsm")
    print("     uv run python 08_train_sae_big.py")
    print("     uv run python 09_inspect_features_big.py")


if __name__ == "__main__":
    main()
