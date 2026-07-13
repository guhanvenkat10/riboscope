"""
11_extract_rinalmo.py — RIBOSCOPE Phase 13: Extract activations from RiNALMo.

Uses multimolecule's RiNALMoModel directly (confirmed available in your
multimolecule install via the diagnostic). RiNALMo (Penić et al. 2024)
is the largest RNA foundation model at 650M parameters.

Memory considerations
---------------------
RiNALMo at 650M params is ~2.6GB FP16, plus per-batch activations. On a
16GB 5060 Ti, the model is loaded in FP16 to keep VRAM usage under control.
If you hit CUDA OOM, reduce BATCH_SIZE from 8 → 4 → 2.

Run with
--------
    cd ~/projects/riboscope
    uv run python 11_extract_rinalmo.py
"""

import shutil
import sys
import time
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
    from multimolecule import RnaTokenizer, RiNALMoModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Your multimolecule should have RiNALMoModel — confirmed by")
    print("   the diagnostic. If not, re-check with:")
    print("     uv run python -c 'import multimolecule; print(dir(multimolecule))'")
    sys.exit(1)


def estimate_output_bytes(activations: dict) -> int:
    return sum(t.numel() * t.element_size() for t in activations.values())


def verify_saved_file(path: Path, expected_count: int) -> None:
    loaded = load_file(str(path))
    if len(loaded) != expected_count:
        raise RuntimeError(
            f"Verification failed: saved {expected_count} tensors but reload sees {len(loaded)}"
        )


# ============================ CONFIG ============================
LAYERS_TO_HOOK = [6]
BATCH_SIZE = 8

FASTA_FILE = Path("sequences/rfam_30k.fasta")
OUTPUT_FILE = Path("outputs/activations_rinalmo_layer6.safetensors")
MODEL_NAME = "multimolecule/rinalmo"
# Fallback model identifiers tried in order if MODEL_NAME 404s
FALLBACK_MODELS = [
    "multimolecule/rinalmo-650m",
    "multimolecule/rinalmo-150m",
    "multimolecule/rinalmo-33m",
    "multimolecule/RiNALMo",
    "lbcb-sci/rinalmo",
    "lbcb-sci/RiNALMo",
]
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
    candidates = [
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "rinalmo.encoder.layer",
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
    print(f"RIBOSCOPE Phase 13: RiNALMo activation extraction (layer 6)")
    print("=" * 70)

    if not FASTA_FILE.exists():
        print(f"❌ FASTA not found: {FASTA_FILE}")
        sys.exit(1)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading RiNALMo...")
    print(f"      (first run downloads ~2.6GB; cached afterward)")
    tokenizer = None
    model = None
    tried = []
    for candidate in [MODEL_NAME] + FALLBACK_MODELS:
        tried.append(candidate)
        try:
            print(f"      Trying {candidate}...")
            tokenizer = RnaTokenizer.from_pretrained(candidate)
            model = RiNALMoModel.from_pretrained(candidate)
            print(f"      ✓ Loaded {candidate}")
            break
        except Exception as e:
            print(f"      ✗ Failed: {type(e).__name__}: {str(e)[:200]}")

    if model is None:
        print(f"\n❌ Could not load any RiNALMo checkpoint.")
        print(f"   Tried: {tried}")
        print(f"   Browse https://huggingface.co/multimolecule for current names")
        print(f"   and edit MODEL_NAME at the top of this script.")
        sys.exit(1)

    model.eval()
    # FP16 to fit in 16 GB VRAM
    model = model.half()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"      Device: {device}, dtype={next(model.parameters()).dtype}")

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
                pbar.write(f"⚠ OOM on {name} (len {len(seq)}); skipped. Reduce BATCH_SIZE if frequent.")
                torch.cuda.empty_cache()
                continue
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

    # Post-save: verify
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
    print("✅ RiNALMo extraction complete!")
    print("=" * 70)
    print(f"   Sequences processed:  {len(sequences)}")
    print(f"   Total tokens:         {total_tokens:,}")
    print(f"   Tensors saved:        {len(all_activations)}")
    print(f"   Wall time:            {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"   Output file:          {OUTPUT_FILE} ({file_size_gb:.2f} GB)")
    print(f"   Hidden dim:           {hidden_dim}")
    print()
    print("   Next: train SAE on RiNALMo by editing 08_train_sae_big.py:")
    print(f'     ACT_FILE = Path("{OUTPUT_FILE}")')
    print(f'     SAE_FINAL = Path("outputs/sae_rinalmo_layer6_v1.safetensors")')
    print(f'     HISTORY_OUT = Path("outputs/sae_rinalmo_layer6_v1_history.json")')


if __name__ == "__main__":
    main()
