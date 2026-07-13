"""
10_extract_omnigenome.py — RIBOSCOPE Phase 13: Extract activations from OmniGenome.

Update (v2): bypasses multimolecule entirely for OmniGenome (since the
multimolecule library doesn't ship the OmniGenome class in many versions).
Loads OmniGenome from its original HuggingFace repo using transformers'
AutoModel/AutoTokenizer with trust_remote_code=True.

OmniGenome (Yang et al. 2024) is a structure-contextualised RNA FM that
explicitly models RNA secondary structure during training. We expect its
features to differ from RNA-FM's (which is structure-naive).

Run with
--------
    cd ~/projects/riboscope
    uv run python 10_extract_omnigenome.py

If the model name in MODEL_NAME (below) 404s, try one of the candidate
names listed in the FALLBACK_MODELS list — the script will try them in
order.
"""

import sys
import time
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Should be installed already. Try:")
    print("     uv pip install transformers safetensors tqdm")
    sys.exit(1)


# ============================ CONFIG ============================
LAYERS_TO_HOOK = [6]
BATCH_SIZE = 16

FASTA_FILE = Path("sequences/rfam_30k.fasta")
OUTPUT_FILE = Path("outputs/activations_omnigenome_layer6.safetensors")

# Primary OmniGenome checkpoint to try.
MODEL_NAME = "yangheng/OmniGenome-186M"

# Fallback candidates if the primary 404s
FALLBACK_MODELS = [
    "yangheng/OmniGenome-52M",
    "yangheng/OmniGenome-186M-v2",
    "multimolecule/omnigenome",
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
    """
    Find the encoder layers regardless of how the model wraps them.
    Different HF model classes put encoder.layer at different attribute paths.
    """
    candidates = [
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "transformer.encoder.layer",
        "transformer.h",
        "transformer.layer",
        "model.encoder.layer",
        "esm.encoder.layer",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            # We found the layer container — check it acts like a sequence
            _ = len(obj)
            return obj, path
        except (AttributeError, TypeError):
            continue
    # Last resort: print the module tree to help debug
    print("❌ Could not auto-detect encoder.layer path. Model top-level modules:")
    for name, _ in model.named_modules():
        if name.count(".") <= 2:
            print(f"   {name}")
    raise AttributeError("encoder.layer not found in model")


def try_load_model(name: str):
    """Try to load a HuggingFace model by name. Returns (tokenizer, model) or (None, None)."""
    try:
        print(f"      Trying {name}...")
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
        print(f"      ✓ Loaded {name}")
        return tokenizer, model
    except Exception as e:
        print(f"      ✗ Failed: {type(e).__name__}: {str(e)[:200]}")
        return None, None


def main() -> None:
    print("=" * 70)
    print(f"RIBOSCOPE Phase 13: OmniGenome activation extraction (layer 6)")
    print("=" * 70)

    if not FASTA_FILE.exists():
        print(f"❌ FASTA not found: {FASTA_FILE}.  Run Phase 12 step 2 first.")
        sys.exit(1)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load model (with fallback chain)
    print(f"[1/4] Loading OmniGenome from HuggingFace...")
    print(f"      (first run downloads ~750MB; cached afterward)")
    tokenizer, model = try_load_model(MODEL_NAME)
    if model is None:
        print(f"      Trying fallbacks...")
        for fb in FALLBACK_MODELS:
            tokenizer, model = try_load_model(fb)
            if model is not None:
                break
    if model is None:
        print(f"\n❌ Could not load any OmniGenome checkpoint.")
        print(f"   Tried: {[MODEL_NAME] + FALLBACK_MODELS}")
        print(f"   Browse https://huggingface.co/yangheng for current names,")
        print(f"   then edit MODEL_NAME at the top of this script.")
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

    print(f"\nSaving {len(all_activations)} tensors to {OUTPUT_FILE}...")
    save_file(all_activations, str(OUTPUT_FILE))

    file_size_gb = OUTPUT_FILE.stat().st_size / 1e9
    print()
    print("=" * 70)
    print("✅ OmniGenome extraction complete!")
    print("=" * 70)
    print(f"   Sequences processed:  {len(sequences)}")
    print(f"   Total tokens:         {total_tokens:,}")
    print(f"   Tensors saved:        {len(all_activations)}")
    print(f"   Wall time:            {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"   Output file:          {OUTPUT_FILE} ({file_size_gb:.2f} GB)")
    print(f"   Hidden dim:           {hidden_dim}  (will be d_input for OmniGenome SAE)")
    print()
    print("   Next: train SAE on OmniGenome by editing 08_train_sae_big.py:")
    print(f'     ACT_FILE = Path("{OUTPUT_FILE}")')
    print(f'     SAE_FINAL = Path("outputs/sae_omnigenome_layer6_v1.safetensors")')
    print(f'     HISTORY_OUT = Path("outputs/sae_omnigenome_layer6_v1_history.json")')


if __name__ == "__main__":
    main()
