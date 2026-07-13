"""
29_fix_validate_ernierna.py — RIBOSCOPE: repair + validate the ErnieRNA load.

After the (blind) `uv pip install -U multimolecule` (now 0.2.0), ErnieRnaModel
loads but its structure-aware `pairwise_bias_proj` weights DON'T load — the
checkpoint names them `model.pairwise_bias_proj.dense1/dense2` while the current
code expects the nn.Sequential `pairwise_bias_proj.0/2`. So that component was
randomly initialized and ErnieRNA's features were untrustworthy.

This script:
  1. Loads ErnieRNA the working way (pad tokenizer to config vocab so the
     len(tokenizer)==vocab_size guard passes).
  2. REMAPS the pairwise_bias_proj weights from the checkpoint into the model.
  3. VALIDATES the fixed model with the Rfam positive-control self-test: do the
     locked ErnieRNA SAE specialists fire ~their known max on their OWN Rfam
     family members? If yes, the load now matches the SAE's training
     distribution and ErnieRNA is back online for the snoRNA cross-model work.

Run with
--------
    cd ~/projects/riboscope
    uv run python 29_fix_validate_ernierna.py

Pass criterion: >=60% of tested specialist families fire >=0.5x their max
(same bar RNA-FM cleared). If it fails, the remap assumption is wrong and we
fall back to pinning a multimolecule version that loads ErnieRNA cleanly.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
    from multimolecule import RnaTokenizer, ErnieRnaModel
    from transformers import AutoConfig
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE

MODEL_NAME = "multimolecule/ernierna"
LAYER = 6
SAE_FILE = Path("outputs/sae_erniarna_layer6_v3.safetensors")
INSPECTION = Path("outputs/inspection_erniarna_layer6_v3.json")
RFAM_FASTA = Path("sequences/rfam_30k.fasta")
N_FAMILIES = 20


def find_encoder_layers(model):
    for p in ("encoder.layer", "bert.encoder.layer", "roberta.encoder.layer",
              "ernie.encoder.layer", "model.encoder.layer"):
        obj = model
        try:
            for part in p.split("."):
                obj = getattr(obj, part)
            _ = len(obj)
            return obj
        except (AttributeError, TypeError):
            continue
    raise AttributeError("encoder.layer not found")


def make_hook(captured, layer_idx):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured[layer_idx] = output.detach().to(torch.float16).cpu()
    return hook


def parse_rfam_fasta(path):
    fams = defaultdict(list)
    name = fam = None
    buf = []

    def flush():
        if name is not None and fam:
            fams[fam].append((name, "".join(buf).upper().replace("T", "U")))

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                parts = [p.strip() for p in line[1:].split("|")]
                name = parts[0] if parts else None
                fam = parts[1] if len(parts) > 1 else None
                buf = []
            else:
                buf.append(line)
        flush()
    return fams


def load_ernierna_fixed(device):
    """Load ErnieRNA and repair the pairwise_bias_proj weights from the checkpoint."""
    tok = RnaTokenizer.from_pretrained(MODEL_NAME)
    conf = AutoConfig.from_pretrained(MODEL_NAME)
    want = int(getattr(conf, "vocab_size", len(tok)))
    if want > len(tok):
        tok.add_tokens([f"<unused{i}>" for i in range(want - len(tok))], special_tokens=True)
    model = ErnieRnaModel.from_pretrained(MODEL_NAME, tokenizer=tok)

    # locate checkpoint
    ckpt = None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(MODEL_NAME, fname)
            ckpt = load_file(path) if fname.endswith(".safetensors") else torch.load(path, map_location="cpu")
            break
        except Exception:  # noqa: BLE001
            continue
    if ckpt is None:
        print("⚠ Could not fetch checkpoint to remap pairwise weights.")
        return tok, model.eval().to(device), 0

    sd = model.state_dict()
    to_load = {}
    for k, v in ckpt.items():
        kk = k[6:] if k.startswith("model.") else k
        kk = kk.replace("pairwise_bias_proj.dense1", "pairwise_bias_proj.0")
        kk = kk.replace("pairwise_bias_proj.dense2", "pairwise_bias_proj.2")
        if "pairwise_bias_proj" in kk and kk in sd and tuple(sd[kk].shape) == tuple(v.shape):
            to_load[kk] = v
    if to_load:
        model.load_state_dict(to_load, strict=False)
    print(f"   remapped {len(to_load)} pairwise_bias_proj tensors: {sorted(to_load)}")
    return tok, model.eval().to(device), len(to_load)


def main():
    print("=" * 74)
    print("RIBOSCOPE: fix + validate ErnieRNA load")
    print("=" * 74)
    for p in (SAE_FILE, INSPECTION, RFAM_FASTA):
        if not Path(p).exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[1/3] Loading + repairing ErnieRNA ...")
    tok, model, n_remap = load_ernierna_fixed(device)
    if n_remap == 0:
        print("⚠ No pairwise weights were remapped — key names differ from expectation.")
        print("  Validation will likely fail; if so we pin a multimolecule version instead.")
    layers = find_encoder_layers(model)
    captured = {}
    handle = layers[LAYER].register_forward_hook(make_hook(captured, LAYER))

    print("[2/3] Loading ErnieRNA SAE + specialist list ...")
    sae_state = load_file(str(SAE_FILE))
    d_input, d_dict = sae_state["W_enc"].shape[0], sae_state["W_enc"].shape[1]
    sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=32)
    sae.load_state_dict(sae_state)
    sae = sae.eval().to(device)

    insp = json.loads(Path(INSPECTION).read_text())
    spec = [r for r in insp.get("specialist_features", []) if float(r.get("max_activation", 0)) > 0]
    fam_to_feat = {}
    for r in spec:
        for fam in r.get("families_sample", []):
            if fam not in fam_to_feat or r["max_activation"] > fam_to_feat[fam]["max_activation"]:
                fam_to_feat[fam] = r
    fams = parse_rfam_fasta(RFAM_FASTA)

    print("[3/3] Self-test: ErnieRNA specialists vs their own Rfam family members\n")
    print(f"  {'family':<10}{'feat':>6}{'feat_max':>9}{'best':>9}{'frac':>6}  verdict")
    print("  " + "-" * 52)
    tested = passed = 0
    with torch.no_grad():
        for fam, r in fam_to_feat.items():
            if fam not in fams:
                continue
            best = 0.0
            for _, seq in fams[fam]:
                if not seq:
                    continue
                inputs = tok(seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
                act = captured[LAYER][0].float().to(device)
                ntok = act.shape[0]
                if ntok <= 2:
                    continue
                col = sae.encode(act)[1:ntok - 1, r["feature_idx"]]
                best = max(best, float(col.max().item()))
            frac = best / r["max_activation"]
            ok = frac >= 0.5
            tested += 1
            passed += int(ok)
            print(f"  {fam:<10}{r['feature_idx']:>6}{r['max_activation']:>9.3f}{best:>9.3f}{frac:>6.2f}  {'OK' if ok else 'WEAK'}")
            if tested >= N_FAMILIES:
                break
    handle.remove()

    rate = passed / tested if tested else 0.0
    print("\n--- VERDICT ---")
    print(f"  {passed}/{tested} specialist families fired >=0.5x max  ({rate:.0%})")
    if rate >= 0.6:
        print("  ✓ ErnieRNA LOAD FIXED — matches the SAE's training distribution.")
        print("    Next: re-extract snoDB ErnieRNA features and re-run the functional axis.")
    else:
        print("  ✗ Still broken — the pairwise remap didn't restore correct features.")
        print("    Fallback: pin a multimolecule version that loads ErnieRNA cleanly")
        print("    (report `uv pip show multimolecule` and we'll choose one).")


if __name__ == "__main__":
    main()
