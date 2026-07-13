"""
19_extract_rnamsm_single_subset.py — RIBOSCOPE: the DECISIVE CONTROL for the
RNA-MSM "interpretability is conditional on in-distribution input" claim.

Why this exists
---------------
17/18/08/09 showed that feeding RNA-MSM its NATIVE Rfam-SEED MSA recovers 376
healthy family specialists at layer 8 (top specialist max 9.1) where the
single-sequence SAE produced none (top specialist 0.137). But the MSA run also
used a smaller, top-100-family subset (2327 sequences). A skeptic can argue the
specialists appeared because the *dataset* got easier, not because of the MSA.

This script kills that alternative. It extracts SINGLE-SEQUENCE activations for
the EXACT SAME 2327 sequences (sequences/rfam_msa_query.fasta, written by 17),
at the SAME layers (6 and 8), with the SAME [CLS, residue_1..L, EOS] token
convention. Run through the IDENTICAL downstream pipeline (18 --stem single ->
set_model rnamsm_single -> 08 -> 09), the ONLY difference vs the MSA run is the
input: single sequence (msa_depth=1, out-of-distribution) instead of native MSA.

Prediction (pre-registered): the single-seq SAE on this same subset STILL fails
(top ≤2-family specialist ~0.1), because the geometry on this identical subset
stayed collapsed in single-seq form (15: var@32 ≈ 94%) vs de-collapsed under MSA
(16: L8 var@32 67%). If single-seq fails on the same data and MSA succeeds, the
causal claim "native MSA input is what recovers specialists" is locked.

Extraction mechanics are VERBATIM from 11_extract_rnamsm.py (the validated
single-sequence extractor): tokenizer adds CLS/EOS; RNA-MSM emits a depth-1 MSA
axis so each encoder-layer capture is squeezed from [..,1,..] to [seq_len,hidden].

Run with
--------
    cd ~/projects/riboscope
    uv run python 19_extract_rnamsm_single_subset.py
    # SMOKE phase asserts the token count matches the MSA file before the full run.

Output
------
    outputs/activations_rnamsm_single_layer6.safetensors   ({name}__layer6)
    outputs/activations_rnamsm_single_layer8.safetensors   ({name}__layer8)
    outputs/activations_rnamsm_single_extract_meta.json
"""

from __future__ import annotations

import gc
import json
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
    print("   Needs: torch, multimolecule (RnaMsmModel), tqdm.")
    sys.exit(1)


# ============================ CONFIG ============================
MODEL_NAME = "multimolecule/rnamsm"
FALLBACK_MODELS = [
    "multimolecule/RNA-MSM",
    "multimolecule/RNAMsm",
    "multimolecule/rna-msm",
    "yikun-zhang/RNA-MSM",
]

# THE SAME query set the MSA run used (17 wrote it). This is what makes the
# control matched: identical sequences + identical family metadata.
FASTA_FILE = Path("sequences/rfam_msa_query.fasta")

# Same layers as 17 so the control lines up layer-for-layer.
LAYERS_TO_HOOK = [6, 8]

# Cross-check token counts against the MSA extraction (must match exactly).
MSA_REF = {li: Path(f"outputs/activations_rnamsm_msa_layer{li}.safetensors")
           for li in LAYERS_TO_HOOK}

SMOKE_N = 3      # shape/▏count-checked dry run before the full loop; 0 to skip
# ================================================================


def out_path_for(layer: int) -> Path:
    return Path(f"outputs/activations_rnamsm_single_layer{layer}.safetensors")


META_OUT = Path("outputs/activations_rnamsm_single_extract_meta.json")


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """VERBATIM from 11: name = token before first '|'; T->U. The sequences in
    rfam_msa_query.fasta are already the cleaned queries (upper, U, non-ACGU->N)."""
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
        "encoder.layer", "bert.encoder.layer", "roberta.encoder.layer",
        "rnamsm.encoder.layer", "msm.encoder.layer", "model.encoder.layer",
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


def squeeze_layer_capture(cap: torch.Tensor) -> torch.Tensor:
    """VERBATIM logic from 11: take row 0, drop the depth-1 MSA axis.
    captured[li] is [B, seq_len, msa_depth=1, hidden]; [0] -> [seq_len, 1, hidden];
    squeeze(1) -> [seq_len, hidden]."""
    act = cap[0]
    if act.ndim == 3 and act.shape[1] == 1:
        act = act.squeeze(1)
    return act


def estimate_output_bytes(d: dict) -> int:
    return sum(t.numel() * 2 for t in d.values())   # fp16 store -> 2 bytes/elem


def verify_saved_file(path: Path, expected: dict, sample_key: str) -> None:
    chk = load_file(str(path))
    if len(chk) != len(expected):
        print(f"❌ Verify failed for {path}: saved {len(expected)} but reload sees {len(chk)}.")
        sys.exit(1)
    if chk[sample_key].shape != expected[sample_key].shape:
        print(f"❌ Shape mismatch on {sample_key} in {path}.")
        sys.exit(1)
    if set(chk.keys()) != set(expected.keys()):
        print(f"❌ Key set changed during save of {path}.")
        sys.exit(1)
    print(f"      ✓ Verified {path.name}: {len(chk)} tensors, keys + shapes preserved.")


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print("RIBOSCOPE: RNA-MSM SINGLE-SEQUENCE control extraction (matched 2327-seq subset)")
    print("=" * 92)

    if not FASTA_FILE.exists():
        print(f"❌ {FASTA_FILE} not found — run 17_extract_rnamsm_msa.py first (it writes it).")
        sys.exit(1)
    for layer in LAYERS_TO_HOOK:
        op = out_path_for(layer)
        if op.exists():
            print(f"⚠ {op} already exists. Refusing to overwrite. Delete it to re-run.")
            sys.exit(1)
    out_path_for(LAYERS_TO_HOOK[0]).parent.mkdir(parents=True, exist_ok=True)

    # ---- load model + hook target layers ----
    print(f"[1/4] Loading RNA-MSM + hooking layers {LAYERS_TO_HOOK}...")
    tokenizer = model = None
    tried = []
    for candidate in [MODEL_NAME] + FALLBACK_MODELS:
        tried.append(candidate)
        try:
            tokenizer = RnaTokenizer.from_pretrained(candidate)
            model = RnaMsmModel.from_pretrained(candidate)
            print(f"      ✓ Loaded {candidate}")
            break
        except Exception as e:
            print(f"      ✗ {candidate}: {type(e).__name__}: {str(e)[:120]}")
    if model is None:
        print(f"❌ Could not load any RNA-MSM checkpoint. Tried: {tried}")
        sys.exit(1)

    model.eval().to(device)
    layers, path = find_encoder_layers(model)
    n_layers = len(layers)
    hidden_dim = getattr(model.config, "hidden_size", None)
    if max(LAYERS_TO_HOOK) >= n_layers:
        print(f"❌ Requested layer {max(LAYERS_TO_HOOK)} but model has {n_layers} layers.")
        sys.exit(1)
    print(f"      Layers at model.{path}: n_layers={n_layers}, hidden_dim={hidden_dim}")

    captured: dict[int, torch.Tensor] = {}
    handles = [layers[li].register_forward_hook(make_hook(captured, li)) for li in LAYERS_TO_HOOK]

    # ---- read the matched query set ----
    print(f"[2/4] Reading {FASTA_FILE}...")
    sequences = parse_fasta(FASTA_FILE)
    print(f"      {len(sequences)} sequences (should equal the MSA run's 'used' count).")

    # ---- SMOKE: verify token count matches the MSA file for the same key ----
    if SMOKE_N > 0:
        print(f"[3/4] SMOKE: checking {SMOKE_N} sequences (token count must match MSA file)...")
        ref_ok = MSA_REF[LAYERS_TO_HOOK[0]].exists()
        ref = load_file(str(MSA_REF[LAYERS_TO_HOOK[0]])) if ref_ok else None
        if not ref_ok:
            print(f"      (MSA reference {MSA_REF[LAYERS_TO_HOOK[0]]} not found — skipping count cross-check)")
        n_checked = 0
        with torch.no_grad():
            for name, seq in sequences:
                inputs = tokenizer(seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                captured.clear()
                _ = model(**inputs)
                T = inputs["input_ids"].shape[1]
                for li in LAYERS_TO_HOOK:
                    act = squeeze_layer_capture(captured[li])
                    assert act.ndim == 2 and int(act.shape[1]) == hidden_dim, \
                        f"L{li} unexpected shape {tuple(act.shape)}"
                    assert act.shape[0] == T, f"L{li} token count {act.shape[0]} != {T}"
                if ref is not None:
                    rk = f"{name}__layer{LAYERS_TO_HOOK[0]}"
                    if rk in ref:
                        assert ref[rk].shape[0] == T, (
                            f"token-count MISMATCH vs MSA for {name}: single={T} "
                            f"msa={ref[rk].shape[0]} — control not aligned!")
                print(f"      seq={name}  tokens={T}  (matches MSA: "
                      f"{'yes' if ref is not None and f'{name}__layer{LAYERS_TO_HOOK[0]}' in ref else 'n/a'})")
                n_checked += 1
                if n_checked >= SMOKE_N:
                    break
        del ref
        print(f"      ✓ SMOKE passed ({n_checked} seqs). Proceeding to full extraction.")

    # ---- full extraction ----
    print(f"[4/4] Extracting single-seq tokens over {len(sequences)} queries...")
    out_by_layer: dict[int, dict[str, torch.Tensor]] = {li: {} for li in LAYERS_TO_HOOK}
    n_used = n_skip = 0
    total_tokens = 0
    start = time.perf_counter()

    with torch.no_grad():
        for name, seq in tqdm(sequences, desc="single-extract", unit="seq"):
            try:
                inputs = tokenizer(seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                captured.clear()
                _ = model(**inputs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                n_skip += 1
                continue
            except Exception:
                n_skip += 1
                continue
            try:
                for li in LAYERS_TO_HOOK:
                    act = squeeze_layer_capture(captured[li]).clone()   # [L+2, D] fp16
                    out_by_layer[li][f"{name}__layer{li}"] = act
                total_tokens += int(inputs["input_ids"].shape[1])
                n_used += 1
            except Exception:
                n_skip += 1
                continue

    for h in handles:
        h.remove()
    elapsed = time.perf_counter() - start
    print(f"      Done in {elapsed:.1f}s. used={n_used}  skipped={n_skip}")
    if n_used == 0:
        print("❌ No sequences produced activations — aborting before save.")
        sys.exit(1)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- disk pre-check (both layer files) ----
    total_est = sum(estimate_output_bytes(out_by_layer[li]) for li in LAYERS_TO_HOOK)
    _, _, free = shutil.disk_usage(out_path_for(LAYERS_TO_HOOK[0]).parent)
    print(f"      Disk check: need ~{total_est / 1e9:.2f} GB, free {free / 1e9:.2f} GB")
    if free < total_est * 1.5:
        print(f"❌ Insufficient disk space (want {total_est * 1.5 / 1e9:.2f} GB headroom).")
        sys.exit(1)

    # ---- save per-layer + verify ----
    tokens_per_layer = 0
    for li in LAYERS_TO_HOOK:
        d_out = out_by_layer[li]
        op = out_path_for(li)
        print(f"      Saving {len(d_out)} tensors → {op}")
        save_file(d_out, str(op))
        verify_saved_file(op, d_out, next(iter(d_out)))
        if li == LAYERS_TO_HOOK[0]:
            tokens_per_layer = sum(int(t.shape[0]) for t in d_out.values())

    with open(META_OUT, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "purpose": "matched single-sequence control vs native-MSA run",
            "fasta": str(FASTA_FILE),
            "layers": LAYERS_TO_HOOK,
            "hidden_dim": hidden_dim,
            "n_used": n_used,
            "n_skip": n_skip,
            "n_tokens_per_layer": tokens_per_layer,
            "out_files": [str(out_path_for(li)) for li in LAYERS_TO_HOOK],
            "note": (
                "Single-sequence (msa_depth=1) activations over the SAME 2327-seq "
                "subset as the MSA run. Token set identical to the MSA file's kept "
                "rows ([CLS, residue_1..L, EOS]). Feed through 18 --stem single -> "
                "set_model rnamsm_single -> 08 -> 09. Only difference vs rnamsm_msa "
                "is the input distribution."
            ),
        }, f, indent=2)

    print()
    print("=" * 92)
    print("✅ Single-sequence control extraction complete.")
    print("=" * 92)
    for li in LAYERS_TO_HOOK:
        op = out_path_for(li)
        print(f"   L{li}: {op} ({op.stat().st_size / 1e9:.2f} GB)")
    print(f"   Tokens per layer: {tokens_per_layer:,}")
    print()
    print("   Next steps (matched control at layer 8):")
    print("     uv run python 18_prep_msa_acts.py 8 --stem single   # same z-score + position-center")
    print("     uv run python set_model.py rnamsm_single            # point 08/09 at single-seq L8 acts")
    print("     uv run python 08_train_sae_big.py                   # retrain (~2 hr)")
    print("     uv run python 09_inspect_features_big.py            # inspect (~10 min)")
    print("   Expectation: specialists STILL collapse (top ≤2-fam max ~0.1) -> MSA was the cause.")


if __name__ == "__main__":
    main()
