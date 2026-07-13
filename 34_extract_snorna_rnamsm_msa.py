"""
34_extract_snorna_rnamsm_msa.py — RIBOSCOPE step R2: RNA-MSM activations for the
human snoRNAs, under native-MSA input (built by 33_build_snorna_msa.py).

For each snoRNA we feed RNA-MSM the MSA [query row + SEED context rows] (all in
the family's SEED columns), exactly the way the headline MSA run worked, and keep
the query row's CLS + EOS + non-gap columns at layer 8. With MAFFT --keeplength
the query may have had insertions dropped, so we keep WHATEVER non-gap query
columns exist (we don't assert == ungapped length — that's expected here).

Output raw fp16 acts keyed {snodb_id}__layer8, for step R3 (preprocess with the
TRAINING normalization) → R4 (SAE encode).

Run with
--------
    cd ~/projects/riboscope
    uv run python 34_extract_snorna_rnamsm_msa.py --pilot   # only families already built
    uv run python 34_extract_snorna_rnamsm_msa.py

Inputs : outputs/snorna_msa/<rfam>.fasta  (from 33)
Output : outputs/activations_rnamsm_msa_snorna_layer8.safetensors  ({snodb_id}__layer8)
         outputs/snorna_rnamsm_manifest.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file, load_file
    from multimolecule import RnaTokenizer, RnaMsmModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

MSA_DIR = Path("outputs/snorna_msa")
LAYER = 8
OUT = Path("outputs/activations_rnamsm_msa_snorna_layer8.safetensors")
MANIFEST = Path("outputs/snorna_rnamsm_manifest.json")
MODEL_NAME = "multimolecule/rnamsm"
FALLBACKS = ["multimolecule/RNA-MSM", "multimolecule/RNAMsm", "multimolecule/rna-msm", "yikun-zhang/RNA-MSM"]


def parse_fasta(path: Path):
    recs = []
    name = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            recs.append((name, "".join(buf)))
    return recs


def find_encoder_layers(model):
    for p in ("encoder.layer", "bert.encoder.layer", "roberta.encoder.layer",
              "rnamsm.encoder.layer", "msm.encoder.layer", "model.encoder.layer"):
        obj = model
        try:
            for part in p.split("."):
                obj = getattr(obj, part)
            _ = len(obj)
            return obj
        except (AttributeError, TypeError):
            continue
    raise AttributeError("encoder.layer not found")


def make_hook(captured, li):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured[li] = output.detach().to(torch.float16).cpu()
    return hook


def msa_to_ids(tokenizer, rows, device):
    enc = [tokenizer(r, return_tensors="pt")["input_ids"][0] for r in rows]
    if len({t.shape[0] for t in enc}) != 1:
        raise ValueError("ragged token lengths within an MSA")
    ids = torch.stack(enc, dim=0).unsqueeze(0).to(device)   # [1, R, C]
    return ids, torch.ones_like(ids)


def residue_col_mask(query_aln: str, n_tokens: int):
    m = torch.zeros(n_tokens, dtype=torch.bool)
    m[0] = True
    m[n_tokens - 1] = True
    for col in range(1, n_tokens - 1):
        if query_aln[col - 1] != ".":
            m[col] = True
    return m


def main():
    pilot = "--pilot" in sys.argv
    if OUT.exists():
        print(f"⚠ {OUT} exists; refusing to overwrite. Delete to re-run.")
        sys.exit(1)
    files = sorted(MSA_DIR.glob("*.fasta"))
    if not files:
        print(f"❌ No MSA files in {MSA_DIR} (run 33_build_snorna_msa.py first).")
        sys.exit(1)
    if pilot:
        files = files[:3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print(f"RIBOSCOPE R2: RNA-MSM MSA activations for snoRNAs (layer {LAYER})")
    print("=" * 72)
    print(f"Family MSA files: {len(files)}")

    tokenizer = model = None
    for cand in [MODEL_NAME] + FALLBACKS:
        try:
            tokenizer = RnaTokenizer.from_pretrained(cand)
            model = RnaMsmModel.from_pretrained(cand)
            print(f"  loaded {cand}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {cand}: {type(e).__name__}")
    if model is None:
        print("❌ could not load RNA-MSM.")
        sys.exit(1)
    model.eval().to(device)
    if not (hasattr(model, "embeddings") and hasattr(model, "encoder")):
        print("❌ model lacks .embeddings/.encoder for MSA bypass.")
        sys.exit(1)
    layers = find_encoder_layers(model)
    cfg_maxpos = getattr(model.config, "max_position_embeddings", 1024)
    captured = {}
    handle = layers[LAYER].register_forward_hook(make_hook(captured, LAYER))

    out = {}
    n_used = n_skip = 0
    start = time.perf_counter()
    with torch.no_grad():
        for fpath in files:
            recs = parse_fasta(fpath)
            seed_rows = [s for n, s in recs if n.startswith("SEED_")]
            queries = [(n, s) for n, s in recs if not n.startswith("SEED_")]
            if not seed_rows:
                continue
            for qname, qrow in tqdm(queries, desc=fpath.stem, unit="sno", leave=False):
                rows = [qrow] + seed_rows
                W = len(qrow)
                if W + 2 > cfg_maxpos:
                    n_skip += 1
                    continue
                try:
                    ids, mask = msa_to_ids(tokenizer, rows, device)
                    captured.clear()
                    emb = model.embeddings(input_ids=ids, attention_mask=mask)
                    _ = model.encoder(emb, attention_mask=mask)
                    cap = captured[LAYER]              # [R, C, 1, D] or [1,R,C,D]?
                    # 17 convention: cap[0].squeeze(1) -> [C, D] for row 0
                    qtok = cap[0].squeeze(1) if cap.ndim == 4 else cap[0]
                    C = qtok.shape[0]
                    m = residue_col_mask(qrow, C)
                    snodb_id = qname.split("|")[0]
                    out[f"{snodb_id}__layer{LAYER}"] = qtok[m].clone()   # [k, D] fp16
                    n_used += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    n_skip += 1
                except Exception:
                    n_skip += 1
    handle.remove()
    elapsed = time.perf_counter() - start
    print(f"\nExtracted {n_used} snoRNAs (skipped {n_skip}) in {elapsed:.0f}s.")
    if n_used == 0:
        print("❌ nothing extracted.")
        sys.exit(1)

    save_file(out, str(OUT))
    chk = load_file(str(OUT))
    assert len(chk) == len(out), "save verify failed"
    sample = next(iter(out))
    print(f"Saved {len(out)} tensors → {OUT}  (sample {sample}: {tuple(out[sample].shape)})")
    with open(MANIFEST, "w") as f:
        json.dump({"n_snorna": n_used, "n_skip": n_skip, "layer": LAYER,
                   "snodb_ids": sorted(k.replace(f"__layer{LAYER}", "") for k in out)}, f, indent=2)
    print(f"Wrote {MANIFEST}. Next: step R3 (preprocess with TRAINING normalization) + R4 (SAE).")


if __name__ == "__main__":
    main()
