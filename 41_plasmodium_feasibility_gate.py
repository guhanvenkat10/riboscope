"""
41_plasmodium_feasibility_gate.py — RIBOSCOPE (malaria) step P2: feasibility gate.

Question: does RNA-FM recognize AT-rich P. falciparum structured ncRNAs, or are
they out-of-distribution (which would null the malaria hunt like the human lncRNAs)?

Test: build the human C/D-snoRNA "feature fingerprint" (the SAE features that mark
human snoRNAs), then check whether Plasmodium snoRNAs activate those same features
and resemble the human fingerprint — and clearly MORE than Plasmodium rRNA (a
within-species negative). If yes → in-distribution → GO. If Plasmodium snoRNA
features are degenerate/near-zero or no better than rRNA → OOD → adapt.

Run with
--------
    cd ~/projects/riboscope
    uv run python 41_plasmodium_feasibility_gate.py
    ~/projects/riboscope/sync_to_windows.sh

Inputs : sequences/plasmodium_ncrna.fasta (from P1), outputs/snodb_cd_features_rnafm.safetensors,
         outputs/sae_big_layer6_v3.safetensors
Output : outputs/plasmodium_feasibility_gate.json (+ saves Plasmodium features)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from multimolecule import RnaTokenizer, RnaFmModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE

FASTA = Path("sequences/plasmodium_ncrna.fasta")
HUMAN_FEATS = Path("outputs/snodb_cd_features_rnafm.safetensors")
SAE_FILE = Path("outputs/sae_big_layer6_v3.safetensors")
LAYER = 6
N_MARKER = 50
OUT_FEATS = Path("outputs/plasmodium_features_rnafm.safetensors")
OUT_JSON = Path("outputs/plasmodium_feasibility_gate.json")


def find_encoder_layers(model):
    for p in ("encoder.layer", "bert.encoder.layer", "roberta.encoder.layer", "model.encoder.layer"):
        obj = model
        try:
            for part in p.split("."):
                obj = getattr(obj, part)
            _ = len(obj)
            return obj
        except (AttributeError, TypeError):
            continue
    raise AttributeError("encoder.layer not found")


def make_hook(cap, li):
    def hook(m, i, o):
        cap[li] = (o[0] if isinstance(o, tuple) else o).detach().to(torch.float16).cpu()
    return hook


def parse_fasta(path):
    out, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(buf)))
                name = line[1:]
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            out.append((name, "".join(buf)))
    return out


def main():
    for p in (FASTA, HUMAN_FEATS, SAE_FILE):
        if not p.exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    print("=" * 72)
    print("RIBOSCOPE P2: Plasmodium feasibility gate (RNA-FM)")
    print("=" * 72)

    recs = parse_fasta(FASTA)
    seqs = [(h.split("|")[0], (h.split("|")[1] if "|" in h else "ncRNA"),
             s.upper().replace("T", "U")) for h, s in recs]

    sae_state = load_file(str(SAE_FILE))
    d_in, d_dict = sae_state["W_enc"].shape[0], sae_state["W_enc"].shape[1]
    sae = BatchTopKSAE(d_input=d_in, d_dict=d_dict, k=32)
    sae.load_state_dict(sae_state); sae.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)

    tok = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model = RnaFmModel.from_pretrained("multimolecule/rnafm").eval().to(device)
    layers = find_encoder_layers(model)
    cap = {}
    h = layers[LAYER].register_forward_hook(make_hook(cap, LAYER))

    print(f"[1/3] Extracting RNA-FM SAE features for {len(seqs)} Plasmodium ncRNAs ...")
    feats_by_type = defaultdict(list)
    all_rows, all_types, all_ids = [], [], []
    with torch.no_grad():
        for acc, rtype, seq in tqdm(seqs, unit="seq"):
            inp = tok(seq, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            _ = model(**inp)
            act = cap[LAYER][0].float().to(device)
            nt = act.shape[0]
            if nt <= 2:
                continue
            sm = sae.encode(act)[1:nt - 1].max(dim=0).values.cpu().numpy()
            feats_by_type[rtype].append(sm)
            all_rows.append(sm); all_types.append(rtype); all_ids.append(acc)
    h.remove()
    P = np.stack(all_rows)  # [n_plas, d_dict]

    human = load_file(str(HUMAN_FEATS))["sae_max"].float().numpy()  # [1103, d_dict] human C/D snoRNAs
    fingerprint = human.mean(axis=0)
    marker = np.argsort(-fingerprint)[:N_MARKER]

    def report(name, X):
        if len(X) == 0:
            return None
        X = np.asarray(X)
        marker_score = float(X[:, marker].mean())
        # cosine to human fingerprint
        fn = fingerprint / (np.linalg.norm(fingerprint) + 1e-9)
        cos = (X @ fn) / (np.linalg.norm(X, axis=1) + 1e-9)
        return {"n": len(X), "marker_score": round(marker_score, 3),
                "cos_to_human_snoRNA": round(float(cos.mean()), 3),
                "mean_activation": round(float(X.mean()), 3)}

    print("\n[2/3] Recognition profile (higher marker_score / cosine = more snoRNA-like to RNA-FM):")
    groups = {
        "HUMAN C/D snoRNA (ref)": human,
        "Plasmodium snoRNA": feats_by_type.get("snoRNA", []),
        "Plasmodium snRNA": feats_by_type.get("snRNA", []),
        "Plasmodium rRNA (neg)": feats_by_type.get("rRNA", []),
        "Plasmodium ncRNA (other)": feats_by_type.get("ncRNA", []),
    }
    res = {}
    print(f"  {'group':<26}{'n':>5}{'marker':>9}{'cos_sno':>9}{'mean_act':>9}")
    for name, X in groups.items():
        r = report(name, X)
        if r:
            res[name] = r
            print(f"  {name:<26}{r['n']:>5}{r['marker_score']:>9.3f}{r['cos_to_human_snoRNA']:>9.3f}{r['mean_activation']:>9.3f}")

    # verdict
    hs = res.get("HUMAN C/D snoRNA (ref)", {})
    ps = res.get("Plasmodium snoRNA", {})
    pr = res.get("Plasmodium rRNA (neg)", {})
    print("\n[3/3] VERDICT")
    ok = False
    if ps and hs:
        sno_vs_human = ps["cos_to_human_snoRNA"] / (hs["cos_to_human_snoRNA"] + 1e-9)
        sep = ps["cos_to_human_snoRNA"] - (pr["cos_to_human_snoRNA"] if pr else 0.0)
        print(f"  Plasmodium snoRNA cosine = {ps['cos_to_human_snoRNA']:.3f} "
              f"(human ref {hs['cos_to_human_snoRNA']:.3f}; ratio {sno_vs_human:.2f})")
        if pr:
            print(f"  vs Plasmodium rRNA cosine = {pr['cos_to_human_snoRNA']:.3f}  (separation {sep:+.3f})")
        ok = (ps["marker_score"] > 0.3 * hs["marker_score"]) and (not pr or ps["cos_to_human_snoRNA"] > pr["cos_to_human_snoRNA"])
    if ok:
        print("  ✓ IN-DISTRIBUTION — RNA-FM recognizes Plasmodium snoRNAs as snoRNA-like and")
        print("    distinguishes them from rRNA. The malaria hunt is LIVE → proceed to scoring +")
        print("    essentiality cross-reference.")
    else:
        print("  ✗ LIKELY OUT-OF-DISTRIBUTION — Plasmodium snoRNA features are weak/degenerate or")
        print("    not separable from rRNA. We adapt (e.g., per-family MSA for RNA-MSM) or reconsider.")

    save_file({"sae_max": torch.tensor(P)}, str(OUT_FEATS))
    with open(OUT_JSON, "w") as f:
        json.dump({"n_plasmodium": len(all_ids), "marker_feature_idx": marker.tolist(),
                   "groups": res, "verdict_in_distribution": bool(ok),
                   "ids": all_ids, "types": all_types}, f, indent=2)
    print(f"\nSaved {OUT_FEATS} and {OUT_JSON}.")


if __name__ == "__main__":
    main()
