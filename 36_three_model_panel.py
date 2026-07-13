"""
36_three_model_panel.py — RIBOSCOPE R5: the THREE-MODEL snoRNA functional-axis panel.

On the snoRNAs covered by all three models (RNA-FM, ErnieRNA, RNA-MSM/MSA), this:
  1. REPLICATION: per model, family-held-out (grouped) CV value-add of SAE features
     over trivial (length+conservation+3-mer) on non-canonical-vs-canonical.
  2. AGREEMENT: pairwise Spearman of each model's P(non-canonical) across the
     held-out orphans (do three independent architectures rank snoRNAs the same?).
  3. REDISCOVERY: P(non-canonical) for U3 / U8 / SNORD97 in all three models
     (orphans never in training).

Run with
--------
    cd ~/projects/riboscope
    uv run python 36_three_model_panel.py
    ~/projects/riboscope/sync_to_windows.sh

Output: outputs/snodb_three_model_panel.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

try:
    import numpy as np
    from safetensors.torch import load_file
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ Missing dependency: {e}  (uv pip install scikit-learn scipy)")
    sys.exit(1)

META = Path("outputs/snodb_cd_metadata.tsv")
FASTA = Path("sequences/snodb_cd.fasta")
RNAMSM_META = Path("outputs/snodb_cd_rnamsm_meta.tsv")
FEATS = {
    "RNA-FM":   Path("outputs/snodb_cd_features_rnafm.safetensors"),
    "ErnieRNA": Path("outputs/snodb_cd_features_erniarna.safetensors"),
    "RNA-MSM":  Path("outputs/snodb_cd_features_rnamsm.safetensors"),
}
N_SPLITS = 5


def load_meta():
    with open(META, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_fasta():
    seqs, name, buf = {}, None, []
    with open(FASTA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name = line[1:].split("|")[0]
                buf = []
            else:
                buf.append(line)
        if name:
            seqs[name] = "".join(buf)
    return seqs


def family_of(name, fallback):
    if not name or not name.strip():
        return fallback
    return re.sub(r"[-_]\d+[A-Za-z]?$", "", name.strip()) or fallback


def kmer_features(seqs, k=3):
    kmers = ["".join(p) for p in product("ACGU", repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)))
    for r, s in enumerate(seqs):
        n = 0
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k])
            if j is not None:
                X[r, j] += 1; n += 1
        if n:
            X[r] /= n
    return X


def grouped_auc(X, y, groups):
    splits = min(N_SPLITS, len(set(groups)))
    aucs = []
    for tr, te in GroupKFold(n_splits=splits).split(X, y, groups):
        if len(set(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return (float(np.mean(aucs)), float(np.std(aucs))) if aucs else (float("nan"), float("nan"))


def main():
    for p in [META, FASTA, RNAMSM_META] + list(FEATS.values()):
        if not Path(p).exists():
            print(f"❌ Missing {p}")
            sys.exit(1)

    meta = load_meta()
    id2row = {m["snodb_id"]: i for i, m in enumerate(meta)}
    seqs_by_id = load_fasta()

    # RNA-MSM covers a subset; its meta gives row->snodb_id
    rnamsm_ids = []
    with open(RNAMSM_META) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rnamsm_ids.append(r["snodb_id"])

    feats = {name: load_file(str(p)) for name, p in FEATS.items()}
    # shared set = RNA-MSM-covered ids that exist in metadata
    shared = [sid for sid in rnamsm_ids if sid in id2row]
    print("=" * 76)
    print(f"RIBOSCOPE R5: three-model snoRNA panel — shared snoRNAs = {len(shared)}")
    print("=" * 76)

    # per-model SAE matrix aligned to `shared`
    rnamsm_pos = {sid: i for i, sid in enumerate(rnamsm_ids)}
    sae = {}
    sae["RNA-FM"] = np.stack([feats["RNA-FM"]["sae_max"][id2row[s]].numpy() for s in shared])
    sae["ErnieRNA"] = np.stack([feats["ErnieRNA"]["sae_max"][id2row[s]].numpy() for s in shared])
    sae["RNA-MSM"] = np.stack([feats["RNA-MSM"]["sae_max"][rnamsm_pos[s]].numpy() for s in shared])

    M = [meta[id2row[s]] for s in shared]
    canon = np.array([int(m["label_canonical_rrna"]) == 1 for m in M])
    noncanon = np.array([int(m["label_noncanonical"]) == 1 for m in M])
    orphan = np.array([int(m["label_orphan"]) == 1 for m in M])
    groups_all = np.array([family_of(m["gene_name"], m["snodb_id"]) for m in M])
    seqs = [seqs_by_id.get(s, "") for s in shared]
    trivial = np.hstack([
        np.array([[float(m["length"]), float(m["conservation_phastcons"] or 0.0)] for m in M]),
        kmer_features(seqs, 3),
    ])

    # ---- 1. REPLICATION: grouped-CV value-add, non-canonical vs canonical ----
    contrast = noncanon | (canon & ~noncanon)
    yb = noncanon[contrast].astype(int)
    gb = groups_all[contrast]
    triv_b = trivial[contrast]
    base_auc = grouped_auc(triv_b, yb, gb)
    print(f"\n[1] REPLICATION — non-canonical vs canonical (n_pos={int(yb.sum())}, "
          f"n_neg={int((1-yb).sum())}), family-held-out CV")
    print(f"    {'model':<10}{'TRIVIAL':>10}{'+SAE':>10}{'value-add':>11}")
    print(f"    {'(trivial)':<10}{base_auc[0]:>10.3f}{'':>10}{'':>11}")
    replication = {"trivial_auc": base_auc[0]}
    for name in FEATS:
        full = grouped_auc(np.hstack([triv_b, sae[name][contrast]]), yb, gb)
        add = full[0] - base_auc[0]
        replication[name] = {"trivial+sae_auc": full[0], "value_add": add}
        print(f"    {name:<10}{base_auc[0]:>10.3f}{full[0]:>10.3f}{add:>+11.3f}")

    # ---- 2. AGREEMENT: per-model orphan P(non-canonical), pairwise Spearman ----
    p_orph = {}
    for name in FEATS:
        sc = StandardScaler().fit(sae[name][contrast])
        clf = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(sc.transform(sae[name][contrast]), yb)
        p_orph[name] = clf.predict_proba(sc.transform(sae[name][orphan]))[:, 1]
    print(f"\n[2] AGREEMENT — pairwise Spearman of P(non-canonical) across {int(orphan.sum())} held-out orphans")
    names = list(FEATS)
    agreement = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho, _ = spearmanr(p_orph[names[i]], p_orph[names[j]])
            agreement[f"{names[i]}~{names[j]}"] = float(rho)
            print(f"    {names[i]:<9} ~ {names[j]:<9}  rho={rho:.3f}")

    # ---- 3. REDISCOVERY across all three ----
    orph_ids = [shared[k] for k in np.where(orphan)[0]]
    id2orphpos = {sid: k for k, sid in enumerate(orph_ids)}
    def gene(sid): return M[shared.index(sid)]["gene_name"].strip() if sid in shared else ""
    targets = {
        "U3": [s for s in orph_ids if gene(s) == "U3"],
        "U8": [s for s in orph_ids if gene(s) == "U8"],
        "SNORD97(snoDB0492)": [s for s in orph_ids if s == "snoDB0492"],
    }
    print(f"\n[3] REDISCOVERY — mean P(non-canonical), held-out orphans, all 3 models")
    print(f"    {'target':<22}{'n':>4}{'RNA-FM':>9}{'ErnieRNA':>10}{'RNA-MSM':>9}")
    rediscovery = {}
    for label, ids in targets.items():
        if not ids:
            print(f"    {label:<22}{'(not in covered set)':>32}")
            continue
        row = {name: float(np.mean([p_orph[name][id2orphpos[s]] for s in ids])) for name in FEATS}
        rediscovery[label] = {"n": len(ids), **row}
        print(f"    {label:<22}{len(ids):>4}{row['RNA-FM']:>9.2f}{row['ErnieRNA']:>10.2f}{row['RNA-MSM']:>9.2f}")

    out = Path("outputs/snodb_three_model_panel.json")
    with open(out, "w") as f:
        json.dump({"n_shared": len(shared), "replication": replication,
                   "agreement_spearman": agreement, "rediscovery": rediscovery}, f, indent=2)
    print(f"\nSaved {out}.")
    print("\nRead: all 3 value-adds > ~0 = axis replicates across architectures; high")
    print("pairwise rho + all 3 flagging U3/U8/SNORD97 = three independent AIs agree.")


if __name__ == "__main__":
    main()
