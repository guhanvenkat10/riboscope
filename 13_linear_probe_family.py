"""
13_linear_probe_family.py — RIBOSCOPE: is Rfam-family identity linearly
decodable from layer-6 activations? The decisive iterate-vs-negative-case test.

Why this exists
---------------
RNA-MSM's SAE has failed three times (v1 dict=8192, v2 dict=4096 + N-mask, v3
position-centered). v3 fixed *global* health (dead 3.8% vs v2's 53%; top feature
magnitudes 3.4-7.5) but did NOT produce strong family-specialist features: the
top family-selective feature maxes at only 0.137 — ~55x below the top generalist
(7.498) and far below RNA-FM's healthy specialists (5-65). Two artifacts remain
(a 67-position uncentered tail; generic single-nucleotide detectors firing on
28-72% of ALL tokens).

Before spending another ~2 hr training run trying to coax specialists out, answer
a prior question with EVIDENCE:

    Is family-discriminative signal even PRESENT (linearly) in RNA-MSM's
    layer-6 representation — and how does it compare to the two models whose
    SAEs DID yield specialists (RNA-FM EV=0.882, ErnieRNA EV=0.679)?

A linear probe is the right instrument. An SAE encoder row is itself a linear
readout + a nonlinearity, so "can a linear classifier recover family from a
single token's activation" upper-bounds what a single SAE feature could pick up.
  - If RNA-MSM's per-token family signal is comparable to RNA-FM / ErnieRNA, the
    SAE is mis-allocating capacity -> ITERATE (fix the tail + penalize generic
    high-frequency features and retrain).
  - If it is far weaker, RNA-MSM is a legitimate cross-model NEGATIVE case: an
    MSA Transformer (ESM-MSA-1b lineage) run out of distribution on single
    sequences (msa_depth=1) genuinely loses family/coevolutionary signal at
    layer 6. That is a publishable CONTRAST that makes the two single-sequence-
    native successes meaningful — not a failure. The cross-model thesis never
    required all three to succeed.

Design (held FIXED across all models for a fair comparison)
-----------------------------------------------------------
- Classes: the TOP_K_FAMILIES most-populated Rfam families (each capped at 25
  seqs by 06_fetch_rfam_sequences.py), chosen ONCE from the FASTA so every model
  solves the identical K-way problem. Restricting to populated families removes
  class scarcity as a confound — a ~4000-way problem with ~7 seqs/class would be
  unwinnable regardless of signal.
- Split: stratified by family at the SEQUENCE level (never token level),
  computed ONCE and applied to every model, so (a) all models are tested on the
  exact same held-out sequences and (b) no token from a training sequence can
  leak into the test set (which would inflate token-level accuracy).
- Two probe granularities:
    * token-level    — each real token's activation -> its family. This is the
      level the SAE actually operates on; it is the load-bearing number.
    * sequence-level — mean-pooled real tokens -> family. A generous test of
      whether family info exists ANYWHERE in the representation. The gap between
      the two is itself diagnostic: high seq / low token => family info is
      diffuse, not localized per-token => structurally hard for a per-token SAE.
- Token filtering mirrors SAE training/inspection EXACTLY: drop CLS/EOS and N.
- Per-model feature standardization (z-score on TRAIN stats) so RNA-MSM's
  normalized+centered scale vs the others' native scales cannot bias the probe.
- Controls: chance = 1/n_classes, plus a label-SHUFFLE refit (must collapse to
  chance) to prove the pipeline is not leaking.

RNA-MSM appears twice on purpose:
    rnamsm_v3   = position-centered acts (what the v3 SAE saw)
    rnamsm_pre  = normalized-but-not-centered acts (the v2 input)
  Position-centering subtracts a per-position mean computed across ALL families,
  i.e. a family-BLIND operation that cannot add or remove family information in
  principle. Probing both confirms that empirically (expect ~equal).

Run with
--------
    cd ~/projects/riboscope
    uv run python 13_linear_probe_family.py
    # To change difficulty, edit TOP_K_FAMILIES (e.g. 50 or 200) and re-run.

Output
------
    outputs/linear_probe_family.json   (full metrics, auditable)
    console comparison table + suggested verdict
"""

from __future__ import annotations

import gc
import json
import random
import sys
from collections import Counter
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)


# ============================ CONFIG ============================
LAYER = 6

# Activation files to probe. RNA-MSM appears twice (v3 vs pre-centering).
# Keys in every file end with "__layer{LAYER}" (verified across the pipeline).
MODELS = [
    {"tag": "rnafm",      "act_file": "outputs/activations_big_layer6_v2.safetensors",
     "note": "RNA-FM native acts (SAE EV=0.882, real specialists)"},
    {"tag": "erniarna",   "act_file": "outputs/activations_erniarna_layer6.safetensors",
     "note": "ErnieRNA native acts (SAE EV=0.679, real specialists)"},
    {"tag": "rnamsm_v3",  "act_file": "outputs/activations_rnamsm_layer6_poscentered.safetensors",
     "note": "RNA-MSM position-centered acts (what the v3 SAE saw)"},
    {"tag": "rnamsm_pre", "act_file": "outputs/activations_rnamsm_layer6.safetensors",
     "note": "RNA-MSM normalized, NOT position-centered (v2 input; control)"},
]

FASTA_FILE = Path("sequences/rfam_30k.fasta")
REPORT_OUT = Path("outputs/linear_probe_family.json")

# Difficulty knob: how many (most-populated) families form the K-way problem.
TOP_K_FAMILIES = 100

# Stratified sequence-level split.
TEST_FRAC = 0.25
SEED = 0

# Token filtering — mirror SAE training / 09 inspection.
SKIP_SPECIAL_TOKENS = True   # drop CLS (pos 0) and EOS (pos n-1)
SKIP_N_TOKENS = True         # drop N nucleotide positions

# Cap on TRAIN tokens for the token-level probe (test tokens are never capped).
# Top-100 families * ~25 seqs * ~200 nt ~= 500k tokens, so this is rarely hit;
# it just bounds runtime/RAM if TOP_K_FAMILIES is raised a lot.
MAX_TRAIN_TOKENS = 1_500_000

# Linear-probe optimization (multinomial logistic regression via a torch
# Linear + cross-entropy = exactly a linear probe).
PROBE_EPOCHS = 60
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
PROBE_BATCH = 4096
# ================================================================


def parse_fasta_with_metadata(path: Path) -> dict:
    """Parse the Rfam FASTA produced by 06_fetch_rfam_sequences.py.

    Header format: ">{name} | {rfam_id} | {rfam_name} | {len} nt".
    Returns dict[name -> {sequence, rfam_id, rfam_name}] with T->U applied.
    (Same parser as 08 / 09 so token indexing is identical.)
    """
    out: dict[str, dict] = {}
    name = rfam_id = rfam_name = None
    seq_buf: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = {"sequence": "".join(seq_buf),
                                 "rfam_id": rfam_id, "rfam_name": rfam_name}
                parts = [p.strip() for p in line[1:].split("|")]
                name = parts[0] if len(parts) > 0 else "unknown"
                rfam_id = parts[1] if len(parts) > 1 else None
                rfam_name = parts[2] if len(parts) > 2 else None
                seq_buf = []
            else:
                seq_buf.append(line.replace("T", "U").replace("t", "u"))
        if name is not None:
            out[name] = {"sequence": "".join(seq_buf),
                         "rfam_id": rfam_id, "rfam_name": rfam_name}
    return out


def real_token_mask(n: int, seq_upper: str) -> torch.Tensor:
    """True for tokens to keep. Mirrors 09: pos 0 = CLS, pos n-1 = EOS,
    pos 1..n-2 = nucleotides (nt_pos = pos-1). Drops special + N tokens."""
    m = torch.ones(n, dtype=torch.bool)
    if SKIP_SPECIAL_TOKENS:
        m[0] = False
        m[n - 1] = False
    if SKIP_N_TOKENS:
        for pos in range(1, n - 1):
            nt = pos - 1
            if nt < len(seq_upper) and seq_upper[nt] == "N":
                m[pos] = False
    return m


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    """Unweighted mean F1 over classes that appear in `labels`."""
    f1s = []
    for c in range(n_classes):
        gt = labels == c
        if not bool(gt.any()):
            continue  # class absent from this (test) split — skip
        pp = preds == c
        tp = int((pp & gt).sum())
        fp = int((pp & ~gt).sum())
        fn = int((~pp & gt).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def standardize(Xtr: torch.Tensor, Xte: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-score per dim using TRAIN statistics; apply to both splits."""
    mu = Xtr.mean(dim=0, keepdim=True)
    sd = Xtr.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_linear_probe(Xtr, ytr, Xte, yte, n_classes, device, *, label=""):
    """Train a linear softmax probe; return {top1, top5, macro_f1} on test."""
    d = Xtr.shape[1]
    lin = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    n = Xtr.shape[0]
    for epoch in range(PROBE_EPOCHS):
        perm = torch.randperm(n)
        lin.train()
        for i in range(0, n, PROBE_BATCH):
            idx = perm[i:i + PROBE_BATCH]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = ytr[idx].to(device, non_blocking=True)
            opt.zero_grad()
            loss = ce(lin(xb), yb)
            loss.backward()
            opt.step()

    # ---- Evaluate (chunked) ----
    lin.eval()
    k5 = min(5, n_classes)
    correct1 = correct5 = 0
    all_preds = torch.empty(Xte.shape[0], dtype=torch.long)
    with torch.no_grad():
        for i in range(0, Xte.shape[0], 65536):
            xb = Xte[i:i + 65536].to(device, non_blocking=True)
            logits = lin(xb)
            top5 = logits.topk(k5, dim=1).indices.cpu()
            yb = yte[i:i + 65536]
            p1 = top5[:, 0]
            all_preds[i:i + xb.shape[0]] = p1
            correct1 += int((p1 == yb).sum())
            correct5 += int((top5 == yb.unsqueeze(1)).any(dim=1).sum())
    n_te = Xte.shape[0]
    return {
        "top1": correct1 / n_te,
        "top5": correct5 / n_te,
        "macro_f1": macro_f1(all_preds, yte, n_classes),
        "n_test": int(n_te),
    }


def build_matrices(acts, selected_seqs, seq_meta, fam_to_int, split):
    """Assemble token-level and sequence-level (mean-pooled) train/test matrices
    for sequences present in this model's activation file."""
    tok_tr, ytk_tr, tok_te, ytk_te = [], [], [], []
    sq_tr, ysq_tr, sq_te, ysq_te = [], [], [], []
    n_used = 0
    for name in selected_seqs:
        key = f"{name}__layer{LAYER}"
        if key not in acts:
            continue
        t = acts[key].float()                 # [n, d]
        n = t.shape[0]
        seq_upper = seq_meta[name]["sequence"].upper()
        y = fam_to_int[seq_meta[name]["rfam_id"]]
        rows = t[real_token_mask(n, seq_upper)]
        if rows.shape[0] == 0:
            continue
        n_used += 1
        pooled = rows.mean(dim=0, keepdim=True)
        if split[name] == "train":
            tok_tr.append(rows); ytk_tr.append(torch.full((rows.shape[0],), y, dtype=torch.long))
            sq_tr.append(pooled); ysq_tr.append(y)
        else:
            tok_te.append(rows); ytk_te.append(torch.full((rows.shape[0],), y, dtype=torch.long))
            sq_te.append(pooled); ysq_te.append(y)

    out = {
        "Xtok_tr": torch.cat(tok_tr), "ytok_tr": torch.cat(ytk_tr),
        "Xtok_te": torch.cat(tok_te), "ytok_te": torch.cat(ytk_te),
        "Xseq_tr": torch.cat(sq_tr), "yseq_tr": torch.tensor(ysq_tr, dtype=torch.long),
        "Xseq_te": torch.cat(sq_te), "yseq_te": torch.tensor(ysq_te, dtype=torch.long),
        "n_seqs_used": n_used,
    }

    # Cap TRAIN tokens (test never capped) to bound runtime/RAM.
    if out["Xtok_tr"].shape[0] > MAX_TRAIN_TOKENS:
        g = torch.Generator().manual_seed(SEED)
        keep = torch.randperm(out["Xtok_tr"].shape[0], generator=g)[:MAX_TRAIN_TOKENS]
        out["Xtok_tr"] = out["Xtok_tr"][keep]
        out["ytok_tr"] = out["ytok_tr"][keep]
    return out


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("RIBOSCOPE: linear probe for Rfam family (iterate-vs-negative-case test)")
    print("=" * 80)

    if not FASTA_FILE.exists():
        print(f"❌ FASTA not found: {FASTA_FILE}")
        sys.exit(1)

    # ---- 1. FASTA + family selection (ONCE, shared across models) ----
    print(f"[1/4] Parsing {FASTA_FILE} and selecting top-{TOP_K_FAMILIES} families...")
    seq_meta = parse_fasta_with_metadata(FASTA_FILE)
    fam_counts = Counter(m["rfam_id"] for m in seq_meta.values() if m["rfam_id"])
    # Most-populated families; deterministic tie-break by family id.
    ranked = sorted(fam_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    selected_families = [fid for fid, _ in ranked[:TOP_K_FAMILIES]]
    fam_to_int = {f: i for i, f in enumerate(selected_families)}
    n_classes = len(selected_families)
    sizes = [fam_counts[f] for f in selected_families]
    selected_seqs = sorted(n for n, m in seq_meta.items() if m["rfam_id"] in fam_to_int)
    print(f"      Total families in FASTA:   {len(fam_counts)}")
    print(f"      Selected families (K):     {n_classes}")
    print(f"      Seqs/family in selection:  min={min(sizes)}  max={max(sizes)}  "
          f"median={sorted(sizes)[len(sizes)//2]}")
    print(f"      Total selected sequences:  {len(selected_seqs)}")
    print(f"      Chance accuracy (1/K):     {1.0 / n_classes:.4f}")

    # ---- 2. Stratified sequence-level split (ONCE, shared across models) ----
    print(f"[2/4] Building stratified sequence-level split (test_frac={TEST_FRAC})...")
    by_fam: dict[str, list[str]] = {f: [] for f in fam_to_int}
    for name in selected_seqs:
        by_fam[seq_meta[name]["rfam_id"]].append(name)
    rng = random.Random(SEED)
    split: dict[str, str] = {}
    n_train = n_test = 0
    for fam, names in by_fam.items():
        names = sorted(names)
        rng.shuffle(names)
        n_te = max(1, int(round(len(names) * TEST_FRAC))) if len(names) > 1 else 0
        for j, nm in enumerate(names):
            split[nm] = "test" if j < n_te else "train"
        n_test += n_te
        n_train += len(names) - n_te
    print(f"      Train sequences: {n_train}    Test sequences: {n_test}")

    # ---- 3. Per-model probes ----
    print(f"[3/4] Probing {len(MODELS)} activation sources on device={device}...")
    results: dict[str, dict] = {}
    for cfg in MODELS:
        tag, act_path = cfg["tag"], Path(cfg["act_file"])
        print("\n" + "-" * 80)
        print(f"MODEL: {tag}   ({cfg['note']})")
        print(f"  act_file: {act_path}")
        if not act_path.exists():
            print(f"  ⚠ MISSING — skipping {tag}.")
            results[tag] = {"error": "missing_act_file", "act_file": str(act_path)}
            continue

        acts = load_file(str(act_path))
        d_input = next(iter(acts.values())).shape[1]
        M = build_matrices(acts, selected_seqs, seq_meta, fam_to_int, split)
        del acts
        gc.collect()
        print(f"  d_input={d_input}  seqs_used={M['n_seqs_used']}  "
              f"train_tok={M['Xtok_tr'].shape[0]:,}  test_tok={M['Xtok_te'].shape[0]:,}")

        # Standardize (train stats) then fit.
        Xtok_tr, Xtok_te = standardize(M["Xtok_tr"], M["Xtok_te"])
        Xseq_tr, Xseq_te = standardize(M["Xseq_tr"], M["Xseq_te"])

        tok = fit_linear_probe(Xtok_tr, M["ytok_tr"], Xtok_te, M["ytok_te"],
                               n_classes, device, label=f"{tag}/token")
        seqr = fit_linear_probe(Xseq_tr, M["yseq_tr"], Xseq_te, M["yseq_te"],
                                n_classes, device, label=f"{tag}/seq")

        # Shuffle control (sequence level): permute train labels, refit.
        g = torch.Generator().manual_seed(SEED)
        y_shuf = M["yseq_tr"][torch.randperm(M["yseq_tr"].shape[0], generator=g)]
        shuf = fit_linear_probe(Xseq_tr, y_shuf, Xseq_te, M["yseq_te"],
                                n_classes, device, label=f"{tag}/seq-shuffled")

        chance = 1.0 / n_classes
        results[tag] = {
            "d_input": int(d_input),
            "n_seqs_used": M["n_seqs_used"],
            "n_train_tokens": int(M["Xtok_tr"].shape[0]),
            "n_test_tokens": int(M["Xtok_te"].shape[0]),
            "chance": chance,
            "token_level": tok,
            "sequence_level": seqr,
            "sequence_level_shuffled": shuf,
            "token_signal_above_chance": tok["top1"] - chance,
            "sequence_signal_above_chance": seqr["top1"] - chance,
        }
        print(f"  TOKEN-LEVEL : top1={tok['top1']:.4f}  top5={tok['top5']:.4f}  "
              f"macroF1={tok['macro_f1']:.4f}   (chance={chance:.4f})")
        print(f"  SEQ-LEVEL   : top1={seqr['top1']:.4f}  top5={seqr['top5']:.4f}  "
              f"macroF1={seqr['macro_f1']:.4f}")
        print(f"  SHUFFLE CTRL: top1={shuf['top1']:.4f}  (should ~= chance {chance:.4f})")

        del M, Xtok_tr, Xtok_te, Xseq_tr, Xseq_te
        gc.collect()

    # ---- 4. Comparison + suggested verdict ----
    print("\n" + "=" * 80)
    print("[4/4] COMPARISON (test top-1; signal = top1 - chance)")
    print("=" * 80)
    hdr = f"{'model':<14}{'tok_top1':>10}{'tok_sig':>10}{'seq_top1':>10}{'seq_sig':>10}{'shuffle':>10}"
    print(hdr)
    print("-" * len(hdr))
    for tag, r in results.items():
        if "error" in r:
            print(f"{tag:<14}  MISSING ({r['act_file']})")
            continue
        print(f"{tag:<14}"
              f"{r['token_level']['top1']:>10.4f}"
              f"{r['token_signal_above_chance']:>10.4f}"
              f"{r['sequence_level']['top1']:>10.4f}"
              f"{r['sequence_signal_above_chance']:>10.4f}"
              f"{r['sequence_level_shuffled']['top1']:>10.4f}")

    # Automated reading (token-level signal-above-chance is the load-bearing #).
    def tok_sig(tag):
        r = results.get(tag, {})
        return r.get("token_signal_above_chance") if "error" not in r else None

    rf, er, ms = tok_sig("rnafm"), tok_sig("erniarna"), tok_sig("rnamsm_v3")
    print("\nSUGGESTED VERDICT")
    print("-" * 80)
    refs = [v for v in (rf, er) if v is not None]
    if ms is None or not refs:
        print("  Incomplete — need rnamsm_v3 plus at least one of rnafm/erniarna.")
    else:
        ref = sum(refs) / len(refs)
        ratio = ms / ref if ref > 0 else 0.0
        print(f"  RNA-MSM(v3) token signal-above-chance : {ms:.4f}")
        print(f"  Reference (mean of rnafm/erniarna)    : {ref:.4f}")
        print(f"  Ratio RNA-MSM / reference             : {ratio:.2f}")
        seq_ms = results["rnamsm_v3"]["sequence_signal_above_chance"]
        tok_ms = ms
        if ratio >= 0.7:
            print("  => ITERATE. Per-token family signal in RNA-MSM is comparable to the")
            print("     models whose SAEs succeeded. The SAE is mis-allocating capacity,")
            print("     not missing signal. Next: center the full sequence length (drop the")
            print("     MIN_CONTRIBUTORS tail) and penalize high-frequency generic features,")
            print("     then retrain.")
        elif ratio <= 0.4:
            print("  => NEGATIVE CASE is justified. Per-token family signal in RNA-MSM is")
            print("     far weaker than RNA-FM/ErnieRNA. No SAE tuning can extract signal")
            print("     that is not there. Document RNA-MSM as the cross-model negative case")
            print("     (MSA model at msa_depth=1 loses family signal at layer 6).")
            if seq_ms > 0 and tok_ms / max(seq_ms, 1e-9) < 0.5:
                print("     NOTE: sequence-level signal is much stronger than token-level —")
                print("     family info exists but is DIFFUSE, not localized per token, which")
                print("     is exactly why a per-token SAE cannot form specialists here.")
        else:
            print("  => AMBIGUOUS. Inspect the token-vs-sequence gap and consider one")
            print("     targeted iteration before deciding. If seq>>token, lean negative")
            print("     (diffuse signal); if seq~token, lean iterate.")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump({
            "config": {
                "top_k_families": TOP_K_FAMILIES,
                "n_classes": n_classes,
                "test_frac": TEST_FRAC,
                "seed": SEED,
                "skip_special_tokens": SKIP_SPECIAL_TOKENS,
                "skip_n_tokens": SKIP_N_TOKENS,
                "max_train_tokens": MAX_TRAIN_TOKENS,
                "probe_epochs": PROBE_EPOCHS,
                "probe_lr": PROBE_LR,
                "probe_weight_decay": PROBE_WEIGHT_DECAY,
                "n_train_sequences": n_train,
                "n_test_sequences": n_test,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n✅ Done. Full metrics -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
