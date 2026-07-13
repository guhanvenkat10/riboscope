"""
21_scan_disease_rna.py — RIBOSCOPE G3 (disease-discovery) step 2: the scanner.

Idea
----
Use the locked SAE family-specialist features as an INTERPRETABLE ANNOTATOR.
Tile a disease lncRNA into in-distribution windows, run the model, encode the
layer activations with the trained SAE, and record WHERE each named specialist
(or moderate) feature fires on the transcript and WHAT Rfam family it stands
for. A "hit" means: the model recognizes this region as resembling family X.

This directly respects our own headline (interpretability is conditional on
in-distribution input): MALAT1/NEAT1/BACE1-AS are far longer than the Rfam
50-510 nt training families, so we NEVER feed full length — we feed
WINDOW_SIZE-nt windows that sit inside the training distribution.

Scope
-----
RNA-FM and ErnieRNA only (both are in-distribution on single sequences).
RNA-MSM is deliberately NOT scanned here: it needs native-MSA input (our
locked result), so single windows would be out-of-distribution for it. If a
hit is worth confirming, we build per-window MSAs in a later step.

Firing threshold
----------------
A feature fires on a position iff its activation >= FIRING_FRAC * (that
feature's own max activation measured on Rfam during inspection). FIRING_FRAC
matches MAGNITUDE_THRESHOLD_FRAC in 09_inspect_features_big.py, so "fires"
means exactly what it means in the locked SAEs.

Run with
--------
    cd ~/projects/riboscope
    uv run python 21_scan_disease_rna.py rnafm
    uv run python 21_scan_disease_rna.py erniarna

Output
------
    outputs/disease_scan_{model}.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
    from multimolecule import RnaTokenizer, RnaFmModel, ErnieRnaModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE

# ============================ CONFIG ============================
WINDOW_SIZE = 200       # nt per window — inside Rfam's 50-510 training range
STRIDE = 100            # 50% overlap so elements aren't split across edges
MIN_WINDOW = 50         # don't score 3'-tail windows shorter than this (OOD)
FIRING_FRAC = 0.25      # == MAGNITUDE_THRESHOLD_FRAC in 09; "fires" definition
CONTEXT_WINDOW = 7      # nt of context on each side of a focal position

FASTA_FILE = Path("sequences/disease_lncrna.fasta")
RFAM_FASTA = Path("sequences/rfam_30k.fasta")   # used by --selftest positive control

# Per-model wiring. Layer + files match the LOCKED single-seq-in-distribution
# SAEs (RNA-FM and ErnieRNA are in-distribution on single sequences).
MODELS = {
    "rnafm": {
        "model_class": RnaFmModel,
        "model_name": "multimolecule/rnafm",
        "layer": 6,
        "sae_file": Path("outputs/sae_big_layer6_v3.safetensors"),
        "inspection_file": Path("outputs/inspection_big_layer6_v3.json"),
    },
    "erniarna": {
        "model_class": ErnieRnaModel,
        "model_name": "multimolecule/ernierna",
        "layer": 6,
        "sae_file": Path("outputs/sae_erniarna_layer6_v3.safetensors"),
        "inspection_file": Path("outputs/inspection_erniarna_layer6_v3.json"),
    },
}

# Known elements for the positive-control read-out (reported, not asserted).
# Rfam tRNA = RF00005; MALAT1's 3' mascRNA is tRNA-like, so a tRNA feature
# firing near MALAT1's 3' end would be a clean positive control.
KNOWN_CONTROLS = {
    "MALAT1": {"families": ["RF00005"], "where": "3' mascRNA (tRNA-like), terminal ~60 nt"},
}
TERMINAL_ZOOM_NT = 200  # report hits in the last N nt of each transcript
# ================================================================


def parse_disease_fasta(path: Path) -> list[dict]:
    """Parse the >GENE|accession|description FASTA. Converts T->U, uppercases."""
    out: list[dict] = []
    gene = acc = desc = None
    buf: list[str] = []

    def flush():
        if gene is not None:
            seq = "".join(buf).upper().replace("T", "U")
            out.append({"gene": gene, "accession": acc, "description": desc, "sequence": seq})

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                parts = [p.strip() for p in line[1:].split("|")]
                gene = parts[0] if parts else "unknown"
                acc = parts[1] if len(parts) > 1 else None
                desc = parts[2] if len(parts) > 2 else None
                buf = []
            else:
                buf.append(line)
        flush()
    return out


def find_encoder_layers(model):
    candidates = [
        "encoder.layer", "bert.encoder.layer", "roberta.encoder.layer",
        "ernie.encoder.layer", "model.encoder.layer",
    ]
    for p in candidates:
        obj = model
        try:
            for part in p.split("."):
                obj = getattr(obj, part)
            _ = len(obj)
            return obj, p
        except (AttributeError, TypeError):
            continue
    raise AttributeError("encoder.layer not found")


def make_hook(captured: dict, layer_idx: int):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured[layer_idx] = output.detach().to(torch.float16).cpu()
    return hook


def format_context(seq: str, pos: int, win: int) -> str:
    if pos < 0 or pos >= len(seq):
        return "[OUT-OF-RANGE]"
    lo, hi = max(0, pos - win), min(len(seq), pos + win + 1)
    pre, focal, post = seq[lo:pos], seq[pos], seq[pos + 1:hi]
    return f"{' ' * (win - len(pre))}{pre}[{focal}]{post}{' ' * (win - len(post))}"


def load_feature_table(inspection_file: Path) -> list[dict]:
    """Build the feature table from the inspection JSON's specialist + moderate buckets.

    Returns a list of {feat_idx, max_act, breadth, families}. Generalists (>=10
    families) are intentionally excluded — they fire everywhere and carry no
    family-specific meaning.
    """
    with open(inspection_file) as f:
        insp = json.load(f)
    table: dict[int, dict] = {}
    for bucket in ("specialist_features", "moderate_features"):
        for rec in insp.get(bucket, []):
            fi = int(rec["feature_idx"])
            if fi in table:
                continue
            table[fi] = {
                "feat_idx": fi,
                "max_act": float(rec["max_activation"]),
                "breadth": int(rec.get("n_families", 0)),
                "families": list(rec.get("families_sample", [])),
            }
    # Drop any feature with a non-positive max (can't define a threshold).
    return [v for v in table.values() if v["max_act"] > 0]


def parse_rfam_fasta(path: Path) -> dict:
    """Parse rfam_30k.fasta (>name|rfam_id|rfam_name). Returns {rfam_id: [(name, seq)]}."""
    fams: dict[str, list] = defaultdict(list)
    name = fam = None
    buf: list[str] = []

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


def run_selftest(model, tokenizer, sae, layer, captured, device, feat_table,
                 rfam_fasta: Path, n_families: int = 20) -> None:
    """POSITIVE CONTROL: do specialist features fire on their OWN Rfam family members?

    The specialist's max activation was measured by 09 on these same Rfam
    sequences. If our forward+encode pipeline matches 09's, a real family
    member should drive the specialist to ~its known max (frac ~1.0). If it
    doesn't, the scanner pipeline is broken and disease results are meaningless.
    """
    if not Path(rfam_fasta).exists():
        print(f"  selftest: {rfam_fasta} not found — cannot validate.")
        return
    specialists = [r for r in feat_table if r["breadth"] <= 2]
    fam_to_feat: dict[str, dict] = {}
    for r in specialists:
        for fam in r["families"]:
            # keep the strongest specialist claiming this family
            if fam not in fam_to_feat or r["max_act"] > fam_to_feat[fam]["max_act"]:
                fam_to_feat[fam] = r
    fams = parse_rfam_fasta(rfam_fasta)

    print("\n  POSITIVE-CONTROL SELF-TEST — specialists vs their own Rfam family members")
    print(f"  {'family':<10} {'feat':>6} {'feat_max':>9} {'best_act':>9} {'frac':>6} {'n':>4}  verdict")
    print("  " + "-" * 64)
    tested = passed = 0
    for fam, r in fam_to_feat.items():
        if fam not in fams:
            continue
        members = fams[fam]  # all members (<=25 per family); short seqs, cheap
        best_act = 0.0
        with torch.no_grad():
            for _, seq in members:
                if not seq:
                    continue
                inputs = tokenizer(seq, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
                act = captured[layer][0].float().to(device)
                ntok = act.shape[0]
                if ntok <= 2:
                    continue
                col = sae.encode(act)[1:ntok - 1, r["feat_idx"]]
                best_act = max(best_act, float(col.max().item()))
        frac = best_act / r["max_act"] if r["max_act"] > 0 else 0.0
        ok = frac >= 0.5
        tested += 1
        passed += int(ok)
        print(f"  {fam:<10} {r['feat_idx']:>6} {r['max_act']:>9.3f} "
              f"{best_act:>9.3f} {frac:>6.2f} {len(members):>4}  {'OK' if ok else 'WEAK ✗'}")
        if tested >= n_families:
            break

    print("  " + "-" * 64)
    if tested == 0:
        print("  ⚠ No specialist families overlapped rfam_30k.fasta — cannot validate.")
    else:
        print(f"  {passed}/{tested} specialist families fired >=0.5x their max on their own members.")
        if passed / tested < 0.6:
            print("  ❌ PIPELINE SUSPECT — specialists don't recover on their own families.")
            print("     Disease-scan results are NOT trustworthy until this is fixed.")
        else:
            print("  ✓ PIPELINE VALIDATED — a disease NEGATIVE can now be trusted.")
    if "RF00005" not in fam_to_feat:
        print("  note: RF00005 (tRNA) has NO specialist in this model, so the MALAT1")
        print("        mascRNA positive control in the disease scan is N/A (not a failure).")


def main() -> None:
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    selftest = "--selftest" in sys.argv
    if len(pos_args) < 1 or pos_args[0] not in MODELS:
        print(f"Usage: python 21_scan_disease_rna.py [{' | '.join(MODELS)}] [--selftest]")
        sys.exit(1)
    model_key = pos_args[0]
    cfg = MODELS[model_key]
    layer = cfg["layer"]

    print("=" * 78)
    print(f"RIBOSCOPE G3 scanner — model={model_key}  layer={layer}")
    print("=" * 78)

    for p in (FASTA_FILE, cfg["sae_file"], cfg["inspection_file"]):
        if not Path(p).exists():
            print(f"❌ Missing required file: {p}")
            sys.exit(1)

    # --- feature table -------------------------------------------------- #
    feat_table = load_feature_table(cfg["inspection_file"])
    n_spec = sum(1 for r in feat_table if r["breadth"] <= 2)
    n_mod = sum(1 for r in feat_table if r["breadth"] >= 3)
    print(f"[1/4] Feature table: {len(feat_table)} features "
          f"({n_spec} specialists <=2 fam, {n_mod} moderate 3-9 fam)")
    if len(feat_table) == 0:
        print("\n❌ ABORT: feature table is EMPTY — refusing to report a false null.")
        print(f"   {cfg['inspection_file']} has no 'specialist_features'/'moderate_features'")
        print("   arrays. That inspection predates the bucket-dump patch in 09. Regenerate it:")
        print(f"     uv run python set_model.py {model_key}")
        print("     uv run python 09_inspect_features_big.py")
        sys.exit(1)

    # --- SAE ------------------------------------------------------------ #
    sae_state = load_file(str(cfg["sae_file"]))
    d_input = sae_state["W_enc"].shape[0]
    d_dict = sae_state["W_enc"].shape[1]
    sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=32)
    sae.load_state_dict(sae_state)
    sae.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)
    print(f"[2/4] SAE: d_input={d_input}, d_dict={d_dict}, device={device}")

    feat_idx_list = [r["feat_idx"] for r in feat_table]
    feat_idx_tensor = torch.tensor(feat_idx_list, dtype=torch.long, device=device)
    feat_max_tensor = torch.tensor([r["max_act"] for r in feat_table],
                                   dtype=torch.float32, device=device)
    thresh_tensor = FIRING_FRAC * feat_max_tensor  # [F]

    # --- model ---------------------------------------------------------- #
    print(f"[3/4] Loading {cfg['model_name']} ...")
    tokenizer = RnaTokenizer.from_pretrained(cfg["model_name"])
    model = cfg["model_class"].from_pretrained(cfg["model_name"]).eval().to(device)
    layers, _ = find_encoder_layers(model)
    if d_input != getattr(model.config, "hidden_size", d_input):
        print(f"⚠ d_input {d_input} != model hidden_size {model.config.hidden_size}")
    captured: dict[int, torch.Tensor] = {}
    handle = layers[layer].register_forward_hook(make_hook(captured, layer))

    # --- positive-control self-test (validate the instrument, then exit) - #
    if selftest:
        run_selftest(model, tokenizer, sae, layer, captured, device, feat_table, RFAM_FASTA)
        handle.remove()
        return

    # --- scan ----------------------------------------------------------- #
    rnas = parse_disease_fasta(FASTA_FILE)
    print(f"[4/4] Scanning {len(rnas)} transcript(s) "
          f"(window={WINDOW_SIZE}, stride={STRIDE}, fire>={FIRING_FRAC}x max)\n")

    results: dict[str, dict] = {}
    with torch.no_grad():
        for rna in rnas:
            gene, seq = rna["gene"], rna["sequence"]
            L = len(seq)
            # hits aggregated to (feat_idx, global_pos) -> max activation
            hit_max: dict[tuple[int, int], float] = {}
            hit_win: dict[tuple[int, int], int] = {}
            n_windows = 0

            starts = list(range(0, L, STRIDE))
            for s in tqdm(starts, desc=f"{gene}", unit="win", leave=False):
                e = min(s + WINDOW_SIZE, L)
                if e - s < MIN_WINDOW:
                    continue
                n_windows += 1
                sub = seq[s:e]
                inputs = tokenizer(sub, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
                act = captured[layer][0].float().to(device)  # [ntok, hidden]
                ntok = act.shape[0]
                if ntok <= 2:
                    continue
                feats = sae.encode(act)                       # [ntok, d_dict]
                real = feats[1:ntok - 1]                      # residue tokens [R, d_dict]
                sub_acts = real.index_select(1, feat_idx_tensor)  # [R, F]
                mask = sub_acts >= thresh_tensor              # [R, F]
                nz = mask.nonzero(as_tuple=False)             # [(r, fcol)]
                if nz.numel() == 0:
                    continue
                for r, fcol in nz.tolist():
                    gpos = s + r                              # 0-based nt position
                    feat = feat_idx_list[fcol]
                    a = float(sub_acts[r, fcol].item())
                    key = (feat, gpos)
                    if a > hit_max.get(key, -1.0):
                        hit_max[key] = a
                        hit_win[key] = s

            # ---- assemble per-RNA hit records ---- #
            fmax = {r["feat_idx"]: r["max_act"] for r in feat_table}
            fbreadth = {r["feat_idx"]: r["breadth"] for r in feat_table}
            ffams = {r["feat_idx"]: r["families"] for r in feat_table}
            hits = []
            for (feat, gpos), a in hit_max.items():
                hits.append({
                    "feat_idx": feat,
                    "nt_pos0": gpos,
                    "nt_pos1": gpos + 1,
                    "breadth": fbreadth[feat],
                    "families": ffams[feat],
                    "activation": round(a, 4),
                    "frac_of_max": round(a / fmax[feat], 4),
                    "window_start": hit_win[(feat, gpos)],
                    "context": format_context(seq, gpos, CONTEXT_WINDOW),
                })
            hits.sort(key=lambda h: h["frac_of_max"], reverse=True)

            # ---- specialist family summary (clean labels, breadth<=2) ---- #
            fam_summary: dict[str, dict] = defaultdict(
                lambda: {"peak_frac": 0.0, "n_positions": 0, "positions": []})
            for h in hits:
                if h["breadth"] <= 2:
                    for fam in h["families"]:
                        fs = fam_summary[fam]
                        fs["n_positions"] += 1
                        fs["peak_frac"] = max(fs["peak_frac"], h["frac_of_max"])
                        if len(fs["positions"]) < 12:
                            fs["positions"].append(h["nt_pos0"])
            fam_ranked = dict(sorted(fam_summary.items(),
                                     key=lambda kv: kv[1]["peak_frac"], reverse=True))

            # ---- 3'-terminal zoom (known elements live here) ---- #
            term_lo = max(0, L - TERMINAL_ZOOM_NT)
            terminal_hits = [h for h in hits if h["nt_pos0"] >= term_lo]
            terminal_hits.sort(key=lambda h: h["frac_of_max"], reverse=True)

            # ---- positive-control flag ---- #
            pc = None
            if gene in KNOWN_CONTROLS:
                want = set(KNOWN_CONTROLS[gene]["families"])
                pc_hits = [h for h in hits
                           if want.intersection(h["families"])]
                pc_term = [h for h in pc_hits if h["nt_pos0"] >= term_lo]
                pc = {
                    "expected_families": sorted(want),
                    "where": KNOWN_CONTROLS[gene]["where"],
                    "any_hit": len(pc_hits) > 0,
                    "hit_in_terminal_zoom": len(pc_term) > 0,
                    "n_hits": len(pc_hits),
                    "best_terminal": pc_term[0] if pc_term else None,
                }

            results[gene] = {
                "accession": rna["accession"],
                "length": L,
                "n_windows": n_windows,
                "n_hits": len(hits),
                "n_specialist_hits": sum(1 for h in hits if h["breadth"] <= 2),
                "family_summary_specialist": fam_ranked,
                "terminal_zoom": {"region_start": term_lo, "hits": terminal_hits[:30]},
                "positive_control": pc,
                "hits": hits,
            }

            # ---- console summary ---- #
            print(f"  {gene:<10} len={L:>6,}  windows={n_windows:>4}  "
                  f"hits={len(hits):>5}  specialist_hits={results[gene]['n_specialist_hits']:>4}")
            top_fams = list(fam_ranked.items())[:6]
            if top_fams:
                print("      top specialist families (peak frac of max | #positions):")
                for fam, d in top_fams:
                    print(f"        {fam:<10} {d['peak_frac']:.2f}  | {d['n_positions']}")
            if pc:
                tag = "✓ HIT in 3' zoom" if pc["hit_in_terminal_zoom"] else (
                    "fired (not in 3' zoom)" if pc["any_hit"] else "✗ no hit")
                print(f"      positive control {pc['expected_families']} "
                      f"({pc['where']}): {tag}")

    handle.remove()

    out_file = Path(f"outputs/disease_scan_{model_key}.json")
    with open(out_file, "w") as f:
        json.dump({
            "model": model_key,
            "layer": layer,
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "min_window": MIN_WINDOW,
            "firing_frac": FIRING_FRAC,
            "sae_file": str(cfg["sae_file"]),
            "inspection_file": str(cfg["inspection_file"]),
            "n_features_scanned": len(feat_table),
            "n_specialists": n_spec,
            "n_moderate": n_mod,
            "rnas": results,
        }, f, indent=2)

    print("\n" + "=" * 78)
    print(f"✅ Scan complete → {out_file}")
    print("   Reminder: hits = 'model recognizes this region as family X'.")
    print("   A hit is only a CANDIDATE; Step 3 filters by cross-model agreement,")
    print("   conservation, cmsearch-negativity, and a novelty audit.")
    print("=" * 78)


if __name__ == "__main__":
    main()
