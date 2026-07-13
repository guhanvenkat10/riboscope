"""
12_cross_model_agreement.py — RIBOSCOPE: Cross-model feature agreement.

This is the analysis that distinguishes RIBOSCOPE from SAE-RNA. SAE-RNA used
RiNALMo only. We have three architectures with different structural inductive
biases — RNA-FM (structure-naive), ErnieRNA (direct secondary-structure),
RNA-MSM (evolutionary MSA) — plus, crucially, a NEGATIVE CONTROL: the same
RNA-MSM run OUT-OF-DISTRIBUTION on single sequences. The hypothesis:

    "Features that represent real RNA biology should appear *consistently*
     across models, while features that are artifacts of any single model's
     architecture — or of feeding a model out-of-distribution input — should
     not. Cross-model agreement is therefore a measure of biological legitimacy."

What changed vs the original (2026-05-30, for the input-conditional result)
--------------------------------------------------------------------------
1. PER-MODEL LAYER. RNA-FM and ErnieRNA SAEs are at layer 6; the RNA-MSM SAEs
   are at layer 8 (the layer where native-MSA input de-collapses the
   representation). The old single global LAYER could not address all three.
   Each model is compared at its own SAE's layer — "each model's best-
   decomposing layer," documented as a methods choice.

2. PER-PAIR SHARED-FAMILY RESTRICTION. The RNA-MSM (MSA / single) SAEs only ever
   saw the top-100 Rfam families (2327-seq subset); RNA-FM / ErnieRNA saw all
   ~3979. Computing Jaccard over the full 3979-family vocab would artificially
   deflate every RNA-MSM pairing purely from vocabulary mismatch. So for EACH
   pair we restrict the family profile to the families BOTH models actually
   observed (≥1 real token). RNA-FM↔ErnieRNA → ~all families; any pair with an
   RNA-MSM model → the ~99 shared families. This is the correct apples-to-apples.

3. NEGATIVE CONTROL. rnamsm_single (single-sequence, msa_depth=1, OOD) is
   included. Prediction: rnamsm_msa agrees with RNA-FM/ErnieRNA substantially
   MORE than rnamsm_single does. That contrast is a fourth, independent line of
   evidence that the MSA features are real biology and the headline holds:
   "SAE interpretability is conditional on in-distribution model input."

How we measure agreement
------------------------
For each model independently, compute a per-feature "family activation profile"
— the set of Rfam families the feature fires on above a 25%-of-its-own-max
threshold. For each pair, on the shared-family columns, compute pairwise Jaccard
between every feature of model A and model B. Two features "match" iff (a) their
Jaccard ≥ 0.5 and (b) they are reciprocal best matches (each is the other's best
match). Reciprocal best matching prevents over-counting.

Run with
--------
    cd ~/projects/riboscope
    uv run python 12_cross_model_agreement.py

Requires the SAEs (missing ones are skipped; need ≥2 to compare):
    RNA-FM            outputs/sae_big_layer6_v3.safetensors           (layer 6)
    ErnieRNA          outputs/sae_erniarna_layer6_v3.safetensors      (layer 6)
    RNA-MSM (MSA)     outputs/sae_rnamsm_msa_layer8_v1.safetensors    (layer 8)
    RNA-MSM (single)  outputs/sae_rnamsm_single_layer8_v1.safetensors (layer 8, control)
"""

import json
import sys
from itertools import combinations
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE


# ============================ CONFIG ============================
# Each model carries its OWN layer + FASTA (the query set its SAE was trained on).
# family lookup is by sequence name -> rfam_id, identical across FASTAs, so the
# unified vocabulary (built from rfam_30k) is a valid superset for all of them.
MODELS = [
    {
        "tag": "rnafm",
        "label": "RNA-FM",
        "layer": 6,
        "act_file": Path("outputs/activations_big_layer6_v2.safetensors"),
        "sae_file": Path("outputs/sae_big_layer6_v3.safetensors"),
    },
    {
        "tag": "erniarna",
        "label": "ErnieRNA",
        "layer": 6,
        "act_file": Path("outputs/activations_erniarna_layer6.safetensors"),
        "sae_file": Path("outputs/sae_erniarna_layer6_v3.safetensors"),
    },
    {
        "tag": "rnamsm_msa",
        "label": "RNA-MSM (MSA, in-dist)",
        "layer": 8,
        # Native-MSA input — the de-collapsed, specialist-rich SAE.
        "act_file": Path("outputs/activations_rnamsm_msa_layer8_poscentered.safetensors"),
        "sae_file": Path("outputs/sae_rnamsm_msa_layer8_v1.safetensors"),
    },
    {
        "tag": "rnamsm_single",
        "label": "RNA-MSM (single, OOD ctrl)",
        "layer": 8,
        # NEGATIVE CONTROL: same model/seqs/layer/preprocessing, single-seq input.
        "act_file": Path("outputs/activations_rnamsm_single_layer8_poscentered.safetensors"),
        "sae_file": Path("outputs/sae_rnamsm_single_layer8_v1.safetensors"),
    },
]

# Family vocabulary source (a superset covering every model's sequence names).
VOCAB_FASTA = Path("sequences/rfam_30k.fasta")

# Feature-firing parameters (match the inspection settings)
MAGNITUDE_THRESHOLD_FRAC = 0.25
SKIP_SPECIAL_TOKENS = True
CHUNK_SIZE = 10_000
SAE_K = 32

# Cross-model matching parameters
JACCARD_THRESHOLD = 0.5
TOP_MATCHES_TO_SHOW = 30
MIN_FAMILIES_PER_FEATURE = 2  # floor: ignore features active on fewer than this many shared families
# CEILING (added 2026-05-30): only SPECIFIC features can match. A feature that
# fires on ~all shared families is a non-discriminative generalist — two such
# features trivially score Jaccard≈1.0 with each other, which says nothing about
# shared biology and floods the reciprocal-match count with how-many-generalists-
# each-model-has. Restricting to features on ≤ this many families makes a match
# biologically meaningful (a shared specialist/moderate). 9 = the inspection's
# specialist(≤2)+moderate(3-9) band; generalists (≥10) are excluded.
MAX_FAMILIES_PER_FEATURE = 9

REPORT_OUT = Path("outputs/cross_model_agreement.json")
# ================================================================


def parse_fasta_metadata(path: Path) -> dict:
    """name -> {rfam_id}. Same parser as 09_inspect_features_big.py."""
    out = {}
    name = None
    rfam_id = None
    seq_buf = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = {"sequence": "".join(seq_buf), "rfam_id": rfam_id}
                parts = [p.strip() for p in line[1:].split("|")]
                name = parts[0]
                rfam_id = parts[1] if len(parts) > 1 else None
                seq_buf = []
            else:
                seq_buf.append(line.replace("T", "U").replace("t", "u"))
        if name is not None:
            out[name] = {"sequence": "".join(seq_buf), "rfam_id": rfam_id}
    return out


def compute_family_presence(model_cfg, seq_meta, family_to_int, n_families, device):
    """
    For one model: stream activations through its SAE and compute
      presence       [d_dict, n_families] bool — feature fires above its own
                     25%-max threshold on ≥1 (real) token of family X
      family_covered [n_families]        bool — family X had ≥1 real token here
    Returns (presence, d_dict, family_covered) or (None, 0, None) on dim mismatch.
    """
    layer = model_cfg["layer"]
    print(f"  Loading activations from {model_cfg['act_file']} (layer {layer})...")
    act_data = load_file(str(model_cfg["act_file"]))
    layer_keys = sorted(k for k in act_data.keys() if k.endswith(f"__layer{layer}"))
    if not layer_keys:
        print(f"  ⚠ No keys ending in __layer{layer} in {model_cfg['act_file']}. Skipping.")
        return None, 0, None

    is_real_list = []
    family_int_list = []
    activation_chunks = []
    for key in layer_keys:
        seq_name = key.replace(f"__layer{layer}", "")
        chunk = act_data[key]  # [seq_len, hidden], fp16
        n = chunk.shape[0]
        rfam = seq_meta.get(seq_name, {"rfam_id": None})["rfam_id"]
        fam_int = family_to_int.get(rfam, -1)
        for pos in range(n):
            is_special = pos == 0 or pos == n - 1
            is_real_list.append(not is_special)
            family_int_list.append(fam_int)
        activation_chunks.append(chunk)

    activations = torch.cat(activation_chunks, dim=0)
    n_tokens, d_input = activations.shape
    is_real = torch.tensor(is_real_list, dtype=torch.bool)
    token_family = torch.tensor(family_int_list, dtype=torch.long)
    del act_data, activation_chunks

    # Families this model actually observed on a real token.
    family_covered = torch.zeros(n_families, dtype=torch.bool)
    real_fams = token_family[is_real]
    real_fams = real_fams[real_fams >= 0].unique()
    family_covered[real_fams] = True

    print(f"  Loading SAE from {model_cfg['sae_file']}...")
    sae_state = load_file(str(model_cfg["sae_file"]))
    d_dict = sae_state["W_enc"].shape[1]
    sae_d_input = sae_state["W_enc"].shape[0]
    if sae_d_input != d_input:
        print(f"  ⚠ SAE input dim ({sae_d_input}) != activations dim ({d_input}). "
              f"Skipping {model_cfg['tag']}.")
        return None, 0, None

    sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=SAE_K)
    sae.load_state_dict(sae_state)
    sae.eval().to(device)

    max_act_per_ff = torch.zeros((d_dict, n_families), dtype=torch.float32)
    n_chunks = (n_tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    pbar = tqdm(range(n_chunks), desc=f"  {model_cfg['tag']}", unit="chunk", leave=False)
    with torch.no_grad():
        for ci in pbar:
            start = ci * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n_tokens)
            chunk = activations[start:end].float().to(device, non_blocking=True)
            chunk_feats = sae.encode(chunk).cpu()
            chunk_fams = token_family[start:end]
            chunk_is_real = is_real[start:end]
            for fam in chunk_fams.unique().tolist():
                if fam < 0:
                    continue
                mask = (chunk_fams == fam)
                if SKIP_SPECIAL_TOKENS:
                    mask = mask & chunk_is_real
                if mask.any():
                    fam_max = chunk_feats[mask].max(dim=0).values
                    max_act_per_ff[:, fam] = torch.maximum(max_act_per_ff[:, fam], fam_max)
            del chunk, chunk_feats

    max_per_feature = max_act_per_ff.max(dim=1).values
    threshold = max_per_feature * MAGNITUDE_THRESHOLD_FRAC
    presence = max_act_per_ff >= threshold.unsqueeze(1)  # [d_dict, n_families]
    # A feature with max 0 everywhere would mark all families "present" via 0>=0;
    # guard: dead features (max==0) present on nothing.
    presence[max_per_feature == 0] = False
    return presence, d_dict, family_covered


def jaccard_matrix(pres_a: torch.Tensor, pres_b: torch.Tensor) -> torch.Tensor:
    pa = pres_a.float()
    pb = pres_b.float()
    intersection = pa @ pb.T
    sum_a = pa.sum(dim=1, keepdim=True)
    sum_b = pb.sum(dim=1, keepdim=True)
    union = sum_a + sum_b.T - intersection
    return intersection / union.clamp(min=1)


def main() -> None:
    print("=" * 74)
    print("RIBOSCOPE: Cross-Model Feature Agreement (per-pair shared families)")
    print("=" * 74)

    if not VOCAB_FASTA.exists():
        print(f"❌ Vocabulary FASTA not found: {VOCAB_FASTA}")
        sys.exit(1)

    print(f"\n[1/3] Building unified Rfam vocabulary from {VOCAB_FASTA}...")
    seq_meta = parse_fasta_metadata(VOCAB_FASTA)
    all_rfams = sorted({m["rfam_id"] for m in seq_meta.values() if m["rfam_id"]})
    family_to_int = {f: i for i, f in enumerate(all_rfams)}
    n_families = len(all_rfams)
    print(f"      {len(seq_meta)} sequences, {n_families} distinct Rfam families")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")

    print(f"\n[2/3] Computing per-model family-presence matrices...")
    presences: dict[str, dict] = {}
    for cfg in MODELS:
        if not cfg["sae_file"].exists() or not cfg["act_file"].exists():
            print(f"\n  ⚠ {cfg['tag']}: missing files, skipping.")
            print(f"    SAE: {cfg['sae_file']}  exists={cfg['sae_file'].exists()}")
            print(f"    ACT: {cfg['act_file']}  exists={cfg['act_file'].exists()}")
            continue
        print(f"\n  Processing {cfg['tag']} ({cfg['label']})...")
        presence, d_dict, covered = compute_family_presence(
            cfg, seq_meta, family_to_int, n_families, device)
        if presence is None:
            continue
        n_alive = int((presence.sum(dim=1) >= MIN_FAMILIES_PER_FEATURE).sum().item())
        n_total_alive = int((presence.sum(dim=1) > 0).sum().item())
        n_cov = int(covered.sum().item())
        print(f"    d_dict={d_dict}, families_observed={n_cov}, "
              f"alive(≥1 fam)={n_total_alive}, alive(≥{MIN_FAMILIES_PER_FEATURE})={n_alive}")
        presences[cfg["tag"]] = {
            "presence": presence, "d_dict": d_dict,
            "label": cfg["label"], "layer": cfg["layer"], "covered": covered,
        }

    if len(presences) < 2:
        print("\n❌ Need at least 2 models to compare. Found: "
              f"{list(presences.keys())}")
        sys.exit(1)

    print(f"\n[3/3] Pairwise reciprocal best matches "
          f"(Jaccard ≥ {JACCARD_THRESHOLD}, SPECIFIC features only: "
          f"{MIN_FAMILIES_PER_FEATURE}–{MAX_FAMILIES_PER_FEATURE} shared families)...")
    report: dict = {
        "models": {tag: {"d_dict": p["d_dict"], "label": p["label"], "layer": p["layer"]}
                   for tag, p in presences.items()},
        "settings": {
            "magnitude_threshold_frac": MAGNITUDE_THRESHOLD_FRAC,
            "jaccard_threshold": JACCARD_THRESHOLD,
            "min_families_per_feature": MIN_FAMILIES_PER_FEATURE,
            "max_families_per_feature": MAX_FAMILIES_PER_FEATURE,
            "skip_special_tokens": SKIP_SPECIAL_TOKENS,
            "shared_family_restriction": True,
            "specific_features_only": True,
        },
        "pairs": [],
    }
    summary_counts: dict[tuple[str, str], int] = {}

    for tag_a, tag_b in combinations(presences.keys(), 2):
        la, lb = presences[tag_a]["label"], presences[tag_b]["label"]
        print(f"\n--- {la}  ↔  {lb} ---")

        # Restrict to families BOTH models actually observed.
        shared_mask = presences[tag_a]["covered"] & presences[tag_b]["covered"]
        shared_idx = shared_mask.nonzero(as_tuple=False).flatten().tolist()
        n_shared = len(shared_idx)
        if n_shared == 0:
            print("  No shared families — skipping pair.")
            report["pairs"].append({"model_a": tag_a, "model_b": tag_b,
                                    "n_shared_families": 0, "n_reciprocal_matches": 0,
                                    "matches": []})
            summary_counts[(tag_a, tag_b)] = 0
            continue

        pres_a = presences[tag_a]["presence"][:, shared_mask]
        pres_b = presences[tag_b]["presence"][:, shared_mask]
        print(f"  Shared families: {n_shared}")

        # "specific" = fires on ≥ MIN and ≤ MAX shared families. The ceiling
        # excludes non-discriminative generalists (see CONFIG note); without it
        # the count is dominated by trivial generalist↔generalist Jaccard≈1.0.
        fam_count_a = pres_a.sum(dim=1)
        fam_count_b = pres_b.sum(dim=1)
        alive_a = (fam_count_a >= MIN_FAMILIES_PER_FEATURE) & (fam_count_a <= MAX_FAMILIES_PER_FEATURE)
        alive_b = (fam_count_b >= MIN_FAMILIES_PER_FEATURE) & (fam_count_b <= MAX_FAMILIES_PER_FEATURE)
        print(f"  Specific features (≤{MAX_FAMILIES_PER_FEATURE} fams): "
              f"A={int(alive_a.sum())}  B={int(alive_b.sum())}")

        J = jaccard_matrix(pres_a, pres_b)
        J[~alive_a, :] = 0
        J[:, ~alive_b] = 0

        best_b_score, best_b_idx = J.max(dim=1)
        best_a_score, best_a_idx = J.max(dim=0)

        reciprocal: list[dict] = []
        for fa in range(pres_a.shape[0]):
            if not alive_a[fa]:
                continue
            fb = int(best_b_idx[fa].item())
            score = float(best_b_score[fa].item())
            if score < JACCARD_THRESHOLD:
                continue
            if int(best_a_idx[fb].item()) != fa:
                continue  # not reciprocal
            fams_a = sorted(all_rfams[shared_idx[i]] for i in pres_a[fa].nonzero().flatten().tolist())
            fams_b = sorted(all_rfams[shared_idx[i]] for i in pres_b[fb].nonzero().flatten().tolist())
            shared_fams = sorted(set(fams_a) & set(fams_b))
            reciprocal.append({
                "feat_a": int(fa), "feat_b": int(fb), "jaccard": score,
                "n_families_a": len(fams_a), "n_families_b": len(fams_b),
                "n_shared_families": len(shared_fams),
                "shared_families_sample": shared_fams[:10],
            })
        reciprocal.sort(key=lambda r: -r["jaccard"])

        n_match = len(reciprocal)
        summary_counts[(tag_a, tag_b)] = n_match
        print(f"  Reciprocal matches @ J ≥ {JACCARD_THRESHOLD}: {n_match}")
        if n_match > 0:
            print(f"  Top {min(TOP_MATCHES_TO_SHOW, n_match)} pairs:")
            print(f"  {'rk':>3}  {'feat_A':>7}  {'feat_B':>7}  {'jaccard':>8}  "
                  f"{'nfam_A':>6}  {'nfam_B':>6}  {'shared':>6}  shared families (sample)")
            for i, m in enumerate(reciprocal[:TOP_MATCHES_TO_SHOW], 1):
                ss = ", ".join(m["shared_families_sample"][:4])
                extra = " …" if m["n_shared_families"] > 4 else ""
                print(f"  {i:>3}  {m['feat_a']:>7}  {m['feat_b']:>7}  {m['jaccard']:>8.3f}  "
                      f"{m['n_families_a']:>6}  {m['n_families_b']:>6}  "
                      f"{m['n_shared_families']:>6}  {ss}{extra}")

        report["pairs"].append({
            "model_a": tag_a, "model_b": tag_b,
            "n_shared_families": n_shared,
            "n_reciprocal_matches": n_match,
            "matches": reciprocal[:200],
        })

    # ---- summary matrix + the headline contrast ----
    print("\n" + "=" * 74)
    print("SUMMARY — reciprocal-match counts per pair")
    print("=" * 74)
    for (ta, tb), n in summary_counts.items():
        print(f"  {presences[ta]['label']:<28}↔ {presences[tb]['label']:<28} {n:>4}")

    def count_for(a, b):
        return summary_counts.get((a, b), summary_counts.get((b, a)))

    if "rnamsm_msa" in presences and "rnamsm_single" in presences:
        print("\n  NEGATIVE-CONTROL CONTRAST (agreement with the two non-MSA models):")
        for other in ("rnafm", "erniarna"):
            if other in presences:
                msa = count_for(other, "rnamsm_msa")
                sng = count_for(other, "rnamsm_single")
                if msa is not None and sng is not None:
                    print(f"    {presences[other]['label']:<10}: MSA(in-dist)={msa}  "
                          f"vs  single(OOD)={sng}  →  "
                          f"{'MSA wins ✓' if msa > sng else 'no separation ✗'}")
        print("  Expectation for the headline: MSA ≫ single. If so, the cross-model")
        print("  panel independently confirms input-conditional interpretability.")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 74)
    print(f"✅ Cross-model agreement complete.  Full report: {REPORT_OUT}")
    print("=" * 74)
    print("Reading the result:")
    print("- Reciprocal match = feature X in A and feature Y in B fire on the same")
    print("  shared Rfam families (Jaccard ≥ 0.5) AND are each other's best match.")
    print("- High count between two GENUINE-biology models → robust, architecture-")
    print("  independent biological features (the RIBOSCOPE thesis).")
    print("- MSA RNA-MSM should match RNA-FM/ErnieRNA far more than single-seq")
    print("  RNA-MSM does — the negative control for the input-conditional claim.")


if __name__ == "__main__":
    main()
