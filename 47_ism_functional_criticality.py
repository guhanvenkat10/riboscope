"""
47_ism_functional_criticality.py — RIBOSCOPE G4 step 3: the centerpiece.

Idea
----
Turn the locked, interpretable SAE features into a CAUSAL probe of which
nucleotides a disease RNA's function depends on — without any supervision.

For an RNA the model RECOGNIZES (step 46: a family specialist/moderate feature
fires on it), we do in-silico saturation mutagenesis: mutate every position to
each of the 3 alternative bases, re-run the model+SAE, and measure how much the
recognized family-feature(s) DROP. A position whose mutation collapses the
family-identity feature is one the model considers defining/critical — an
unsupervised "functional-criticality" score per nucleotide.

We compute TWO complementary maps:
  (A) FEATURE criticality  — mean normalized drop of the firing readout feature(s)
                             when a position is mutated (interpretable; family-specific).
  (B) REPRESENTATION sensitivity — mean L2 change of the layer-6 residue embeddings
                             (model-intrinsic; label-free cross-check).

Validation (step 48 reads this): does the unsupervised map peak at KNOWN
disease-variant hotspots? For RNU4-2 (ReNU syndrome) the hotspot is the
T-loop + Stem III region (~nt 53-76; recurrent n.64_65insT). We report a simple,
honest enrichment of criticality inside vs outside that region.

Scope
-----
RNA-FM and ErnieRNA (single-seq in-distribution). RNA-MSM is handled later with
native per-RNA MSAs. By default we ISM only the RNAs that the given model
recognizes (a feature fires); pass --all to force all, --rna GENE to pick one.

Run with
--------
    cd ~/projects/riboscope
    uv run python 47_ism_functional_criticality.py rnafm
    uv run python 47_ism_functional_criticality.py erniarna
    uv run python 47_ism_functional_criticality.py rnafm --rna RNU4-2

Inputs : sequences/disease_structured_rna.fasta, outputs/disease_recon_{model}.json,
         outputs/disease_rna_hotspots.json, locked SAE + inspection_{...}_v3.json
Output : outputs/ism_criticality_{model}.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    from multimolecule import RnaTokenizer, RnaFmModel, ErnieRnaModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE

# ============================ CONFIG ============================
FIRING_FRAC = 0.25
MAX_BREADTH = 9
BASES = ("A", "C", "G", "U")
FASTA_FILE = Path("sequences/disease_structured_rna.fasta")
HOTSPOTS_FILE = Path("outputs/disease_rna_hotspots.json")
TOP_FRAC = 0.15          # "top criticality" = top 15% of positions (for enrichment)

MODELS = {
    "rnafm": {
        "model_class": RnaFmModel, "model_name": "multimolecule/rnafm", "layer": 6,
        "sae_file": Path("outputs/sae_big_layer6_v3.safetensors"),
        "inspection_file": Path("outputs/inspection_big_layer6_v3.json"),
        "recon_file": Path("outputs/disease_recon_rnafm.json"),
    },
    "erniarna": {
        "model_class": ErnieRnaModel, "model_name": "multimolecule/ernierna", "layer": 6,
        "sae_file": Path("outputs/sae_erniarna_layer6_v3.safetensors"),
        "inspection_file": Path("outputs/inspection_erniarna_layer6_v3.json"),
        "recon_file": Path("outputs/disease_recon_erniarna.json"),
    },
}
# ================================================================


def parse_fasta(path: Path) -> list[dict]:
    out, gene, acc, desc, buf = [], None, None, None, []

    def flush():
        if gene is not None:
            out.append({"gene": gene, "accession": acc, "description": desc,
                        "sequence": "".join(buf).upper().replace("T", "U")})

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


def make_hook(captured: dict, layer_idx: int):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured[layer_idx] = output.detach().to(torch.float16).cpu()
    return hook


def load_model(cfg: dict, key: str, device):
    name = cfg["model_name"]
    tok = RnaTokenizer.from_pretrained(name)
    ModelCls = cfg["model_class"]
    if key == "erniarna":
        conf = AutoConfig.from_pretrained(name)
        want = int(getattr(conf, "vocab_size", len(tok)))
        n_pad = want - len(tok)
        if n_pad > 0:
            tok.add_tokens([f"<unused{i}>" for i in range(n_pad)], special_tokens=True)
            print(f"      ErnieRNA: padded tokenizer to {len(tok)} to match config vocab.")
        model = ModelCls.from_pretrained(name, tokenizer=tok)
        try:
            ckpt = load_file(hf_hub_download(name, "model.safetensors"))
            sd = model.state_dict()
            to_load = {}
            for kk0, v in ckpt.items():
                kk = kk0[6:] if kk0.startswith("model.") else kk0
                kk = kk.replace("pairwise_bias_proj.dense1", "pairwise_bias_proj.0")
                kk = kk.replace("pairwise_bias_proj.dense2", "pairwise_bias_proj.2")
                if "pairwise_bias_proj" in kk and kk in sd and tuple(sd[kk].shape) == tuple(v.shape):
                    to_load[kk] = v
            if to_load:
                model.load_state_dict(to_load, strict=False)
                print(f"      ErnieRNA: remapped {len(to_load)} pairwise_bias_proj tensors.")
            else:
                print("      ⚠ ErnieRNA: no pairwise weights remapped — STOP and report.")
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ ErnieRNA pairwise remap failed: {e}")
    else:
        model = ModelCls.from_pretrained(name)
    return tok, model.eval().to(device)


def load_feature_table(inspection_file: Path) -> dict[int, dict]:
    with open(inspection_file) as f:
        insp = json.load(f)
    table: dict[int, dict] = {}
    for bucket in ("specialist_features", "moderate_features"):
        for rec in insp.get(bucket, []):
            fi = int(rec["feature_idx"])
            if fi in table:
                continue
            mx = float(rec["max_activation"])
            br = int(rec.get("n_families", 0))
            if mx > 0 and br <= MAX_BREADTH:
                table[fi] = {"max_act": mx, "breadth": br,
                             "families": list(rec.get("families_sample", []))}
    return table


def forward_feats(seq, tok, model, sae, layer, captured, device, feat_idx_tensor):
    """Return (peak[F] over residues, real_embed[R,hidden] on device) for one sequence."""
    inputs = tok(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    _ = model(**inputs)
    act = captured[layer][0].float().to(device)        # [ntok, hidden]
    ntok = act.shape[0]
    real = act[1:ntok - 1]                              # [R, hidden]
    feats = sae.encode(real)                            # [R, d_dict]
    peak = feats.index_select(1, feat_idx_tensor).max(dim=0).values  # [F]
    return peak, real


def region_enrichment(scores, region_pos1, L, top_frac):
    """Mean score inside vs outside an annotated 1-based region + top-position hit rate."""
    if not region_pos1 or len(region_pos1) != 2:
        return None
    lo, hi = max(0, region_pos1[0] - 1), min(L, region_pos1[1])   # 0-based [lo,hi)
    if hi <= lo:
        return None
    inreg = [scores[j] for j in range(lo, hi)]
    outreg = [scores[j] for j in range(L) if not (lo <= j < hi)]
    if not inreg or not outreg:
        return None
    mi, mo = sum(inreg) / len(inreg), sum(outreg) / len(outreg)
    order = sorted(range(L), key=lambda j: scores[j], reverse=True)
    top_k = max(1, int(round(top_frac * L)))
    in_top = sum(1 for j in order[:top_k] if lo <= j < hi)
    return {
        "region_pos1": [region_pos1[0], region_pos1[1]],
        "mean_in_region": round(mi, 5),
        "mean_out_region": round(mo, 5),
        "enrichment_ratio": round(mi / mo, 3) if mo > 0 else None,
        "region_frac_of_len": round((hi - lo) / L, 3),
        "top_positions_in_region_frac": round(in_top / top_k, 3),
    }


def variant_ranks(scores, recur_pos1, L):
    """Percentile of named recurrent-variant positions in the score distribution."""
    if not recur_pos1:
        return None
    order = sorted(range(L), key=lambda j: scores[j], reverse=True)
    out = {}
    for p in recur_pos1:
        if 1 <= p <= L:
            r = order.index(p - 1) + 1
            out[str(p)] = {"rank_of_L": r, "percentile": round(1 - (r - 1) / L, 3)}
    return out or None


def main() -> None:
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if len(pos_args) < 1 or pos_args[0] not in MODELS:
        print(f"Usage: python 47_ism_functional_criticality.py [{' | '.join(MODELS)}] "
              f"[--all] [--rna GENE]")
        sys.exit(1)
    key = pos_args[0]
    cfg = MODELS[key]
    layer = cfg["layer"]
    force_all = "--all" in flags
    pick = None
    for f in flags:
        if f.startswith("--rna"):
            pick = f.split("=", 1)[1] if "=" in f else (pos_args[1] if len(pos_args) > 1 else None)
    # allow "--rna RNU4-2" (space-separated)
    if "--rna" in sys.argv:
        i = sys.argv.index("--rna")
        if i + 1 < len(sys.argv):
            pick = sys.argv[i + 1]

    print("=" * 80)
    print(f"RIBOSCOPE G4 ISM functional-criticality — model={key} layer={layer}")
    print("=" * 80)
    for p in (FASTA_FILE, cfg["sae_file"], cfg["inspection_file"], cfg["recon_file"]):
        if not Path(p).exists():
            print(f"❌ Missing required file: {p}")
            sys.exit(1)

    table = load_feature_table(cfg["inspection_file"])
    feat_ids = sorted(table)
    col_of = {fi: c for c, fi in enumerate(feat_ids)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_idx_tensor = torch.tensor(feat_ids, dtype=torch.long, device=device)
    feat_max = torch.tensor([table[fi]["max_act"] for fi in feat_ids],
                            dtype=torch.float32, device=device)
    thresh = FIRING_FRAC * feat_max
    breadth = {fi: table[fi]["breadth"] for fi in feat_ids}

    sae_state = load_file(str(cfg["sae_file"]))
    d_input, d_dict = sae_state["W_enc"].shape[0], sae_state["W_enc"].shape[1]
    sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=32)
    sae.load_state_dict(sae_state)
    sae = sae.eval().to(device)

    print(f"[load] {len(feat_ids)} family features; SAE d_dict={d_dict}; device={device}")
    tok, model = load_model(cfg, key, device)
    layers = find_encoder_layers(model)
    captured: dict = {}
    handle = layers[layer].register_forward_hook(make_hook(captured, layer))

    recon = json.loads(Path(cfg["recon_file"]).read_text())["rnas"]
    hotspots = json.loads(HOTSPOTS_FILE.read_text()).get("genes", {}) if HOTSPOTS_FILE.exists() else {}
    rnas = parse_fasta(FASTA_FILE)
    if pick:
        rnas = [r for r in rnas if r["gene"] == pick]
        if not rnas:
            print(f"❌ --rna {pick} not found in FASTA.")
            sys.exit(1)

    results: dict[str, dict] = {}
    with torch.no_grad():
        for rna in rnas:
            gene, seq = rna["gene"], rna["sequence"]
            L = len(seq)
            rinfo = recon.get(gene, {})
            recognized = rinfo.get("verdict", "").startswith("RECOGNIZED")
            has_any = bool(rinfo.get("hits"))
            if not (recognized or force_all or pick):
                print(f"  {gene:<8} skipped (not recognized; use --all to force).")
                continue
            if not has_any and not force_all and not pick:
                print(f"  {gene:<8} skipped (no firing feature to read out).")
                continue

            # WT pass → identify readout features (those that fire on WT)
            peak_wt, real_wt = forward_feats(seq, tok, model, sae, layer, captured,
                                             device, feat_idx_tensor)
            fired = (peak_wt >= thresh).nonzero(as_tuple=False).flatten().tolist()
            fired_ids = [feat_ids[c] for c in fired]
            spec = [c for c in fired if breadth[feat_ids[c]] <= 2]
            readout_cols = spec if spec else fired      # prefer specialists
            has_feature = bool(readout_cols)
            readout_ids = [feat_ids[c] for c in readout_cols]
            # The representation-sensitivity (embedding) map is ALWAYS computed: it
            # needs no SAE feature, so it is the FAIR cross-architecture readout even
            # when a model does not recognize the RNA (e.g. ErnieRNA on U4). The
            # feature-criticality map is the bonus interpretable layer when a family
            # feature fires.
            if has_feature:
                peak_wt_read = peak_wt[readout_cols].clamp(min=1e-6)  # [Fr]
            else:
                print(f"  {gene:<8} no feature fires → embedding-sensitivity map only.")

            crit_feat = [0.0] * L if has_feature else None
            crit_embed = [0.0] * L
            real_wt_dev = real_wt.to(device)

            for i in tqdm(range(L), desc=f"{gene} ISM", unit="pos", leave=False):
                wt_b = seq[i]
                alts = [b for b in BASES if b != wt_b]
                drop_acc, embed_acc = 0.0, 0.0
                for b in alts:
                    mseq = seq[:i] + b + seq[i + 1:]
                    peak_m, real_m = forward_feats(mseq, tok, model, sae, layer, captured,
                                                   device, feat_idx_tensor)
                    rm = real_m.to(device)
                    if rm.shape == real_wt_dev.shape:
                        embed_acc += float((rm - real_wt_dev).norm(dim=1).mean().item())
                    if has_feature:
                        pm = peak_m[readout_cols]
                        drop = torch.relu(peak_wt_read - pm) / peak_wt_read
                        drop_acc += float(drop.mean().item())
                if has_feature:
                    crit_feat[i] = drop_acc / len(alts)
                crit_embed[i] = embed_acc / len(alts)

            # ---- summary stats ---- (primary = feature map if present, else embedding)
            primary = crit_feat if crit_feat is not None else crit_embed
            order = sorted(range(L), key=lambda j: primary[j], reverse=True)
            top_k = max(1, int(round(TOP_FRAC * L)))
            top_positions = sorted(order[:top_k])
            if crit_feat is not None:
                mx = max(crit_feat) or 1.0
                crit_feat_norm = [round(c / mx, 4) for c in crit_feat]
            else:
                crit_feat_norm = None
            mxe = max(crit_embed) or 1.0
            crit_embed_norm = [round(c / mxe, 4) for c in crit_embed]

            # ---- hotspot validation for BOTH maps (feature + representation) ----
            hs = hotspots.get(gene, {})
            region = hs.get("critical_region_pos1_approx")
            recur = hs.get("recurrent_pos1")
            enr_feat = region_enrichment(crit_feat, region, L, TOP_FRAC) if crit_feat is not None else None
            enr_embed = region_enrichment(crit_embed, region, L, TOP_FRAC)
            rank_feat = variant_ranks(crit_feat, recur, L) if crit_feat is not None else None
            rank_embed = variant_ranks(crit_embed, recur, L)

            results[gene] = {
                "length": L,
                "verdict": rinfo.get("verdict", ""),
                "has_feature_readout": has_feature,
                "readout_features": readout_ids,
                "readout_is_specialist": bool(spec),
                "fired_features_all": fired_ids,
                "primary_map": "feature" if crit_feat is not None else "embedding",
                "top_positions_pos1": [p + 1 for p in top_positions],
                "criticality_feature": crit_feat_norm,
                "criticality_embedding": crit_embed_norm,
                "hotspot_enrichment_feature": enr_feat,
                "hotspot_enrichment_embedding": enr_embed,
                "recurrent_variant_rank_feature": rank_feat,
                "recurrent_variant_rank_embedding": rank_embed,
            }

            # ---- console ----
            spec_tag = ("specialist" if spec else "moderate") if has_feature else "NO feature (embed-only)"
            print(f"\n  {gene:<8} len={L}  readout={readout_ids or '—'} ({spec_tag})")
            pmap = "feature" if crit_feat is not None else "embedding"
            print(f"     top-criticality nt ({pmap} map, 1-based): {[p + 1 for p in order[:8]]}")
            for label, enr in (("feature", enr_feat), ("embed  ", enr_embed)):
                if enr:
                    print(f"     [{label}] hotspot {enr['region_pos1']} enrichment = {enr['enrichment_ratio']} "
                          f"(region={enr['region_frac_of_len']*100:.0f}% of len; "
                          f"top-{int(TOP_FRAC*100)}% hit-rate in region={enr['top_positions_in_region_frac']*100:.0f}%)")
            for label, rk in (("feature", rank_feat), ("embed  ", rank_embed)):
                if rk:
                    pct = ", ".join(f"nt{p}={d['percentile']*100:.0f}%" for p, d in rk.items())
                    print(f"     [{label}] recurrent-variant percentile: {pct}")

    handle.remove()

    out_file = Path(f"outputs/ism_criticality_{key}.json")
    with open(out_file, "w") as f:
        json.dump({"model": key, "layer": layer, "firing_frac": FIRING_FRAC,
                   "top_frac": TOP_FRAC, "rnas": results}, f, indent=2)
    print("\n" + "=" * 80)
    print(f"✅ ISM complete → {out_file}")
    print("   criticality_feature[i] = mean normalized drop of the family feature when nt i is mutated.")
    print("   Step 48 reads this for the cross-model consensus + the disease-hotspot validation.")
    print("=" * 80)


if __name__ == "__main__":
    main()
