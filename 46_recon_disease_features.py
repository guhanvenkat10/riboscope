"""
46_recon_disease_features.py — RIBOSCOPE G4 step 2: the RECOGNITION GATE.

Question
--------
Before we can build an in-silico-mutagenesis functional-criticality map of a
disease RNA, the model must actually RECOGNIZE that RNA — i.e. some interpretable
family-specialist (or moderate) feature must FIRE on it. If nothing fires, the
RNA is out-of-distribution and an ISM readout through these features would be
meaningless (respects our locked headline: interpretability is input-conditional).

So this script "fingerprints" each disease structured ncRNA: it runs the full
(in-distribution, <=510 nt) sequence through the model + locked SAE and reports
WHICH named specialist/moderate features fire, WHAT Rfam family they stand for,
at WHAT position, and how strongly. Output is the gate + the readout-feature
list that step-46/47's ISM will perturb.

"Fires" = activation >= FIRING_FRAC * (feature's own Rfam max), identical to
09_inspect_features_big.py and 21_scan_disease_rna.py.

Scope
-----
RNA-FM and ErnieRNA (both single-sequence in-distribution). RNA-MSM is handled
later with native per-RNA MSAs (single-seq is OOD for it — our locked result).

Run with
--------
    cd ~/projects/riboscope
    uv run python 46_recon_disease_features.py rnafm
    uv run python 46_recon_disease_features.py erniarna

Inputs : sequences/disease_structured_rna.fasta (from 45),
         outputs/sae_{...}_layer6_v3.safetensors + inspection_{...}_v3.json
Output : outputs/disease_recon_{model}.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
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
FIRING_FRAC = 0.25       # == MAGNITUDE_THRESHOLD_FRAC in 09; "fires" definition
CONTEXT_WINDOW = 7
FASTA_FILE = Path("sequences/disease_structured_rna.fasta")
MAX_BREADTH = 9          # only family-meaningful features (specialist<=2, moderate3-9)

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
# ================================================================


def parse_fasta(path: Path) -> list[dict]:
    """Parse >GENE|accession|description FASTA. T->U, uppercase."""
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
    """Load model. ErnieRNA needs the pairwise_bias_proj remap (version drift) so
    its structure-attention is restored rather than randomly initialized."""
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
                print(f"      ErnieRNA: remapped {len(to_load)} pairwise_bias_proj tensors (structure-attention restored).")
            else:
                print("      ⚠ ErnieRNA: no pairwise weights remapped — features unreliable; STOP and report.")
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ ErnieRNA pairwise remap failed: {e}")
    else:
        model = ModelCls.from_pretrained(name)
    return tok, model.eval().to(device)


def load_feature_table(inspection_file: Path) -> list[dict]:
    """Specialist + moderate buckets only (generalists fire everywhere)."""
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
    return [v for v in table.values() if v["max_act"] > 0 and v["breadth"] <= MAX_BREADTH]


def format_context(seq: str, pos: int, win: int) -> str:
    if pos < 0 or pos >= len(seq):
        return "[OUT-OF-RANGE]"
    lo, hi = max(0, pos - win), min(len(seq), pos + win + 1)
    pre, focal, post = seq[lo:pos], seq[pos], seq[pos + 1:hi]
    return f"{pre}[{focal}]{post}"


def main() -> None:
    pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos_args) < 1 or pos_args[0] not in MODELS:
        print(f"Usage: python 46_recon_disease_features.py [{' | '.join(MODELS)}]")
        sys.exit(1)
    key = pos_args[0]
    cfg = MODELS[key]
    layer = cfg["layer"]

    print("=" * 80)
    print(f"RIBOSCOPE G4 recognition gate — model={key} layer={layer}")
    print("=" * 80)

    for p in (FASTA_FILE, cfg["sae_file"], cfg["inspection_file"]):
        if not Path(p).exists():
            print(f"❌ Missing required file: {p}")
            if p == FASTA_FILE:
                print("   Run 45_fetch_disease_structured_rna.py first.")
            sys.exit(1)

    feat_table = load_feature_table(cfg["inspection_file"])
    n_spec = sum(1 for r in feat_table if r["breadth"] <= 2)
    n_mod = sum(1 for r in feat_table if r["breadth"] >= 3)
    print(f"[1/3] Feature table: {len(feat_table)} family features "
          f"({n_spec} specialist <=2 fam, {n_mod} moderate 3-9 fam)")
    if not feat_table:
        print("❌ ABORT: empty feature table — refusing to report a false null.")
        print(f"   Regenerate {cfg['inspection_file']} via set_model.py {key} + 09.")
        sys.exit(1)

    sae_state = load_file(str(cfg["sae_file"]))
    d_input, d_dict = sae_state["W_enc"].shape[0], sae_state["W_enc"].shape[1]
    sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=32)
    sae.load_state_dict(sae_state)
    sae.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(device)
    print(f"[2/3] SAE: d_input={d_input}, d_dict={d_dict}, device={device}")

    feat_idx_list = [r["feat_idx"] for r in feat_table]
    feat_idx_tensor = torch.tensor(feat_idx_list, dtype=torch.long, device=device)
    feat_max = torch.tensor([r["max_act"] for r in feat_table], dtype=torch.float32, device=device)
    thresh = FIRING_FRAC * feat_max
    breadth_list = [r["breadth"] for r in feat_table]
    fams_list = [r["families"] for r in feat_table]

    print(f"[3/3] Loading {cfg['model_name']} ...")
    tok, model = load_model(cfg, key, device)
    layers = find_encoder_layers(model)
    if d_input != getattr(model.config, "hidden_size", d_input):
        print(f"   ⚠ d_input {d_input} != model hidden_size {model.config.hidden_size}")
    captured: dict = {}
    handle = layers[layer].register_forward_hook(make_hook(captured, layer))

    rnas = parse_fasta(FASTA_FILE)
    print(f"\nFingerprinting {len(rnas)} disease RNA(s) (full sequence, in-distribution)\n")

    results: dict[str, dict] = {}
    with torch.no_grad():
        for rna in rnas:
            gene, seq = rna["gene"], rna["sequence"]
            L = len(seq)
            inputs = tok(seq, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
            act = captured[layer][0].float().to(device)   # [ntok, hidden]
            ntok = act.shape[0]
            if ntok <= 2:
                print(f"  {gene:<8} len={L:<4} ⚠ too short after tokenization; skipping.")
                continue
            real = act[1:ntok - 1]                          # residue tokens [R, hidden]
            R = real.shape[0]
            feats = sae.encode(real)                        # [R, d_dict]
            sub = feats.index_select(1, feat_idx_tensor)    # [R, F]
            peak = sub.max(dim=0).values                    # [F]
            peak_pos = sub.argmax(dim=0)                     # [F]
            fired_mask = (peak >= thresh)                   # [F]
            fired_idx = fired_mask.nonzero(as_tuple=False).flatten().tolist()

            hits = []
            for fcol in fired_idx:
                a = float(peak[fcol].item())
                fm = float(feat_max[fcol].item())
                p0 = int(peak_pos[fcol].item())             # 0-based residue index
                hits.append({
                    "feat_idx": feat_idx_list[fcol],
                    "breadth": breadth_list[fcol],
                    "families": fams_list[fcol],
                    "peak_activation": round(a, 4),
                    "feat_max": round(fm, 4),
                    "peak_frac_of_max": round(a / fm, 4) if fm > 0 else 0.0,
                    "nt_pos1": p0 + 1,                       # 1-based on the RNA
                    "context": format_context(seq, p0, CONTEXT_WINDOW),
                })
            hits.sort(key=lambda h: h["peak_frac_of_max"], reverse=True)

            spec_hits = [h for h in hits if h["breadth"] <= 2]
            # family -> best peak frac (specialist-level claims only)
            fam_best: dict[str, float] = defaultdict(float)
            for h in spec_hits:
                for fam in h["families"]:
                    fam_best[fam] = max(fam_best[fam], h["peak_frac_of_max"])
            fam_ranked = dict(sorted(fam_best.items(), key=lambda kv: kv[1], reverse=True))

            if spec_hits:
                verdict = "RECOGNIZED (specialist fires)"
            elif hits:
                verdict = "weak (moderate-only)"
            else:
                verdict = "NOT recognized"

            results[gene] = {
                "accession": rna["accession"],
                "length": L,
                "n_residues_scored": R,
                "verdict": verdict,
                "n_specialist_hits": len(spec_hits),
                "n_moderate_hits": len(hits) - len(spec_hits),
                "specialist_families_ranked": fam_ranked,
                "hits": hits[:40],
            }

            print(f"  {gene:<8} len={L:<4} → {verdict}")
            if spec_hits:
                top = list(fam_ranked.items())[:6]
                print("      specialist families (peak frac | best nt pos):")
                for fam, frac in top:
                    # find the position of the best specialist hit for this family
                    best_h = max((h for h in spec_hits if fam in h["families"]),
                                 key=lambda h: h["peak_frac_of_max"])
                    print(f"        {fam:<10} {frac:.2f}  @nt {best_h['nt_pos1']:<4} {best_h['context']}")
            elif hits:
                h = hits[0]
                print(f"      top moderate feature {h['feat_idx']} {h['families']} "
                      f"frac={h['peak_frac_of_max']:.2f} @nt {h['nt_pos1']}")

    handle.remove()

    out_file = Path(f"outputs/disease_recon_{key}.json")
    with open(out_file, "w") as f:
        json.dump({
            "model": key, "layer": layer, "firing_frac": FIRING_FRAC,
            "sae_file": str(cfg["sae_file"]), "inspection_file": str(cfg["inspection_file"]),
            "n_features": len(feat_table), "n_specialists": n_spec, "n_moderate": n_mod,
            "rnas": results,
        }, f, indent=2)

    n_reco = sum(1 for r in results.values() if r["verdict"].startswith("RECOGNIZED"))
    print("\n" + "=" * 80)
    print(f"✅ Recon complete → {out_file}")
    print(f"   {n_reco}/{len(results)} RNA(s) RECOGNIZED (a specialist feature fires).")
    print("   Recognized RNAs + their firing features = the ISM readout for step 47.")
    print("   (If a target is NOT recognized, it is OOD — drop it or build an MSA for RNA-MSM.)")
    print("=" * 80)


if __name__ == "__main__":
    main()
