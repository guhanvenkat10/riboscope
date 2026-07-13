"""
51_benchmark_localization.py — RIBOSCOPE G4 step 7: the MAKE-OR-BREAK benchmark.

Question
--------
Does the U4/RNU4-2 result generalize? I.e. across MANY disease structured ncRNAs
with known ClinVar pathogenic variants, does the model's unsupervised criticality
map localize the pathogenic positions better than chance — and does the
secondary-structure-aware model (ErnieRNA) beat the structure-naive control
(RNA-FM)? If yes -> we have a generalizable method and the broad disease SCREEN
is meaningful. If U4 is isolated -> honest single-case-study + methods finding.

Method (self-contained; reuses the validated readout from 47 + ClinVar from 49)
------------------------------------------------------------------------------
For each panel RNA, for each model:
  * representation-sensitivity in-silico mutagenesis (NO SAE needed): mutate every
    nucleotide to its 3 alternatives, measure the mean L2 shift of the layer-6
    embedding -> per-nucleotide criticality (exactly the readout that gave U4).
  * AUC = P(criticality_pathogenic > criticality_background) vs the gene's ClinVar
    pathogenic positions, with a Mann-Whitney p-value.
Report a per-RNA table + a summary: median AUC per model, how many RNAs are
individually significant, and how often ErnieRNA beats RNA-FM (paired).

RNA-MSM is intentionally EXCLUDED: its native-MSA embedding-sensitivity readout
is confounded by alignment-column conservation (it anti-correlates; see 50). A
fair MSA readout is future work.

Run with
--------
    cd ~/projects/riboscope
    uv run python 51_benchmark_localization.py
    uv run python 51_benchmark_localization.py --quick   # RNU4-2 + RNU4ATAC only

Inputs : none on disk — fetches sequences (NCBI efetch) + ClinVar live (WSL net).
Output : outputs/benchmark_localization.json
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
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

# ============================ CONFIG ============================
# panel: (gene, RefSeq accession, disease). All nuclear ncRNAs whose ClinVar
# variants are written on the NR_ RefSeq (n. numbering == our sequence). Verified
# accessions + lengths 2026-06-04. RPPH1 may have few variants (reported, not dropped).
PANEL = [
    ("RNU4-2",   "NR_003137.2", "ReNU neurodevelopmental syndrome"),       # known positive
    ("RMRP",     "NR_003051.4", "Cartilage-hair hypoplasia"),              # known ~chance
    ("RNU4ATAC", "NR_023343.1", "MOPD1 / Roifman (minor spliceosome)"),
    ("RNU12",    "NR_029422.1", "Early-onset cerebellar ataxia (minor spliceosome)"),
    ("SNORD118", "NR_033294.1", "Leukoencephalopathy w/ calcifications & cysts (Labrune)"),
    ("TERC",     "NR_001566.1", "Dyskeratosis congenita"),
    ("RPPH1",    "NR_002312.1", "RNase P RNA"),
]
MIN_PATH = 4              # need >=4 ClinVar pathogenic positions for a meaningful AUC
FIRING_BASES = ("A", "C", "G", "U")

MODELS = {
    "rnafm":    {"cls": RnaFmModel,    "name": "multimolecule/rnafm",    "layer": 6},
    "erniarna": {"cls": ErnieRnaModel, "name": "multimolecule/ernierna", "layer": 6},
}

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
from entrez_config import get_entrez_email
TOOL, EMAIL = "riboscope", get_entrez_email()
TIMEOUT, N_RETRIES = 30, 3
# ================================================================


def _get(url: str) -> str:
    last = None
    for attempt in range(1, N_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * attempt)
    print(f"   ⚠ fetch failed: {last}")
    return ""


def efetch_seq(accession: str) -> str:
    txt = _get(f"{EUTILS}/efetch.fcgi?db=nuccore&id={accession}&rettype=fasta&retmode=text"
               f"&tool={TOOL}&email={EMAIL}")
    if not txt.startswith(">"):
        return ""
    seq = "".join(l.strip() for l in txt.splitlines()[1:] if l and not l.startswith(">"))
    return "".join(c if c in "ACGU" else "N" for c in seq.upper().replace("T", "U"))


def fetch_clinvar_positions(gene: str, acc_prefix: str) -> list[int]:
    term = urllib.parse.quote(f'{gene}[gene] AND ("pathogenic"[Germline classification] '
                              f'OR "likely pathogenic"[Germline classification])')
    es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={term}&retmax=500&retmode=json"
              f"&tool={TOOL}&email={EMAIL}")
    try:
        ids = json.loads(es)["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        ids = []
    if not ids:
        es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={urllib.parse.quote(gene + '[gene]')}"
                  f"&retmax=500&retmode=json&tool={TOOL}&email={EMAIL}")
        try:
            ids = json.loads(es)["esearchresult"]["idlist"]
        except Exception:  # noqa: BLE001
            return []
    pos: set[int] = set()
    for k in range(0, len(ids), 100):
        time.sleep(0.34)
        summ = _get(f"{EUTILS}/esummary.fcgi?db=clinvar&id={','.join(ids[k:k+100])}"
                    f"&retmode=json&tool={TOOL}&email={EMAIL}")
        try:
            res = json.loads(summ)["result"]
        except Exception:  # noqa: BLE001
            continue
        for uid in res.get("uids", []):
            rec = res.get(uid, {})
            title = rec.get("title", "") or ""
            cls = (rec.get("germline_classification", {}) or {}).get("description", "") \
                or (rec.get("clinical_significance", {}) or {}).get("description", "")
            if "pathogenic" not in cls.lower() or acc_prefix not in title:
                continue
            for m in re.finditer(r"n\.(\d+)", title):
                pos.add(int(m.group(1)))
    return sorted(pos)


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
        captured[layer_idx] = output.detach().to(torch.float32).cpu()
    return hook


def load_model(key, device):
    cfg = MODELS[key]
    tok = RnaTokenizer.from_pretrained(cfg["name"])
    if key == "erniarna":
        conf = AutoConfig.from_pretrained(cfg["name"])
        n_pad = int(getattr(conf, "vocab_size", len(tok))) - len(tok)
        if n_pad > 0:
            tok.add_tokens([f"<unused{i}>" for i in range(n_pad)], special_tokens=True)
        model = cfg["cls"].from_pretrained(cfg["name"], tokenizer=tok)
        try:
            ckpt = load_file(hf_hub_download(cfg["name"], "model.safetensors"))
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
                print(f"      ErnieRNA: remapped {len(to_load)} pairwise tensors.")
            else:
                print("      ⚠ ErnieRNA: no pairwise remap — features unreliable.")
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ ErnieRNA remap failed: {e}")
    else:
        model = cfg["cls"].from_pretrained(cfg["name"])
    return tok, model.eval().to(device)


def real_embed(seq, tok, model, layer, captured, device):
    inputs = tok(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    _ = model(**inputs)
    act = captured[layer][0].float().to(device)
    ntok = act.shape[0]
    return act[1:ntok - 1]                       # [L, hidden]


def criticality_map(seq, tok, model, layer, captured, device):
    """Representation-sensitivity ISM (no SAE): per-nt mean L2 embedding shift."""
    wt = real_embed(seq, tok, model, layer, captured, device)
    L = len(seq)
    crit = [0.0] * L
    for i in range(L):
        wt_b = seq[i]
        alts = [b for b in FIRING_BASES if b != wt_b]
        acc = 0.0
        for b in alts:
            mut = seq[:i] + b + seq[i + 1:]
            em = real_embed(mut, tok, model, layer, captured, device)
            if em.shape == wt.shape:
                acc += float((em - wt).norm(dim=1).mean().item())
        crit[i] = acc / len(alts)
    return crit


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc_enrichment(scores, pos1):
    L = len(scores)
    P = sorted({p - 1 for p in pos1 if 1 <= p <= L})
    n1, n2 = len(P), L - len(P)
    if n1 == 0 or n2 == 0:
        return None
    ranks = rankdata(scores)
    R1 = sum(ranks[i] for i in P)
    U1 = R1 - n1 * (n1 + 1) / 2
    auc = U1 / (n1 * n2)
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (L + 1) / 12)
    z = (U1 - mu) / sd if sd > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"auc": round(auc, 3), "p_value": float(f"{p:.2e}"), "n_path": n1, "L": L}


def main():
    quick = "--quick" in sys.argv
    panel = PANEL[:1] + [p for p in PANEL if p[0] == "RNU4ATAC"] if quick else PANEL
    print("=" * 84)
    print(f"RIBOSCOPE G4 BENCHMARK — does disease-nt localization generalize? (n={len(panel)} RNAs)")
    print("=" * 84)

    # 1) gather sequences + ClinVar positions
    data = {}
    for gene, acc, disease in panel:
        seq = efetch_seq(acc)
        if not seq:
            print(f"  {gene:<9} ❌ sequence fetch failed ({acc}) — skipping.")
            continue
        cv = fetch_clinvar_positions(gene, acc.split(".")[0])
        flag = "" if len(cv) >= MIN_PATH else f"  (only {len(cv)} ClinVar — below MIN_PATH, AUC skipped)"
        print(f"  {gene:<9} L={len(seq):<4} ClinVar pathogenic n={len(cv)}{flag}")
        data[gene] = {"accession": acc, "disease": disease, "seq": seq, "clinvar": cv}
        time.sleep(0.3)

    usable = [g for g in data if len(data[g]["clinvar"]) >= MIN_PATH]
    if not usable:
        print("❌ No RNAs with enough ClinVar variants. Aborting.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) per model: load once, compute criticality for each usable RNA
    results = {g: {"disease": data[g]["disease"], "L": len(data[g]["seq"]),
                   "n_path": len(data[g]["clinvar"]), "models": {}} for g in usable}
    for key in MODELS:
        print(f"\n[{key}] loading + scoring {len(usable)} RNAs ...")
        tok, model = load_model(key, device)
        layers = find_encoder_layers(model)
        captured = {}
        h = layers[MODELS[key]["layer"]].register_forward_hook(make_hook(captured, MODELS[key]["layer"]))
        with torch.no_grad():
            for g in tqdm(usable, desc=key, unit="rna"):
                try:
                    crit = criticality_map(data[g]["seq"], tok, model, MODELS[key]["layer"], captured, device)
                    results[g]["models"][key] = auc_enrichment(crit, data[g]["clinvar"])
                except Exception as e:  # noqa: BLE001
                    print(f"   ⚠ {key}/{g} failed: {type(e).__name__}: {str(e)[:80]}")
                    results[g]["models"][key] = None
        h.remove()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3) report
    print("\n" + "=" * 84)
    print(f"{'RNA':<10}{'disease':<34}{'n_path':>7}{'RNA-FM':>9}{'ErnieRNA':>10}")
    print("-" * 84)
    ernie_aucs, rnafm_aucs, ernie_wins, ernie_sig = [], [], 0, 0
    for g in usable:
        rf = results[g]["models"].get("rnafm")
        er = results[g]["models"].get("erniarna")
        rfa = rf["auc"] if rf else float("nan")
        era = er["auc"] if er else float("nan")
        star = "*" if (er and er["p_value"] < 0.05) else " "
        win = "✓" if (er and rf and er["auc"] > rf["auc"]) else " "
        print(f"{g:<10}{results[g]['disease'][:33]:<34}{results[g]['n_path']:>7}"
              f"{rfa:>9.3f}{era:>9.3f}{star}{win}")
        if er:
            ernie_aucs.append(er["auc"]); ernie_sig += int(er["p_value"] < 0.05)
        if rf:
            rnafm_aucs.append(rf["auc"])
        if er and rf and er["auc"] > rf["auc"]:
            ernie_wins += 1
    print("-" * 84)

    def med(xs):
        s = sorted(xs); n = len(s)
        return float("nan") if n == 0 else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

    n = len(usable)
    print(f"  median AUC — RNA-FM {med(rnafm_aucs):.3f} | ErnieRNA {med(ernie_aucs):.3f}")
    print(f"  ErnieRNA individually significant (p<0.05): {ernie_sig}/{n}")
    print(f"  ErnieRNA AUC > RNA-FM AUC: {ernie_wins}/{n} RNAs   (* = ErnieRNA p<0.05)")
    print("\n  VERDICT (heuristic):")
    if med(ernie_aucs) >= 0.6 and ernie_sig >= max(2, n // 2):
        print("    → GENERALIZES. ErnieRNA localizes pathogenic nt across the panel.")
        print("      The broad disease SCREEN is justified.")
    elif med(ernie_aucs) > 0.55 and ernie_wins >= (2 * n) // 3:
        print("    → PARTIAL. A real but modest/architecture-specific effect; tighten the")
        print("      panel (e.g. spliceosomal RNAs) and consider the screen on that subclass.")
    else:
        print("    → DOES NOT GENERALIZE on this panel. Honest outcome = U4 case study +")
        print("      the methods/architecture finding; a broad screen is NOT yet justified.")
    print("=" * 84)

    Path("outputs/benchmark_localization.json").write_text(json.dumps(
        {"panel": [{"gene": g, **{k: v for k, v in results[g].items() if k != "seq"}} for g in usable],
         "median_auc": {"rnafm": med(rnafm_aucs), "erniarna": med(ernie_aucs)},
         "ernie_significant": ernie_sig, "ernie_wins": ernie_wins, "n": n}, indent=2))
    print("✅ Saved outputs/benchmark_localization.json")


if __name__ == "__main__":
    main()
