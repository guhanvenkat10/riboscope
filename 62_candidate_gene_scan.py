"""
62_candidate_gene_scan.py — RIBOSCOPE: hunt for a SECOND positive gene class.

Why
---
The locked result is that ErnieRNA's criticality localizes ClinVar-pathogenic
positions specifically in U4-type snRNAs (RNU4-2, RNU4ATAC), through the
inter-RNA base-pairing a secondary-structure model represents. The Discussion
names the other spliceosomal snRNA families — U5, U6, U1, U11 — as the natural
place a second positive could appear, several of which gained de novo
neurodevelopmental disorder links in 2024–2025 (e.g. RNU5B-1 / RNU5A-1).

This script runs the EXACT localization benchmark from 51 (representation-
sensitivity ISM, ErnieRNA vs the structure-naive RNA-FM control, AUC vs live
ClinVar pathogenic positions) on a candidate panel of those snRNAs. It is a
targeted search: a genuine hit is a spliceosomal snRNA where ErnieRNA localizes
pathogenic nt (AUC > ~0.6, p < 0.05) AND beats RNA-FM. There is no guarantee the
biology cooperates; a null result is an honest outcome that tightens the U4-type
claim rather than weakening it.

Self-contained: fetches sequences + ClinVar live (needs network). Accessions are
auto-resolved from the gene symbol via NCBI esearch, so a stale hint accession is
not fatal.

Run with
--------
    cd ~/projects/riboscope
    uv run python 62_candidate_gene_scan.py
    uv run python 62_candidate_gene_scan.py RNU5B-1 RNU5A-1   # subset

Inputs : none on disk — NCBI efetch + ClinVar over the network.
Output : outputs/candidate_gene_scan.json
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
# candidate snRNAs: (gene symbol, hint RefSeq accession, disorder / rationale)
# hint accession is only a fallback; the gene symbol is auto-resolved on NCBI.
PANEL = [
    ("RNU5B-1", "NR_002757.1", "NDD — de novo U5 snRNA variants (2024)"),
    ("RNU5A-1", "NR_002756.2", "NDD — de novo U5 snRNA variants (2024)"),
    ("RNU2-2",  "NR_199791.1", "NDD — minor/ U2-type (2024)"),
    ("RNU1-2",  "NR_004430.1", "U1 snRNA (spliceosome 5' site)"),
    ("RNU11",   "NR_004407.1", "U11 minor-spliceosome snRNA"),
    ("RNU6-1",  "NR_004394.1", "U6 snRNA (pairs with U4)"),
    ("RNU4-1",  "NR_003925.1", "U4 snRNA paralog — U4-type positive control"),
]
MIN_PATH = 4
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


def resolve_refseq(gene: str, hint: str) -> str:
    """Return a curated RefSeq NR_ accession for the gene, preferring the hint if
    it still resolves, else the top ncRNA RefSeq for the gene symbol."""
    term = urllib.parse.quote(f'{gene}[gene] AND refseq[filter] AND biomol_ncrna[prop]')
    es = _get(f"{EUTILS}/esearch.fcgi?db=nuccore&term={term}&retmax=10&retmode=json"
              f"&tool={TOOL}&email={EMAIL}")
    accs = []
    try:
        ids = json.loads(es)["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        ids = []
    if ids:
        summ = _get(f"{EUTILS}/esummary.fcgi?db=nuccore&id={','.join(ids)}&retmode=json"
                    f"&tool={TOOL}&email={EMAIL}")
        try:
            res = json.loads(summ)["result"]
            for uid in res.get("uids", []):
                acc = res.get(uid, {}).get("accessionversion", "")
                if acc.startswith("NR_"):
                    accs.append(acc)
        except Exception:  # noqa: BLE001
            pass
    hint_prefix = hint.split(".")[0]
    for a in accs:                                  # prefer the hint if still valid
        if a.split(".")[0] == hint_prefix:
            return a
    return accs[0] if accs else hint


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
    return act[1:ntok - 1]


def criticality_map(seq, tok, model, layer, captured, device):
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
    want = [a for a in sys.argv[1:] if not a.startswith("-")]
    panel = [p for p in PANEL if p[0] in want] if want else PANEL
    print("=" * 84)
    print(f"RIBOSCOPE — candidate second-positive scan (spliceosomal snRNAs), n={len(panel)}")
    print("=" * 84)

    data = {}
    for gene, hint, disease in panel:
        acc = resolve_refseq(gene, hint)
        seq = efetch_seq(acc)
        if not seq:
            print(f"  {gene:<9} ❌ sequence fetch failed (acc={acc}) — skipping.")
            continue
        cv = fetch_clinvar_positions(gene, acc.split(".")[0])
        flag = "" if len(cv) >= MIN_PATH else f"  (only {len(cv)} ClinVar — below MIN_PATH, AUC skipped)"
        print(f"  {gene:<9} acc={acc:<13} L={len(seq):<4} ClinVar pathogenic n={len(cv)}{flag}")
        data[gene] = {"accession": acc, "disease": disease, "seq": seq, "clinvar": cv}
        time.sleep(0.3)

    usable = [g for g in data if len(data[g]["clinvar"]) >= MIN_PATH]
    if not usable:
        print("\n❌ No candidate RNAs reached MIN_PATH ClinVar pathogenic positions.")
        print("   (Honest null: these snRNAs have too few curated pathogenic nt yet.)")
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/candidate_gene_scan.json").write_text(json.dumps(
            {"panel": [{"gene": g, **{k: v for k, v in data[g].items() if k != "seq"}}
                       for g in data], "usable": []}, indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {g: {"disease": data[g]["disease"], "accession": data[g]["accession"],
                   "L": len(data[g]["seq"]), "n_path": len(data[g]["clinvar"]),
                   "clinvar_positions": data[g]["clinvar"], "models": {}} for g in usable}
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

    print("\n" + "=" * 84)
    print(f"{'RNA':<10}{'disorder':<40}{'n_path':>7}{'RNA-FM':>9}{'ErnieRNA':>10}")
    print("-" * 84)
    hits = []
    for g in usable:
        rf = results[g]["models"].get("rnafm")
        er = results[g]["models"].get("erniarna")
        rfa = rf["auc"] if rf else float("nan")
        era = er["auc"] if er else float("nan")
        star = "*" if (er and er["p_value"] < 0.05) else " "
        win = "✓" if (er and rf and er["auc"] > rf["auc"]) else " "
        print(f"{g:<10}{results[g]['disease'][:39]:<40}{results[g]['n_path']:>7}"
              f"{rfa:>9.3f}{era:>9.3f}{star}{win}")
        if er and rf and er["auc"] > rf["auc"] and er["auc"] >= 0.6 and er["p_value"] < 0.05:
            hits.append(g)
    print("-" * 84)
    if hits:
        print(f"  🎯 CANDIDATE SECOND POSITIVE(S): {hits}")
        print("     ErnieRNA localizes (AUC≥0.6, p<0.05) AND beats RNA-FM. Verify the")
        print("     structure/duplex mechanism before claiming; add to Fig 5 / Table 1.")
    else:
        print("  → No new positive on this panel. Honest outcome: the effect stays")
        print("     specific to the U4-type snRNAs; this tightens rather than weakens it.")
    print("=" * 84)

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/candidate_gene_scan.json").write_text(json.dumps(
        {"panel": [{"gene": g, **{k: v for k, v in results[g].items()}} for g in usable],
         "candidate_hits": hits}, indent=2))
    print("✅ Saved outputs/candidate_gene_scan.json")


if __name__ == "__main__":
    main()
