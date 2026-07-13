"""
55_spliceosome_panel.py — RIBOSCOPE G4 Phase 2b: does it generalize across the
spliceosomal snRNA disease class?

For each spliceosomal snRNA gene, this:
  1. RESOLVES the human RefSeq ncRNA accession from the gene symbol (NCBI esearch
     in WSL — no hand-typed accessions), fetches the sequence.
  2. Runs ErnieRNA + RNA-FM representation-sensitivity ISM (same readout as 51/53/54).
  3. Fetches ClinVar pathogenic AND benign positions.
  4. Scores localization two ways:
       AUC_bg     = pathogenic vs the rest of the molecule (always computable)
       AUC_triage = pathogenic-position vs benign-position (the clinical task; when
                    >=5 benign positions exist)
Reports a per-gene table + an aggregate, to turn "works on RNU4-2" into "works
across the spliceosomal snRNA subclass" (or honestly bound where it doesn't).

Run with
--------
    cd ~/projects/riboscope
    uv run python 55_spliceosome_panel.py

Inputs : none on disk (NCBI live). Output : outputs/spliceosome_panel.json
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

# spliceosomal snRNA genes (major + minor spliceosome), disease-linked where known
GENES = ["RNU4-2", "RNU4ATAC", "RNU12", "RNU2-2", "RNU5A-1", "RNU5B-1",
         "RNU11", "RNU1-1", "RNU6-1", "RNU6ATAC"]
MIN_PATH = 4
MIN_BENIGN = 5
MAX_LEN = 520
BASES = ("A", "C", "G", "U")
MODELS = {"rnafm": {"cls": RnaFmModel, "name": "multimolecule/rnafm", "layer": 6},
          "erniarna": {"cls": ErnieRnaModel, "name": "multimolecule/ernierna", "layer": 6}}
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
from entrez_config import get_entrez_email
TOOL, EMAIL = "riboscope", get_entrez_email()


def _get(u):
    for attempt in range(3):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": f"{TOOL} (mailto:{EMAIL})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            time.sleep(1.2 * (attempt + 1))
    return ""


def resolve_refseq(gene):
    """gene symbol -> (accession, sequence) via NCBI RefSeq ncRNA; verify gene in header."""
    term = urllib.parse.quote(f'{gene}[gene] AND srcdb_refseq[prop] AND biomol_ncRNA[prop] '
                              f'AND "Homo sapiens"[orgn]')
    try:
        ids = json.loads(_get(f"{EUTILS}/esearch.fcgi?db=nuccore&term={term}&retmax=8&retmode=json"
                              f"&tool={TOOL}&email={EMAIL}"))["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        ids = []
    g = gene.upper().replace("-", "")
    for uid in ids:
        time.sleep(0.34)
        fa = _get(f"{EUTILS}/efetch.fcgi?db=nuccore&id={uid}&rettype=fasta&retmode=text"
                  f"&tool={TOOL}&email={EMAIL}")
        if not fa.startswith(">"):
            continue
        header = fa.splitlines()[0]
        seq = "".join(l.strip() for l in fa.splitlines()[1:] if l and not l.startswith(">"))
        seq = "".join(c if c in "ACGU" else "N" for c in seq.upper().replace("T", "U"))
        hdr_norm = header.upper().replace("-", "").replace(" ", "")
        if g in hdr_norm and 40 <= len(seq) <= MAX_LEN:
            return header[1:].split()[0], seq
    return None, None


def clinvar_positions(gene, classification):
    term = urllib.parse.quote(f'{gene}[gene] AND ({classification}[Germline classification])')
    try:
        ids = json.loads(_get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={term}&retmax=500&retmode=json"
                              f"&tool={TOOL}&email={EMAIL}"))["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        return set()
    pos = set()
    for k in range(0, len(ids), 100):
        time.sleep(0.34)
        try:
            res = json.loads(_get(f"{EUTILS}/esummary.fcgi?db=clinvar&id={','.join(ids[k:k+100])}"
                                  f"&retmode=json&tool={TOOL}&email={EMAIL}"))["result"]
        except Exception:  # noqa: BLE001
            continue
        for uid in res.get("uids", []):
            t = res.get(uid, {}).get("title", "") or ""
            for m in re.finditer(r"n\.(\d+)", t):
                pos.add(int(m.group(1)))
    return pos


def find_encoder_layers(model):
    for p in ("encoder.layer", "bert.encoder.layer", "roberta.encoder.layer",
              "ernie.encoder.layer", "model.encoder.layer"):
        obj = model
        try:
            for part in p.split("."):
                obj = getattr(obj, part)
            _ = len(obj); return obj
        except (AttributeError, TypeError):
            continue
    raise AttributeError("encoder.layer not found")


def make_hook(cap, li):
    def hook(m, i, o):
        cap[li] = (o[0] if isinstance(o, tuple) else o).detach().to(torch.float32).cpu()
    return hook


def load_model(key, device):
    cfg = MODELS[key]; tok = RnaTokenizer.from_pretrained(cfg["name"])
    if key == "erniarna":
        conf = AutoConfig.from_pretrained(cfg["name"])
        npad = int(getattr(conf, "vocab_size", len(tok))) - len(tok)
        if npad > 0:
            tok.add_tokens([f"<unused{i}>" for i in range(npad)], special_tokens=True)
        model = cfg["cls"].from_pretrained(cfg["name"], tokenizer=tok)
        try:
            ckpt = load_file(hf_hub_download(cfg["name"], "model.safetensors")); sd = model.state_dict(); tl = {}
            for kk0, v in ckpt.items():
                kk = (kk0[6:] if kk0.startswith("model.") else kk0)
                kk = kk.replace("pairwise_bias_proj.dense1", "pairwise_bias_proj.0").replace("pairwise_bias_proj.dense2", "pairwise_bias_proj.2")
                if "pairwise_bias_proj" in kk and kk in sd and tuple(sd[kk].shape) == tuple(v.shape):
                    tl[kk] = v
            if tl:
                model.load_state_dict(tl, strict=False); print(f"      ErnieRNA: remapped {len(tl)} pairwise tensors.")
        except Exception as e:  # noqa: BLE001
            print(f"      ⚠ ErnieRNA remap failed: {e}")
    else:
        model = cfg["cls"].from_pretrained(cfg["name"])
    return tok, model.eval().to(device)


def criticality_map(seq, tok, model, layer, cap, device):
    def emb(s):
        inp = tok(s, return_tensors="pt"); inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            _ = model(**inp)
        a = cap[layer][0].float().to(device); return a[1:a.shape[0] - 1]
    wt = emb(seq); L = len(seq); crit = [0.0] * L
    for i in range(L):
        acc = 0.0; alts = [b for b in BASES if b != seq[i]]
        for b in alts:
            e = emb(seq[:i] + b + seq[i + 1:])
            if e.shape == wt.shape:
                acc += float((e - wt).norm(dim=1).mean().item())
        crit[i] = acc / len(alts)
    return crit


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        a = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = a
        i = j + 1
    return r


def auc(scores, pos_idx):
    L = len(scores); P = set(pos_idx); n1 = len(P); n2 = L - n1
    if n1 == 0 or n2 == 0:
        return None
    r = rankdata(scores); U = sum(r[i] for i in P) - n1 * (n1 + 1) / 2
    return U / (n1 * n2)


def auc_two(score_path, score_benign):
    xs = score_path + score_benign; labels = [1] * len(score_path) + [0] * len(score_benign)
    idx = [i for i, l in enumerate(labels) if l == 1]
    return auc(xs, idx)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 84)
    print("RIBOSCOPE G4 Phase 2b — spliceosomal snRNA panel (does it generalize?)")
    print("=" * 84)

    # 1) resolve + fetch sequences and ClinVar labels
    genes = {}
    for g in GENES:
        acc, seq = resolve_refseq(g)
        if not seq:
            print(f"  {g:<10} ❌ could not resolve RefSeq ncRNA — skipping.")
            continue
        path = clinvar_positions(g, "pathogenic")
        ben = clinvar_positions(g, "benign")
        path = {p for p in path if 1 <= p <= len(seq)}
        ben = {p for p in ben if 1 <= p <= len(seq)} - path
        print(f"  {g:<10} {acc:<14} L={len(seq):<4} ClinVar path={len(path):<3} benign={len(ben)}")
        if len(path) >= MIN_PATH:
            genes[g] = {"acc": acc, "seq": seq, "path": sorted(path), "benign": sorted(ben)}
        time.sleep(0.3)

    if not genes:
        print("❌ No genes with enough ClinVar pathogenic positions. Aborting.")
        return

    # 2) per model: criticality maps for each gene
    results = {g: {"acc": genes[g]["acc"], "L": len(genes[g]["seq"]),
                   "n_path": len(genes[g]["path"]), "n_benign": len(genes[g]["benign"]),
                   "models": {}} for g in genes}
    for key in MODELS:
        print(f"\n[{key}] scoring {len(genes)} genes ...")
        tok, model = load_model(key, device)
        layers = find_encoder_layers(model); cap = {}
        h = layers[MODELS[key]["layer"]].register_forward_hook(make_hook(cap, MODELS[key]["layer"]))
        for g in tqdm(list(genes), desc=key, unit="gene"):
            try:
                crit = criticality_map(genes[g]["seq"], tok, model, MODELS[key]["layer"], cap, device)
                L = len(crit)
                path_idx = [p - 1 for p in genes[g]["path"] if 1 <= p <= L]
                a_bg = auc(crit, path_idx)
                a_tri = None
                if len(genes[g]["benign"]) >= MIN_BENIGN:
                    sp = [crit[p - 1] for p in genes[g]["path"] if 1 <= p <= L]
                    sb = [crit[p - 1] for p in genes[g]["benign"] if 1 <= p <= L]
                    a_tri = auc_two(sp, sb)
                results[g]["models"][key] = {"auc_bg": round(a_bg, 3) if a_bg is not None else None,
                                             "auc_triage": round(a_tri, 3) if a_tri is not None else None}
            except Exception as e:  # noqa: BLE001
                print(f"   ⚠ {key}/{g} failed: {type(e).__name__}: {str(e)[:70]}")
                results[g]["models"][key] = None
        h.remove(); del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3) report
    print("\n" + "=" * 84)
    print(f"{'gene':<10}{'n_path':>7}{'n_ben':>6}{'RNA-FM_bg':>11}{'Ernie_bg':>10}{'Ernie_triage':>14}")
    print("-" * 84)
    e_bg, f_bg, e_tri, wins = [], [], [], 0
    for g in genes:
        m = results[g]["models"]
        er = m.get("erniarna") or {}; rf = m.get("rnafm") or {}
        eb, fb, et = er.get("auc_bg"), rf.get("auc_bg"), er.get("auc_triage")
        print(f"{g:<10}{results[g]['n_path']:>7}{results[g]['n_benign']:>6}"
              f"{(fb if fb is not None else float('nan')):>11.3f}{(eb if eb is not None else float('nan')):>10.3f}"
              f"{(et if et is not None else float('nan')):>14.3f}")
        if eb is not None:
            e_bg.append(eb)
        if fb is not None:
            f_bg.append(fb)
        if et is not None:
            e_tri.append(et)
        if eb is not None and fb is not None and eb > fb:
            wins += 1
    print("-" * 84)

    def med(x):
        s = sorted(x); n = len(s)
        return float("nan") if not s else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)
    npair = sum(1 for g in genes if (results[g]["models"].get("erniarna") or {}).get("auc_bg") is not None
                and (results[g]["models"].get("rnafm") or {}).get("auc_bg") is not None)
    print(f"  genes scored: {len(e_bg)}   ErnieRNA>RNA-FM (localization): {wins}/{npair}")
    print(f"  median ErnieRNA AUC_bg {med(e_bg):.3f} vs RNA-FM {med(f_bg):.3f}")
    if e_tri:
        print(f"  median ErnieRNA AUC_triage (pathogenic vs benign): {med(e_tri):.3f}  (n_genes={len(e_tri)})")
    print("=" * 84)
    Path("outputs/spliceosome_panel.json").write_text(json.dumps(
        {"genes": results, "median_ernie_bg": med(e_bg), "median_rnafm_bg": med(f_bg),
         "median_ernie_triage": med(e_tri), "ernie_wins": wins, "n_pairs": npair}, indent=2))
    print("✅ Saved outputs/spliceosome_panel.json")


if __name__ == "__main__":
    main()
