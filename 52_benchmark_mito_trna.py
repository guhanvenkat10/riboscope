"""
52_benchmark_mito_trna.py — RIBOSCOPE G4 step 8: widen the benchmark with
mitochondrial tRNAs, then produce the COMBINED (nuclear + mito) verdict.

Why mito-tRNAs
--------------
They're the ideal stress-test for the "structure-aware models localize disease
nucleotides" claim: each is a small, strongly-structured (cloverleaf) RNA with
many catalogued pathogenic variants (MELAS, MERRF, deafness, cardiomyopathy...).
Adding ~10 of them takes the panel from "2 of 6 significant" to a properly
powered aggregate test, and tells us whether the effect extends from spliceosomal
snRNAs to structured tRNAs (which RNA classes a screen should target).

Coordinate handling (with a built-in safety net)
------------------------------------------------
Mito tRNAs live on the mt genome (NC_012920.1); ClinVar uses m.<pos> numbering.
We use HEAVY-strand tRNAs only (so the reference sequence read 5'->3' in
increasing coordinate = the tRNA), and map local = m_pos - start + 1. CRUCIAL
SELF-CHECK: we parse each ClinVar substitution's reference base (m.3243**A**>G)
and confirm it matches our sequence at the mapped position. If the ref-match rate
is low, the coordinates/strand are wrong for that tRNA and it is SKIPPED — so a
bad coordinate can never silently corrupt the result.

Readout = identical to 51 (representation-sensitivity ISM, ErnieRNA vs RNA-FM).

Run with
--------
    cd ~/projects/riboscope
    uv run python 52_benchmark_mito_trna.py

Inputs : none on disk (efetch + ClinVar live); merges outputs/benchmark_localization.json (51) if present.
Output : outputs/benchmark_mito.json  +  outputs/benchmark_combined.json
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
MT_ACC = "NC_012920.1"
# HEAVY-strand mito tRNAs: (gene, start, end, disease). rCRS coordinates; the
# ref-match self-check below validates each one, so a wrong coord just gets skipped.
PANEL_MITO = [
    ("MT-TF",  577,   647,   "MELAS/epilepsy (tRNA-Phe)"),
    ("MT-TV",  1602,  1670,  "MELAS/leigh (tRNA-Val)"),
    ("MT-TL1", 3230,  3304,  "MELAS (tRNA-Leu UUR)"),
    ("MT-TI",  4263,  4331,  "cardiomyopathy (tRNA-Ile)"),
    ("MT-TM",  4402,  4469,  "(tRNA-Met)"),
    ("MT-TW",  5512,  5579,  "Leigh/encephalopathy (tRNA-Trp)"),
    ("MT-TD",  7518,  7585,  "(tRNA-Asp)"),
    ("MT-TK",  8295,  8364,  "MERRF (tRNA-Lys)"),
    ("MT-TG",  9991,  10058, "(tRNA-Gly)"),
    ("MT-TR",  10405, 10469, "(tRNA-Arg)"),
    ("MT-TH",  12138, 12206, "MERRF-like (tRNA-His)"),
    ("MT-TS2", 12207, 12265, "(tRNA-Ser AGY)"),
    ("MT-TL2", 12266, 12336, "CPEO/cardiomyopathy (tRNA-Leu CUN)"),
    ("MT-TT",  15888, 15953, "(tRNA-Thr)"),
]
MIN_PATH = 4
REF_MATCH_MIN = 0.7          # min fraction of ClinVar subs whose ref base matches our seq
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


def efetch_region(acc, start, end) -> str:
    txt = _get(f"{EUTILS}/efetch.fcgi?db=nuccore&id={acc}&rettype=fasta&retmode=text"
               f"&seq_start={start}&seq_stop={end}&tool={TOOL}&email={EMAIL}")
    if not txt.startswith(">"):
        return ""
    seq = "".join(l.strip() for l in txt.splitlines()[1:] if l and not l.startswith(">"))
    return "".join(c if c in "ACGU" else "N" for c in seq.upper().replace("T", "U"))


def fetch_clinvar_mito(gene: str):
    """Return (set of m. positions, list of (m_pos, ref_base)) for pathogenic variants."""
    term = urllib.parse.quote(f'{gene}[gene] AND ("pathogenic"[Germline classification] '
                              f'OR "likely pathogenic"[Germline classification])')
    used_filter = True
    es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={term}&retmax=500&retmode=json"
              f"&tool={TOOL}&email={EMAIL}")
    try:
        ids = json.loads(es)["esearchresult"]["idlist"]
    except Exception:  # noqa: BLE001
        ids = []
    if not ids:
        used_filter = False
        es = _get(f"{EUTILS}/esearch.fcgi?db=clinvar&term={urllib.parse.quote(gene + '[gene]')}"
                  f"&retmax=500&retmode=json&tool={TOOL}&email={EMAIL}")
        try:
            ids = json.loads(es)["esearchresult"]["idlist"]
        except Exception:  # noqa: BLE001
            return set(), []
    positions, subs = set(), []
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
            # When the esearch classification filter was applied, the returned IDs
            # are already pathogenic/likely-pathogenic — trust it (mito records
            # populate the classification field inconsistently). Only re-check when
            # we had to fall back to the unfiltered gene search.
            if not used_filter:
                cls = (rec.get("germline_classification", {}) or {}).get("description", "") \
                    or (rec.get("clinical_significance", {}) or {}).get("description", "")
                if "pathogenic" not in cls.lower():
                    continue
            for m in re.finditer(r"m\.(\d+)", title):
                positions.add(int(m.group(1)))
            for m in re.finditer(r"m\.(\d+)([ACGT])>([ACGT])", title):
                subs.append((int(m.group(1)), m.group(2)))
    return positions, subs


# ---- model + ISM (identical readout to 51) ----
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
        alts = [b for b in FIRING_BASES if b != seq[i]]
        acc = 0.0
        for b in alts:
            em = real_embed(seq[:i] + b + seq[i + 1:], tok, model, layer, captured, device)
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
    U1 = sum(ranks[i] for i in P) - n1 * (n1 + 1) / 2
    auc = U1 / (n1 * n2)
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (L + 1) / 12)
    z = (U1 - mu) / sd if sd > 0 else 0.0
    return {"auc": round(auc, 3), "p_value": float(f"{math.erfc(abs(z)/math.sqrt(2)):.2e}"), "n_path": n1, "L": L}


def med(xs):
    s = sorted(xs); n = len(s)
    return float("nan") if n == 0 else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def binom_tail(k, n, p=0.5):
    """P(X >= k) for X~Binom(n,p) — one-sided sign-test p-value."""
    from math import comb
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def main():
    print("=" * 84)
    print(f"RIBOSCOPE G4 BENCHMARK — mito-tRNA expansion ({len(PANEL_MITO)} tRNAs)")
    print("=" * 84)

    data = {}
    for gene, start, end, disease in PANEL_MITO:
        seq = efetch_region(MT_ACC, start, end)
        if not seq:
            print(f"  {gene:<8} ❌ region fetch failed — skipping.")
            continue
        L = len(seq)
        mpos, subs = fetch_clinvar_mito(gene)
        local = sorted({mp - start + 1 for mp in mpos if 1 <= mp - start + 1 <= L})
        # SELF-CHECK: ClinVar ref base must match our sequence at the mapped position
        checked = matched = 0
        for mp, ref in subs:
            loc = mp - start + 1
            if 1 <= loc <= L:
                checked += 1
                refU = ref.replace("T", "U")
                matched += int(seq[loc - 1] == refU)
        rate = matched / checked if checked else 0.0
        ok = rate >= REF_MATCH_MIN and len(local) >= MIN_PATH
        tag = "" if ok else f"  ⚠ SKIP (ref-match {rate:.0%}, n_path {len(local)})"
        print(f"  {gene:<8} L={L:<3} m-pos n={len(local):<3} ref-match={rate:.0%} ({matched}/{checked}){tag}")
        if ok:
            data[gene] = {"disease": disease, "seq": seq, "local": local,
                          "n_path": len(local), "ref_match": round(rate, 3), "L": L}
        time.sleep(0.3)

    if not data:
        print("❌ No mito-tRNAs passed the self-check + ClinVar threshold.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usable = list(data)

    mito_results = {g: {"disease": data[g]["disease"], "L": data[g]["L"],
                        "n_path": data[g]["n_path"], "ref_match": data[g]["ref_match"],
                        "models": {}} for g in usable}
    for key in MODELS:
        print(f"\n[{key}] scoring {len(usable)} mito-tRNAs ...")
        tok, model = load_model(key, device)
        layers = find_encoder_layers(model)
        captured = {}
        h = layers[MODELS[key]["layer"]].register_forward_hook(make_hook(captured, MODELS[key]["layer"]))
        with torch.no_grad():
            for g in tqdm(usable, desc=key, unit="trna"):
                try:
                    crit = criticality_map(data[g]["seq"], tok, model, MODELS[key]["layer"], captured, device)
                    mito_results[g]["models"][key] = auc_enrichment(crit, data[g]["local"])
                except Exception as e:  # noqa: BLE001
                    print(f"   ⚠ {key}/{g} failed: {type(e).__name__}: {str(e)[:80]}")
                    mito_results[g]["models"][key] = None
        h.remove()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- merge with nuclear panel (51) for the COMBINED verdict ----
    combined = {}
    nuc_path = Path("outputs/benchmark_localization.json")
    if nuc_path.exists():
        for ent in json.loads(nuc_path.read_text()).get("panel", []):
            combined[ent["gene"]] = {"class": "nuclear", "disease": ent.get("disease", ""),
                                     "models": ent.get("models", {}), "n_path": ent.get("n_path")}
    for g in usable:
        combined[g] = {"class": "mito", "disease": mito_results[g]["disease"],
                       "models": mito_results[g]["models"], "n_path": mito_results[g]["n_path"]}

    # ---- report ----
    print("\n" + "=" * 84)
    print(f"{'RNA':<10}{'class':<8}{'disease':<30}{'n':>4}{'RNA-FM':>9}{'ErnieRNA':>10}")
    print("-" * 84)
    e_aucs, r_aucs, wins, sig = [], [], 0, 0
    for g, ent in sorted(combined.items(), key=lambda kv: kv[1]["class"]):
        rf = ent["models"].get("rnafm"); er = ent["models"].get("erniarna")
        if not er:
            continue
        rfa = rf["auc"] if rf else float("nan")
        star = "*" if er["p_value"] < 0.05 else " "
        win = "✓" if (rf and er["auc"] > rf["auc"]) else " "
        print(f"{g:<10}{ent['class']:<8}{(ent['disease'] or '')[:29]:<30}{ent.get('n_path',0):>4}"
              f"{rfa:>9.3f}{er['auc']:>9.3f}{star}{win}")
        e_aucs.append(er["auc"]); sig += int(er["p_value"] < 0.05)
        if rf:
            r_aucs.append(rf["auc"]); wins += int(er["auc"] > rf["auc"])
    print("-" * 84)
    n = len(e_aucs)
    npair = len(r_aucs)
    sign_p = binom_tail(wins, npair) if npair else 1.0
    print(f"  COMBINED panel: {n} RNAs ({sum(1 for e in combined.values() if e['class']=='mito')} mito + "
          f"{sum(1 for e in combined.values() if e['class']=='nuclear')} nuclear)")
    print(f"  median AUC — RNA-FM {med(r_aucs):.3f} | ErnieRNA {med(e_aucs):.3f}")
    print(f"  ErnieRNA individually significant (p<0.05): {sig}/{n}")
    print(f"  ErnieRNA > RNA-FM: {wins}/{npair}  (sign-test one-sided p={sign_p:.1e})")
    print("\n  VERDICT:")
    if med(e_aucs) >= 0.6 and sign_p < 0.05 and sig >= max(3, n // 3):
        print("    → GENERALIZES across structured RNAs. Screen is justified (scope to")
        print("      structured snRNAs + tRNAs where the effect is validated).")
    elif sign_p < 0.05 or sig >= max(3, n // 4):
        print("    → ROBUST-BUT-MODEST architecture effect (structure-aware > naive, broadly).")
        print("      Defensible as a prioritization/mechanism method; screen the strongest subclass.")
    else:
        print("    → Effect is narrow (mainly spliceosomal). Honest = case studies + methods finding.")
    print("=" * 84)

    Path("outputs/benchmark_mito.json").write_text(json.dumps(
        {"mito": [{"gene": g, **{k: v for k, v in mito_results[g].items()}} for g in usable]}, indent=2))
    Path("outputs/benchmark_combined.json").write_text(json.dumps(
        {"n": n, "median_auc": {"rnafm": med(r_aucs), "erniarna": med(e_aucs)},
         "ernie_significant": sig, "ernie_wins": wins, "n_pairs": npair,
         "sign_test_p_one_sided": sign_p,
         "panel": [{"gene": g, **ent} for g, ent in combined.items()]}, indent=2))
    print("✅ Saved outputs/benchmark_mito.json + outputs/benchmark_combined.json")


if __name__ == "__main__":
    main()
