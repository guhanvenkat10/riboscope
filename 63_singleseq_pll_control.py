"""
63_singleseq_pll_control.py — RIBOSCOPE: the control that makes the reframe hold.

Why
---
61 showed RNA-MSM's masked-language pseudo-likelihood (PLL) localizes RNU4-2 /
RNU4ATAC pathogenic nt (AUC 0.76 / 0.86), rescuing it from the embedding-shift
inversion. The reframed thesis is: disease-critical nt are legible in
representations with structural OR evolutionary context, but NOT in single-
sequence representation. A reviewer's first objection: maybe RNA-FM (single
sequence) also localizes once read by PLL, and single-sequence is not actually
deficient — we just used the wrong readout for it.

This script closes that hole: it computes the SAME masked-PLL readout on the
single-sequence models RNA-FM and ERNIE-RNA for RNU4-2 and RNU4ATAC, completing
the architecture x readout matrix. Prediction for the reframe to hold: RNA-FM PLL
stays near chance (single-sequence context can't constrain these positions),
while ERNIE PLL may localize (it has structural context).

Self-contained: fetches sequences + ClinVar live (needs network). No MSA / MAFFT.

Run with
--------
    cd ~/projects/riboscope
    uv run python 63_singleseq_pll_control.py

Inputs : none on disk — NCBI efetch + ClinVar over the network.
Output : outputs/singleseq_pll_control.json
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
    import multimolecule
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    from multimolecule import RnaTokenizer, RnaFmModel, ErnieRnaModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

# ============================ CONFIG ============================
GENES = {
    "RNU4-2":   "NR_003137.2",   # ReNU
    "RNU4ATAC": "NR_023343.1",   # MOPD1
}
BASES = ("A", "C", "G", "U")

# base model class + its ForMaskedLM class name (for the LM head)
MODELS = {
    "rnafm":    {"cls": RnaFmModel,    "name": "multimolecule/rnafm",    "mlm": "RnaFmForMaskedLM"},
    "erniarna": {"cls": ErnieRnaModel, "name": "multimolecule/ernierna", "mlm": "ErnieRnaForMaskedLM"},
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


def load_base(key, device):
    """Load base model (ErnieRNA gets the pairwise-bias remap from 51)."""
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


def load_lm_head(key, device):
    """Load the ForMaskedLM just to borrow its LM head."""
    mlm_cls = getattr(multimolecule, MODELS[key]["mlm"], None)
    if mlm_cls is None:
        print(f"      ⚠ {MODELS[key]['mlm']} not found; skipping {key} PLL.")
        return None
    try:
        mlm = mlm_cls.from_pretrained(MODELS[key]["name"]).eval().to(device)
        return getattr(mlm, "lm_head", None)
    except Exception as e:  # noqa: BLE001
        print(f"      ⚠ could not load {MODELS[key]['mlm']}: {str(e)[:80]}")
        return None


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


def pll_map(seq, tok, base, lm_head, device):
    """Single-sequence masked-marginal PLL criticality per residue."""
    base_ids = {b: tok.convert_tokens_to_ids(b) for b in BASES}
    mask_id = tok.mask_token_id
    inputs = tok(seq, return_tensors="pt")
    ids = inputs["input_ids"].to(device)           # [1, C]
    attn = torch.ones_like(ids)
    ntok = ids.shape[1]
    L = len(seq)
    crit = [0.0] * L
    for i in range(L):
        tok_pos = i + 1                             # +1 for CLS
        if tok_pos >= ntok - 1:
            break
        wt_b = seq[i]
        if wt_b not in base_ids:
            continue
        alts = [b for b in BASES if b != wt_b]
        mids = ids.clone()
        mids[0, tok_pos] = mask_id
        bout = base(input_ids=mids, attention_mask=attn)
        hidden = bout.last_hidden_state if hasattr(bout, "last_hidden_state") else bout[0]  # [1,C,D]
        _out = lm_head((hidden,))
        logits = (_out.logits if hasattr(_out, "logits") else _out)                          # [1,C,V]
        logp = torch.log_softmax(logits[0, tok_pos].float(), dim=-1)
        lp_wt = float(logp[base_ids[wt_b]])
        crit[i] = sum(lp_wt - float(logp[base_ids[b]]) for b in alts) / len(alts)
    return crit


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 84)
    print("RIBOSCOPE — single-sequence masked-PLL control (RNA-FM, ERNIE-RNA)")
    print("=" * 84)

    data = {}
    for gene, acc in GENES.items():
        seq = efetch_seq(acc)
        if not seq:
            print(f"  {gene}: sequence fetch failed ({acc}) — skipping.")
            continue
        cv = fetch_clinvar_positions(gene, acc.split(".")[0])
        print(f"  {gene:<9} L={len(seq):<4} ClinVar pathogenic n={len(cv)}")
        data[gene] = {"accession": acc, "seq": seq, "clinvar": cv}
        time.sleep(0.3)

    results = {g: {"L": len(data[g]["seq"]), "n_clinvar": len(data[g]["clinvar"]),
                   "models": {}} for g in data}
    with torch.no_grad():
        for key in MODELS:
            print(f"\n[{key}] loading base + LM head ...")
            tok, base = load_base(key, device)
            lm_head = load_lm_head(key, device)
            if lm_head is None:
                continue
            for g in data:
                try:
                    crit = pll_map(data[g]["seq"], tok, base, lm_head, device)
                    results[g]["models"][key] = auc_enrichment(crit, data[g]["clinvar"])
                except Exception as e:  # noqa: BLE001
                    print(f"   ⚠ {key}/{g} PLL failed: {type(e).__name__}: {str(e)[:100]}")
                    results[g]["models"][key] = None
            del base, lm_head
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 84)
    print("ARCHITECTURE × READOUT (localization AUC vs ClinVar pathogenic)")
    print("  representation-sensitivity: RNA-FM 0.53 | ERNIE 0.69 | RNA-MSM 0.26 (confounded)")
    print("  masked-PLL (RNA-MSM, from 61):          RNU4-2 0.76 | RNU4ATAC 0.86")
    print("-" * 84)
    print(f"  {'gene':<10}{'RNA-FM PLL':>14}{'ERNIE PLL':>14}")
    for g in data:
        rf = results[g]["models"].get("rnafm")
        er = results[g]["models"].get("erniarna")
        rfs = f"{rf['auc']:.3f} (p={rf['p_value']:.1e})" if rf else "n/a"
        ers = f"{er['auc']:.3f} (p={er['p_value']:.1e})" if er else "n/a"
        print(f"  {g:<10}{rfs:>20}{ers:>22}")
    print("-" * 84)
    print("  Reframe holds if RNA-FM PLL stays near/below chance (~0.5): single-sequence")
    print("  really lacks the signal, and PLL is not a universal fix. If RNA-FM PLL is high,")
    print("  the story changes — tell Claude before submitting.")
    print("=" * 84)

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/singleseq_pll_control.json").write_text(json.dumps(
        {"readout": "singleseq_masked_pll", "genes": results}, indent=2))
    print("✅ Saved outputs/singleseq_pll_control.json")


if __name__ == "__main__":
    main()
