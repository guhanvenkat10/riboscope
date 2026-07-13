"""
23_extract_snodb_features.py — RIBOSCOPE G3 (snoRNA discovery) step 1b.

Takes the C/D-box snoRNAs from snoDB, runs each through the locked single-seq
SAEs (RNA-FM, ErnieRNA), and saves, per model:
  - sae_max     [n_snoRNA, d_dict]   max SAE-feature activation over the snoRNA
  - embed_mean  [n_snoRNA, hidden]   mean-pooled raw layer-6 embedding (BASELINE)
plus a metadata table with snoDB-derived functional labels (canonical / orphan /
non-canonical) for the functional-axis test in step 2.

Why C/D only: our strongest, all-3-model-replicated specialist group is C/D-box
snoRNAs, so that is where the method is most trustworthy.

ErnieRNA load fix (version drift): multimolecule's ErnieRnaModel.__init__ raises
if len(tokenizer) != config.vocab_size. The checkpoint embedding has
config.vocab_size rows; the tokenizer is a couple of unused tokens short. We pad
the tokenizer with dummy tokens that real RNA input never emits, so the guard
passes and the pretrained weights still load fully. RNA-FM is unaffected and is
extracted independently, so an ErnieRNA hiccup can't block it.

RNA-MSM is NOT here: it needs native-MSA input (our locked result). It is added
later as confirmation via per-snoRNA alignments.

Run with
--------
    cd ~/projects/riboscope
    uv run python 23_extract_snodb_features.py              # both models
    uv run python 23_extract_snodb_features.py rnafm        # one model

Inputs : data/snodb_all.tsv  (from 22_fetch_snodb.py)
Outputs: outputs/snodb_cd_metadata.tsv
         outputs/snodb_cd_features_{model}.safetensors  (keys: sae_max, embed_mean)
         sequences/snodb_cd.fasta  (for later RNA-MSM/MSA use)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
    from multimolecule import RnaTokenizer, RnaFmModel, ErnieRnaModel
    from transformers import AutoConfig
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE

# ============================ CONFIG ============================
SNODB_TSV = Path("data/snodb_all.tsv")
BOX_TYPES_KEEP = {"C/D"}          # our replicated specialist class
MIN_LEN, MAX_LEN = 50, 510        # model training-distribution length window
FASTA_OUT = Path("sequences/snodb_cd.fasta")
META_OUT = Path("outputs/snodb_cd_metadata.tsv")

MODELS = {
    "rnafm": {
        "model_class": RnaFmModel,
        "model_name": "multimolecule/rnafm",
        "layer": 6,
        "sae_file": Path("outputs/sae_big_layer6_v3.safetensors"),
    },
    "erniarna": {
        "model_class": ErnieRnaModel,
        "model_name": "multimolecule/ernierna",
        "layer": 6,
        "sae_file": Path("outputs/sae_erniarna_layer6_v3.safetensors"),
    },
}

# target columns used to derive functional labels
TARGET_COLS = ["rrna_targets", "snrna_targets", "lncrna_targets",
               "protein_coding_targets", "snorna_targets", "mirna_targets",
               "trna_targets", "ncrna_targets", "pseudogene_targets", "other_targets"]
NONCANON_COLS = ["protein_coding_targets", "lncrna_targets", "mirna_targets"]
# ================================================================


def nonempty(v) -> bool:
    return bool(v is not None and str(v).strip() and str(v).strip().lower() != "nan")


def parse_snodb(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def clean_seq(s: str) -> str:
    return s.strip().upper().replace("T", "U").replace(" ", "")


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
            print(f"      ErnieRNA: padded tokenizer {want - n_pad}->{len(tok)} to match config vocab.")
        model = ModelCls.from_pretrained(name, tokenizer=tok)
        # CRITICAL: repair structure-attention weights — checkpoint names them
        # model.pairwise_bias_proj.dense1/dense2; current code wants the Sequential
        # pairwise_bias_proj.0/2. Without this they load random and features are wrong.
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


def build_metadata(rows: list[dict]) -> list[dict]:
    """Filter to C/D snoRNAs in the length window with a usable sequence; derive labels."""
    out = []
    for r in rows:
        if r.get("box_type", "").strip() not in BOX_TYPES_KEEP:
            continue
        seq = clean_seq(r.get("sequence", "") or "")
        if not (MIN_LEN <= len(seq) <= MAX_LEN):
            continue
        if any(c not in "ACGUN" for c in seq):
            seq = "".join(c if c in "ACGUN" else "N" for c in seq)
        any_target = nonempty(r.get("target_count")) or any(nonempty(r.get(c)) for c in TARGET_COLS)
        out.append({
            "snodb_id": r.get("snodb_id", ""),
            "gene_name": r.get("gene_name", ""),
            "box_type": r.get("box_type", ""),
            "length": len(seq),
            "sequence": seq,
            "conservation_phastcons": r.get("conservation_phastcons", ""),
            "host_biotype": r.get("host_biotype", ""),
            "host_function": r.get("host_function", ""),
            "target_count": r.get("target_count", ""),
            "target_biotypes": r.get("target_biotypes", ""),
            "rrna_targets": r.get("rrna_targets", ""),
            "label_canonical_rrna": int(nonempty(r.get("rrna_targets"))),
            "label_noncanonical": int(any(nonempty(r.get(c)) for c in NONCANON_COLS)),
            "label_orphan": int(not any_target),
        })
    return out


def main() -> None:
    which = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] in MODELS else list(MODELS)
    print("=" * 76)
    print(f"RIBOSCOPE G3 step 1b: snoDB C/D-box feature extraction — models={which}")
    print("=" * 76)

    if not SNODB_TSV.exists():
        print(f"❌ {SNODB_TSV} not found. Run 22_fetch_snodb.py first.")
        sys.exit(1)

    rows = parse_snodb(SNODB_TSV)
    meta = build_metadata(rows)
    print(f"[1/3] C/D-box snoRNAs kept (len {MIN_LEN}-{MAX_LEN}): {len(meta)} / {len(rows)} total")
    nc = sum(m["label_canonical_rrna"] for m in meta)
    no = sum(m["label_orphan"] for m in meta)
    nn = sum(m["label_noncanonical"] for m in meta)
    print(f"      labels — canonical(rRNA): {nc}   orphan(no target): {no}   non-canonical: {nn}")

    # Write metadata (row order == matrix row order) + FASTA
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    FASTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = [k for k in meta[0] if k != "sequence"]
    with open(META_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=meta_cols, delimiter="\t")
        w.writeheader()
        for m in meta:
            w.writerow({k: m[k] for k in meta_cols})
    with open(FASTA_OUT, "w") as f:
        for m in meta:
            f.write(f">{m['snodb_id']}|{m['gene_name']}|{m['box_type']}\n{m['sequence']}\n")
    print(f"      wrote {META_OUT} and {FASTA_OUT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for key in which:
        cfg = MODELS[key]
        layer = cfg["layer"]
        if not cfg["sae_file"].exists():
            print(f"  ⚠ {key}: SAE {cfg['sae_file']} missing — skipping.")
            continue
        print(f"\n[2/3] {key}: loading SAE + model ...")
        sae_state = load_file(str(cfg["sae_file"]))
        d_input, d_dict = sae_state["W_enc"].shape[0], sae_state["W_enc"].shape[1]
        sae = BatchTopKSAE(d_input=d_input, d_dict=d_dict, k=32)
        sae.load_state_dict(sae_state)
        sae = sae.eval().to(device)
        try:
            tok, model = load_model(cfg, key, device)
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {key}: failed to load model ({type(e).__name__}: {e}). Skipping this model.")
            continue
        layers = find_encoder_layers(model)
        captured: dict = {}
        handle = layers[layer].register_forward_hook(make_hook(captured, layer))

        sae_max_rows, embed_mean_rows = [], []
        with torch.no_grad():
            for m in tqdm(meta, desc=f"{key}", unit="sno"):
                inputs = tok(m["sequence"], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs)
                act = captured[layer][0].float().to(device)     # [ntok, hidden]
                ntok = act.shape[0]
                if ntok <= 2:
                    sae_max_rows.append(torch.zeros(d_dict))
                    embed_mean_rows.append(torch.zeros(d_input))
                    continue
                real = act[1:ntok - 1]                            # drop CLS/EOS
                feats = sae.encode(real)                          # [R, d_dict]
                sae_max_rows.append(feats.max(dim=0).values.cpu())
                embed_mean_rows.append(real.mean(dim=0).cpu())
        handle.remove()

        out_file = Path(f"outputs/snodb_cd_features_{key}.safetensors")
        save_file({
            "sae_max": torch.stack(sae_max_rows).contiguous(),
            "embed_mean": torch.stack(embed_mean_rows).contiguous(),
        }, str(out_file))
        print(f"[3/3] {key}: saved {out_file}  "
              f"(sae_max [{len(meta)},{d_dict}], embed_mean [{len(meta)},{d_input}])")

    print("\n" + "=" * 76)
    print("✅ Feature extraction done. Next: step 2 functional-axis test (canonical vs orphan)")
    print("   with single-model / k-mer / shuffled-label baselines.")
    print("   Run ~/projects/riboscope/sync_to_windows.sh so the matrices + metadata sync back.")
    print("=" * 76)


if __name__ == "__main__":
    main()
