"""
set_model.py — switch train/inspect scripts to target a specific model.

Updates the path variables in 08_train_sae_big.py and 09_inspect_features_big.py
so they point at the activations / SAE / report files for a specific RNA
foundation model, AND patches the LAYER constant + FASTA_FILE in both scripts so
a single command fully reconfigures the run.

Preserves all other code in those scripts (resampling logic, dead-feature
monitoring, hyperparameters, etc.). Only the Path(...) lines, the LAYER line,
and the FASTA_FILE line change.

Usage
-----
    uv run python set_model.py rnafm
    uv run python set_model.py erniarna
    uv run python set_model.py rnamsm
    uv run python set_model.py rnamsm_msa        # RNA-MSM, NATIVE MSA input, layer 8
    uv run python set_model.py rnamsm_msa_l6     # same, layer-6 control

After running this for a model, you can run training/inspection normally:
    uv run python 08_train_sae_big.py
    uv run python 09_inspect_features_big.py
"""

import re
import sys
from pathlib import Path

# Every config carries a `layer` and `fasta_file` so set_model fully configures
# the run. For the original three models these are the historical defaults
# (layer 6, rfam_30k.fasta) — patching them is a no-op, so behavior is unchanged.
CONFIGS = {
    "rnafm": {
        "layer": 6,
        "fasta_file": "sequences/rfam_30k.fasta",
        "act_file": "outputs/activations_big_layer6_v2.safetensors",
        "sae_name": "sae_big_layer{LAYER}_v3",
        "inspection_name": "inspection_big_layer{LAYER}_v3",
        "checkpoint_dir": "outputs/checkpoints_rnafm_v3",
    },
    "erniarna": {
        "layer": 6,
        "fasta_file": "sequences/rfam_30k.fasta",
        "act_file": "outputs/activations_erniarna_layer6.safetensors",
        "sae_name": "sae_erniarna_layer{LAYER}_v3",
        "inspection_name": "inspection_erniarna_layer{LAYER}_v3",
        "checkpoint_dir": "outputs/checkpoints_erniarna_v3",
    },
    "rnamsm": {
        # v3 (2026-05-28): retrained on POSITION-CENTERED activations.
        #   v1 (dict=8192) and v2 (dict=4096 + N-mask) both hit EV=0.998 with
        #   ZERO biology — the dictionary collapsed onto a per-position signal
        #   (top features fire on fixed token positions across all Rfam
        #   families; specialists max out at 0.5-0.9 vs RNA-FM's 5-65).
        #   Verified cause (config.json + modeling source): RNA-MSM is an MSA
        #   Transformer run on single sequences (msa_depth=1), so its learned
        #   absolute position term dominates the residual stream. Fix: subtract
        #   the per-position mean before SAE training (compute_position_means.py
        #   -> subtract_position_means.py). v3 holds ALL of 08's hyperparameters
        #   fixed (dict=4096, N-mask, k=32, LR=4e-4) so v2->v3 is a clean A/B
        #   isolating position-centering; v2 is the negative control.
        #   v1/v2 SAEs preserved as baselines.
        "layer": 6,
        "fasta_file": "sequences/rfam_30k.fasta",
        "act_file": "outputs/activations_rnamsm_layer6_poscentered.safetensors",
        "sae_name": "sae_rnamsm_layer{LAYER}_v3",
        "inspection_name": "inspection_rnamsm_layer{LAYER}_v3",
        "checkpoint_dir": "outputs/checkpoints_rnamsm_v3",
    },
    "rnamsm_msa": {
        # RIBOSCOPE MSA path (2026-05-30): the decisive test of whether RNA-MSM's
        #   SAE failure is caused by OUT-OF-DISTRIBUTION single-sequence input
        #   rather than the model itself. 16_msa_geometry_scan.py showed native
        #   Rfam-SEED MSA input PARTIALLY de-collapses the representation, most
        #   strongly at LAYER 8 (var@32 94%->67%, fam_pcs90 ~10->38, eta ~4x).
        #   17_extract_rnamsm_msa.py re-extracts activations under that MSA input
        #   (same query seqs, same [CLS,res,EOS] token convention); 18 applies the
        #   IDENTICAL v3 preprocessing (z-score + position-center). So the ONLY
        #   difference vs the failed single-seq v3 SAE is the input distribution.
        #   Primary layer = 8 (most de-collapsed). FASTA = the matched query set
        #   17 wrote (a subset of the 30k that had a locatable SEED row), so 08's
        #   N-mask and 09's family tracking line up 1:1 with the extracted tokens.
        #   Judge the result by top-specialist MAX ACTIVATION (must approach
        #   RNA-FM's 5-65) + motif consistency, NOT EV alone (feedback_sae_health).
        "layer": 8,
        "fasta_file": "sequences/rfam_msa_query.fasta",
        "act_file": "outputs/activations_rnamsm_msa_layer8_poscentered.safetensors",
        "sae_name": "sae_rnamsm_msa_layer{LAYER}_v1",
        "inspection_name": "inspection_rnamsm_msa_layer{LAYER}_v1",
        "checkpoint_dir": "outputs/checkpoints_rnamsm_msa_l8_v1",
    },
    "rnamsm_msa_l6": {
        # Same MSA path at LAYER 6 — the same layer the other two models' SAEs use.
        #   Acts as a within-MSA control: if specialists appear at L8 but not L6,
        #   that localizes the in-distribution signal; if at neither, it
        #   strengthens the mechanistic-negative read. {LAYER}=6 keeps the SAE /
        #   inspection filenames distinct from the L8 run.
        "layer": 6,
        "fasta_file": "sequences/rfam_msa_query.fasta",
        "act_file": "outputs/activations_rnamsm_msa_layer6_poscentered.safetensors",
        "sae_name": "sae_rnamsm_msa_layer{LAYER}_v1",
        "inspection_name": "inspection_rnamsm_msa_layer{LAYER}_v1",
        "checkpoint_dir": "outputs/checkpoints_rnamsm_msa_l6_v1",
    },
    "rnamsm_single": {
        # THE DECISIVE CONTROL (2026-05-30) for the input-conditional-interpretability
        #   claim. The MSA run (rnamsm_msa) yielded 376 healthy specialists at L8 on a
        #   2327-seq / top-100-family subset. To prove that the *MSA input* — not the
        #   smaller subset — is what recovered specialists, this trains a SINGLE-SEQUENCE
        #   SAE on the EXACT SAME 2327 sequences (sequences/rfam_msa_query.fasta), same
        #   layer 8, same preprocessing. The ONLY difference vs rnamsm_msa is the input
        #   distribution (msa_depth=1, out-of-distribution). Acts come from
        #   19_extract_rnamsm_single_subset.py -> 18_prep_msa_acts.py 8 --stem single.
        #   PRE-REGISTERED PREDICTION: still collapses (top ≤2-fam specialist ~0.1),
        #   since 15-vs-16 geometry on this same subset stayed collapsed single-seq
        #   (var@32 94%) but de-collapsed under MSA (67%). Judge by specialist max-act +
        #   motif consistency per feedback_sae_health, NOT EV.
        "layer": 8,
        "fasta_file": "sequences/rfam_msa_query.fasta",
        "act_file": "outputs/activations_rnamsm_single_layer8_poscentered.safetensors",
        "sae_name": "sae_rnamsm_single_layer{LAYER}_v1",
        "inspection_name": "inspection_rnamsm_single_layer{LAYER}_v1",
        "checkpoint_dir": "outputs/checkpoints_rnamsm_single_l8_v1",
    },
    "rnamsm_single_l6": {
        # Layer-6 single-seq control over the same subset (parallels rnamsm_msa_l6).
        "layer": 6,
        "fasta_file": "sequences/rfam_msa_query.fasta",
        "act_file": "outputs/activations_rnamsm_single_layer6_poscentered.safetensors",
        "sae_name": "sae_rnamsm_single_layer{LAYER}_v1",
        "inspection_name": "inspection_rnamsm_single_layer{LAYER}_v1",
        "checkpoint_dir": "outputs/checkpoints_rnamsm_single_l6_v1",
    },
}


def replace_path_line(text: str, var_name: str, new_path_expr: str) -> tuple[str, bool]:
    """
    Find a line like `VAR = Path(...)` and replace the Path(...) call.
    Preserves indentation, leaves trailing comments alone (drops them — safer).
    Returns (new_text, was_replaced).
    """
    pattern = rf'^(\s*){var_name}\s*=\s*Path\([^)]*\).*$'
    replacement = rf'\g<1>{var_name} = Path({new_path_expr})'
    new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    return new_text, n > 0


def replace_int_line(text: str, var_name: str, new_int: int) -> tuple[str, bool]:
    """
    Find a bare integer assignment line like `VAR = 6` and replace the value.
    Requires the RHS to be an integer literal so it can't accidentally match a
    Path(...) or f-string line. Drops trailing comments on that line (safer).
    """
    pattern = rf'^(\s*){var_name}\b\s*=\s*\d+.*$'
    replacement = rf'\g<1>{var_name} = {new_int}'
    new_text, n = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    return new_text, n > 0


def update_train_script(cfg: dict) -> bool:
    p = Path("08_train_sae_big.py")
    if not p.exists():
        print(f"❌ {p} not found in current directory")
        return False
    t = p.read_text()
    original = t

    t, ok_layer = replace_int_line(t, "LAYER", cfg["layer"])
    t, ok_fasta = replace_path_line(t, "FASTA_FILE", f'"{cfg["fasta_file"]}"')
    t, ok1 = replace_path_line(t, "ACT_FILE", f'"{cfg["act_file"]}"')
    t, ok2 = replace_path_line(t, "SAE_FINAL", f'f"outputs/{cfg["sae_name"]}.safetensors"')
    t, ok3 = replace_path_line(t, "HISTORY_OUT", f'f"outputs/{cfg["sae_name"]}_history.json"')
    t, ok4 = replace_path_line(t, "CHECKPOINT_DIR", f'"{cfg["checkpoint_dir"]}"')

    checks = {"LAYER": ok_layer, "FASTA_FILE": ok_fasta, "ACT_FILE": ok1,
              "SAE_FINAL": ok2, "HISTORY_OUT": ok3, "CHECKPOINT_DIR": ok4}
    if not all(checks.values()):
        missing = [n for n, ok in checks.items() if not ok]
        print(f"⚠ {p}: could not find these variables to replace: {missing}")
        print(f"   The file may have a different structure than expected. No changes saved.")
        return False

    if t == original:
        print(f"  {p}: (already up to date)")
    else:
        p.write_text(t)
        print(f"✓ Updated {p}")
    return True


def update_inspect_script(cfg: dict) -> bool:
    p = Path("09_inspect_features_big.py")
    if not p.exists():
        print(f"❌ {p} not found in current directory")
        return False
    t = p.read_text()
    original = t

    t, ok_layer = replace_int_line(t, "LAYER", cfg["layer"])
    t, ok_fasta = replace_path_line(t, "FASTA_FILE", f'"{cfg["fasta_file"]}"')
    t, ok1 = replace_path_line(t, "ACT_FILE", f'"{cfg["act_file"]}"')
    t, ok2 = replace_path_line(t, "SAE_FILE", f'f"outputs/{cfg["sae_name"]}.safetensors"')
    t, ok3 = replace_path_line(t, "REPORT_OUT", f'f"outputs/{cfg["inspection_name"]}.json"')

    checks = {"LAYER": ok_layer, "FASTA_FILE": ok_fasta, "ACT_FILE": ok1,
              "SAE_FILE": ok2, "REPORT_OUT": ok3}
    if not all(checks.values()):
        missing = [n for n, ok in checks.items() if not ok]
        print(f"⚠ {p}: could not find these variables to replace: {missing}")
        print(f"   No changes saved.")
        return False

    if t == original:
        print(f"  {p}: (already up to date)")
    else:
        p.write_text(t)
        print(f"✓ Updated {p}")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: uv run python set_model.py <model>")
        print(f"  Available models: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    model = sys.argv[1]
    if model not in CONFIGS:
        print(f"❌ Unknown model: {model}")
        print(f"   Available: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    cfg = CONFIGS[model]
    resolved_sae = cfg["sae_name"].replace("{LAYER}", str(cfg["layer"]))
    resolved_insp = cfg["inspection_name"].replace("{LAYER}", str(cfg["layer"]))
    print(f"Switching train + inspect scripts to target: {model}")
    print(f"  LAYER:           {cfg['layer']}")
    print(f"  FASTA_FILE:      {cfg['fasta_file']}")
    print(f"  ACT_FILE:        {cfg['act_file']}")
    print(f"  SAE file:        outputs/{resolved_sae}.safetensors")
    print(f"  Inspection JSON: outputs/{resolved_insp}.json")
    print(f"  Checkpoints:     {cfg['checkpoint_dir']}")
    print()

    ok1 = update_train_script(cfg)
    ok2 = update_inspect_script(cfg)

    if ok1 and ok2:
        print()
        print(f"Ready. Next steps for {model}:")
        print(f"  uv run python 08_train_sae_big.py        # train (~2 hr)")
        print(f"  uv run python 09_inspect_features_big.py # inspect (~10 min)")


if __name__ == "__main__":
    main()
