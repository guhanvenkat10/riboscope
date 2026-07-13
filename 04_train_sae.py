"""
04_train_sae.py — RIBOSCOPE Phase 9: Train BatchTopK SAE on RNA-FM layer 6.

What this script does
---------------------
1. Loads the activations file from Phase 8.
2. Filters to layer 6 (the middle layer; structural-motif features tend to
   appear here).
3. Concatenates all per-sequence tensors into one big [N_tokens, hidden_dim]
   matrix.
4. Trains a BatchTopK SAE with 8x dictionary expansion (5120 features) for
   500 gradient steps.
5. Saves SAE weights + per-step training history.
6. Reports final reconstruction loss, L0 sparsity, explained variance, and
   the dead-feature count.

Honest caveat
-------------
This SAE is trained on only ~736 tokens — not enough for biologically
meaningful features to emerge. This is a PIPELINE TEST, intended to verify:
  - the training loop runs end-to-end
  - reconstruction loss decreases monotonically
  - average L0 hits the target (~32 features active per token)
  - dead-feature count is reasonable

Real biology starts in Phase 11+, when we scale the activation extraction
to ~100k–1M sequences from RNAcentral.

Run with
--------
    cd ~/projects/riboscope
    uv run python 04_train_sae.py
"""

import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Run Phase 3 setup steps in GETTING_STARTED.md")
    sys.exit(1)

# Local module — this is sae_models.py in the same directory
from sae_models import BatchTopKSAE


# ============================ CONFIG ============================
# Which layer's activations to train on (1, 6, or 11 — these were the
# layers we hooked in Phase 8)
LAYER = 6

# SAE hyperparameters
DICT_SIZE = 5120          # 8x expansion over RNA-FM hidden_dim=640
K_SPARSITY = 32           # target average L0 (active features per token)
LEARNING_RATE = 5e-4
BATCH_SIZE = 128

# Training duration. We use "steps" instead of "epochs" because with such
# a tiny dataset (~736 tokens), epoch boundaries are meaningless. Each
# step samples a random batch of BATCH_SIZE tokens.
N_TRAINING_STEPS = 500

# I/O paths
ACT_FILE = Path("outputs/activations_rnafm_test.safetensors")
SAE_OUT = Path(f"outputs/sae_layer{LAYER}_v1.safetensors")
HISTORY_OUT = Path(f"outputs/sae_layer{LAYER}_v1_history.json")
# ================================================================


def main() -> None:
    print("=" * 60)
    print(f"RIBOSCOPE Phase 9: Train BatchTopK SAE on layer {LAYER}")
    print("=" * 60)

    # --------------------------------------------------------------- #
    # 1. Load + prep activations
    # --------------------------------------------------------------- #
    print(f"[1/5] Loading activations from {ACT_FILE}...")
    if not ACT_FILE.exists():
        print(f"❌ Activations file not found at {ACT_FILE}.")
        print(f"   Run Phase 8 first:  uv run python 03_extract_activations.py")
        sys.exit(1)

    all_acts = load_file(str(ACT_FILE))

    # Filter keys to the chosen layer
    layer_keys = sorted(k for k in all_acts.keys() if k.endswith(f"__layer{LAYER}"))
    if not layer_keys:
        print(f"❌ No layer-{LAYER} activations found in {ACT_FILE}.")
        print(f"   First 5 available keys:")
        for k in list(all_acts.keys())[:5]:
            print(f"     {k}")
        sys.exit(1)

    print(f"      Found {len(layer_keys)} sequences with layer-{LAYER} activations.")

    # Concatenate into one big [N_tokens, hidden_dim] matrix.
    # Each per-sequence tensor has shape [seq_len_i, hidden_dim]; cat along dim=0
    # stacks them all into one matrix. We cast back to float32 for training
    # (Phase 8 saved as float16 for disk efficiency).
    activations = torch.cat([all_acts[k].float() for k in layer_keys], dim=0)
    n_tokens, d_input = activations.shape
    print(f"      Total tokens:     {n_tokens}")
    print(f"      Hidden dim:       {d_input}")
    print(
        f"      Activation stats: mean={activations.mean():.4f}, "
        f"std={activations.std():.4f}, abs_max={activations.abs().max():.4f}"
    )

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = activations.to(device)
    print(f"      Device:           {device}")

    # --------------------------------------------------------------- #
    # 2. Initialize SAE
    # --------------------------------------------------------------- #
    print(f"[2/5] Initializing BatchTopK SAE...")
    sae = BatchTopKSAE(d_input=d_input, d_dict=DICT_SIZE, k=K_SPARSITY).to(device)
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"      Dictionary size: {DICT_SIZE}  ({DICT_SIZE / d_input:.0f}x expansion)")
    print(f"      Target avg L0:   {K_SPARSITY}")
    print(f"      Parameters:      {n_params:,}")

    # --------------------------------------------------------------- #
    # 3. Train
    # --------------------------------------------------------------- #
    print(f"[3/5] Training for {N_TRAINING_STEPS} steps "
          f"(batch size {BATCH_SIZE}, lr {LEARNING_RATE})...")
    optimizer = torch.optim.AdamW(sae.parameters(), lr=LEARNING_RATE)

    history = {
        "step": [],
        "recon_loss": [],
        "l0_sparsity": [],
        "explained_variance": [],
    }

    sae.train()
    for step in range(N_TRAINING_STEPS):
        # Sample a random batch of tokens (with replacement is fine)
        batch_idx = torch.randint(0, n_tokens, (BATCH_SIZE,), device=device)
        batch = activations[batch_idx]

        # Forward pass
        x_hat, h_sparse = sae(batch)

        # Reconstruction loss = mean squared error
        recon_loss = ((x_hat - batch) ** 2).mean()

        # Backward + update
        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()

        # CRITICAL: re-normalize decoder rows after every step.
        # Without this, the SAE collapses into trivial solutions.
        sae.normalize_decoder()

        # Track diagnostics
        with torch.no_grad():
            # L0 = number of non-zero features per sample, averaged
            l0 = (h_sparse > 0).float().sum(dim=1).mean().item()
            # Explained variance = 1 - var(residual) / var(input)
            var_resid = (batch - x_hat).var().item()
            var_input = batch.var().item()
            explained_var = 1.0 - (var_resid / max(var_input, 1e-12))

        history["step"].append(step)
        history["recon_loss"].append(float(recon_loss.item()))
        history["l0_sparsity"].append(float(l0))
        history["explained_variance"].append(float(explained_var))

        # Print progress every 50 steps
        if step == 0 or (step + 1) % 50 == 0:
            print(
                f"      step {step + 1:4d}  "
                f"|  recon_loss={recon_loss.item():7.4f}  "
                f"|  L0={l0:5.1f}  "
                f"|  explained_var={explained_var:6.3f}"
            )

    # --------------------------------------------------------------- #
    # 4. Save
    # --------------------------------------------------------------- #
    print(f"[4/5] Saving SAE weights and training history...")
    SAE_OUT.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.detach().cpu() for k, v in sae.state_dict().items()}
    save_file(state_dict, str(SAE_OUT))

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)

    # --------------------------------------------------------------- #
    # 5. Final summary + sanity checks
    # --------------------------------------------------------------- #
    print(f"[5/5] Final stats:")
    print(f"      Final recon loss:    {history['recon_loss'][-1]:.4f}")
    print(f"      Final L0 sparsity:   {history['l0_sparsity'][-1]:.1f}  "
          f"(target: {K_SPARSITY})")
    print(f"      Final explained var: {history['explained_variance'][-1]:.3f}")

    # Dead-feature count: how many features never fire on any token?
    sae.eval()
    with torch.no_grad():
        all_features = sae.encode(activations)
        any_active = (all_features > 0).any(dim=0)
        n_dead = (~any_active).sum().item()
        n_alive = DICT_SIZE - n_dead
    print(f"      Dead features:       {n_dead} / {DICT_SIZE} "
          f"({100 * n_dead / DICT_SIZE:.1f}%)  — alive: {n_alive}")

    # Loss reduction sanity check
    initial_loss = history['recon_loss'][0]
    final_loss = history['recon_loss'][-1]
    reduction = (initial_loss - final_loss) / max(initial_loss, 1e-8)
    print(f"      Loss reduction:      {100 * reduction:.1f}% "
          f"(from {initial_loss:.4f} to {final_loss:.4f})")

    print()
    print("=" * 60)
    print("✅ SAE training complete!")
    print("=" * 60)
    print(f"   SAE weights:      {SAE_OUT}")
    print(f"   Training history: {HISTORY_OUT}")
    print()
    print("   What to look for in the output above:")
    print("   ✓ Loss reduction > 60%               (training actually learned)")
    print(f"   ✓ Final L0 ≈ {K_SPARSITY}               (sparsity hit target)")
    print("   ✓ Explained var > 0.5                (decent reconstruction)")
    print("   ✓ Dead features < 50%                (most of the dictionary is in use)")
    print()
    print("   Reminder: this SAE was trained on only 736 tokens. The features")
    print("   are unlikely to represent meaningful biology yet — we need")
    print("   millions of tokens for that. Phase 10 will inspect what these")
    print("   small-dataset features look like, then Phase 11 scales up.")


if __name__ == "__main__":
    main()
