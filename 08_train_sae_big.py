"""
08_train_sae_big.py — RIBOSCOPE Phase 12/13: Production SAE training (v4).

v4 changes from v3:
  - Resampling BUG FIX. v3 tracked "fired in any prior training step" via a
    cumulative window_fired bool — but with k=32 and batch_size=512, every
    feature gets a chance to fire within a few batches, so the window
    saturated within ~10 steps and the dead-mask became permanently empty.
    Resampling never triggered (the v3 run showed "Total resampled: 0").
  - v4 drops the window-based tracking entirely. Dead features are now
    identified by chunked_eval()'s ever_fired_mask on the HELD-OUT EVAL
    SET — which is the meaningful definition of dead (didn't generalize)
    and is what v2's working resampling effectively measured.
  - chunked_eval() now returns the full per-feature ever_fired_mask
    rather than just a percentage, so the same eval pass that computes
    diagnostics can also drive resampling.

v3 features kept (all still good):
  - Chunked eval (bounded VRAM regardless of d_dict)
  - DICT_SIZE = 8192
  - Anthropic-style residual-direction resampling (just now actually fires)
  - N_STEPS = 16000

Run with
--------
    uv run python set_model.py <model_tag>   # to point at the right paths
    uv run python 08_train_sae_big.py
"""

import json
import math
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    sys.exit(1)

from sae_models import BatchTopKSAE


# ============================ CONFIG ============================
LAYER = 6

# SAE hyperparameters
DICT_SIZE = 4096          # v3: 8192 (RNA-FM/ErnieRNA). v2-rnamsm: dropped to
                          # 4096 because RNA-MSM's representations are lower-rank
                          # — first 8192 run collapsed into N-token artifacts.
K_SPARSITY = 32
LEARNING_RATE = 4e-4
BATCH_SIZE = 512

# N-token masking (Phase 13b NEW — RNA-MSM mitigation)
# RNA-MSM was pretrained on MSAs that contain N (gap/unknown) tokens at high
# frequency. The model's per-position embeddings for N tokens are distinct
# enough that a vanilla SAE allocates substantial capacity to detecting them,
# squeezing out biological motif features. Excluding N tokens from training
# forces the SAE to learn from real nucleotide positions only.
# When True, also requires FASTA_FILE to be readable so we can map each token
# back to its nucleotide character.
MASK_N_TOKENS = True
FASTA_FILE = Path("sequences/rfam_30k.fasta")

# Training duration
N_STEPS = 16000           # v3: bumped from 12000
WARMUP_STEPS = 400        # v3: bumped from 300 (longer training → longer warmup)

# Eval / checkpointing
EVAL_FRAC = 0.05
EVAL_EVERY = 500
EVAL_CHUNK_SIZE = 20_000  # v3 NEW: tokens per chunk during eval (memory bound)
CHECKPOINT_EVERY = 4000
LOG_EVERY = 250

# Dead-feature monitoring + resampling
DEAD_FEATURE_CHECK_EVERY = 1000
RESAMPLE_TOP_K = 256      # how many dead features to resample per check
RESAMPLING_STARTS_AT = 1500   # skip first ~10% of training before first resample
RESAMPLING_ENDS_AT = 10_000   # stop resampling in the last third (let things settle)

# I/O paths (set_model.py rewrites these for each model)
ACT_FILE = Path("outputs/activations_big_layer6_v2.safetensors")
SAE_FINAL = Path(f"outputs/sae_big_layer{LAYER}_v2.safetensors")
HISTORY_OUT = Path(f"outputs/sae_big_layer{LAYER}_v2_history.json")
CHECKPOINT_DIR = Path("outputs/checkpoints_v2")
# ================================================================


# ----------------------------------------------------------------- #
# FASTA parser (Phase 13b NEW — needed for N-token masking)
# ----------------------------------------------------------------- #
def parse_fasta_with_metadata(path: Path) -> dict:
    """Parse the Rfam FASTA produced by 06_fetch_rfam_sequences.py.

    Returns dict[seq_name -> {sequence: str, rfam_id: str, rfam_name: str}].
    Sequence has T→U normalized (matches what the model tokenizers consume).
    Duplicated from 09_inspect_features_big.py — kept inline so 08 has no
    extra import surface.
    """
    out: dict[str, dict] = {}
    name = None
    rfam_id = None
    rfam_name = None
    seq_buf: list[str] = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = {
                        "sequence": "".join(seq_buf),
                        "rfam_id": rfam_id,
                        "rfam_name": rfam_name,
                    }
                parts = [p.strip() for p in line[1:].split("|")]
                name = parts[0] if len(parts) > 0 else "unknown"
                rfam_id = parts[1] if len(parts) > 1 else None
                rfam_name = parts[2] if len(parts) > 2 else None
                seq_buf = []
            else:
                seq_buf.append(line.replace("T", "U").replace("t", "u"))
        if name is not None:
            out[name] = {
                "sequence": "".join(seq_buf),
                "rfam_id": rfam_id,
                "rfam_name": rfam_name,
            }
    return out


def build_n_token_mask(layer_keys: list[str], raw: dict, seq_meta: dict) -> torch.Tensor:
    """Build a [n_tokens] bool mask where True = keep, False = drop (N token).

    Mirrors the per-token indexing convention used everywhere else: position 0
    of each per-sequence tensor is CLS, position n-1 is EOS, positions 1..n-2
    are nucleotides corresponding to sequence[0..n-3]. We keep CLS/EOS (they
    don't dominate training) and drop only true N nucleotide positions.
    """
    chunks: list[torch.Tensor] = []
    for key in layer_keys:
        seq_name = key.replace(f"__layer{LAYER}", "")
        n = raw[key].shape[0]
        seq_str = seq_meta.get(seq_name, {}).get("sequence", "").upper()
        m = torch.ones(n, dtype=torch.bool)
        # Positions 1..n-2 correspond to seq_str[0..n-3]. Drop where char is N.
        for pos in range(1, n - 1):
            nt_pos = pos - 1
            if nt_pos < len(seq_str) and seq_str[nt_pos] == "N":
                m[pos] = False
        chunks.append(m)
    return torch.cat(chunks)


# ----------------------------------------------------------------- #
# LR schedule
# ----------------------------------------------------------------- #
def lr_schedule(step: int, peak_lr: float, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to 10% of peak."""
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ----------------------------------------------------------------- #
# Streaming moments — for chunked variance / explained-variance compute
# ----------------------------------------------------------------- #
class StreamingMoments:
    """Accumulate sum, sum-of-squares, and count for variance computation."""
    def __init__(self) -> None:
        self.sum = 0.0
        self.sum_sq = 0.0
        self.n = 0

    def update(self, x: torch.Tensor) -> None:
        self.sum += x.sum().item()
        self.sum_sq += (x * x).sum().item()
        self.n += x.numel()

    def mean(self) -> float:
        return self.sum / max(self.n, 1)

    def variance(self) -> float:
        if self.n == 0:
            return 0.0
        m = self.sum / self.n
        return max(self.sum_sq / self.n - m * m, 0.0)


# ----------------------------------------------------------------- #
# Chunked eval
# ----------------------------------------------------------------- #
def chunked_eval(sae: BatchTopKSAE, eval_acts_cpu: torch.Tensor, device, chunk_size: int):
    """
    Run the eval set through the SAE in chunks.

    Returns (avg_recon_loss, explained_variance, pct_dead, ever_fired_mask).
    ever_fired_mask is a [d_dict] bool on `device`, True for features that
    fire on at least one eval token. Used both for diagnostics and for
    driving Anthropic-style resampling.

    Bounded peak VRAM = O(chunk_size * d_dict).
    """
    sae.eval()
    loss_sum = 0.0
    n_elems = 0

    input_stats = StreamingMoments()
    resid_stats = StreamingMoments()

    d_dict = sae.W_enc.shape[1]
    ever_fired = torch.zeros(d_dict, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i in range(0, eval_acts_cpu.shape[0], chunk_size):
            chunk = eval_acts_cpu[i:i + chunk_size].to(device, non_blocking=True)
            x_hat, h_sparse = sae(chunk)
            residual = chunk - x_hat

            loss_sum += (residual * residual).sum().item()
            n_elems += chunk.numel()

            input_stats.update(chunk)
            resid_stats.update(residual)

            ever_fired |= (h_sparse > 0).any(dim=0)

            del chunk, x_hat, h_sparse, residual

    sae.train()

    avg_loss = loss_sum / max(n_elems, 1)
    var_in = input_stats.variance()
    var_resid = resid_stats.variance()
    explained_var = 1.0 - var_resid / max(var_in, 1e-12)
    pct_dead = 100.0 * (~ever_fired).sum().item() / d_dict
    return avg_loss, explained_var, pct_dead, ever_fired


# ----------------------------------------------------------------- #
# Anthropic-style dead-feature resampling
# ----------------------------------------------------------------- #
@torch.no_grad()
def resample_dead_features(
    sae: BatchTopKSAE,
    recent_activations: torch.Tensor,   # on device
    dead_mask: torch.Tensor,            # bool [d_dict], True = dead
    max_resample: int = RESAMPLE_TOP_K,
) -> int:
    """
    Reinitialize up to `max_resample` dead features.

    For each dead feature:
      - Pick an input token with high reconstruction loss (uniform from top
        `max_resample * 4` by per-token loss to add randomness).
      - Set the decoder row to the unit-normalized residual of that token.
      - Mirror the encoder column from the decoder row, scaled by the
        average alive-feature encoder norm so the new feature can fire at
        roughly the typical magnitude.
      - Reset encoder bias to 0.

    Returns the number of features actually resampled.
    """
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    n_resample = min(n_dead, max_resample)

    # Per-token reconstruction loss on the recent batch
    x_hat, _ = sae(recent_activations)
    residual = recent_activations - x_hat
    per_token_loss = (residual * residual).sum(dim=1)

    # Pick top-K-by-loss * 4 candidates, sample n_resample of them
    n_candidates = min(n_resample * 4, per_token_loss.shape[0])
    top_loss_vals, top_loss_idx = per_token_loss.topk(n_candidates)
    perm = torch.randperm(n_candidates, device=recent_activations.device)[:n_resample]
    chosen_token_idx = top_loss_idx[perm]
    chosen_residuals = residual[chosen_token_idx]  # [n_resample, d_input]

    # Normalize each residual to unit norm
    residual_norms = chosen_residuals.norm(dim=1, keepdim=True).clamp(min=1e-8)
    new_directions = chosen_residuals / residual_norms  # [n_resample, d_input]

    # Average alive-feature encoder norm — to give new encoder a sensible scale
    alive_mask = ~dead_mask
    if alive_mask.any():
        avg_enc_norm = sae.W_enc.data[:, alive_mask].norm(dim=0).mean().item()
    else:
        avg_enc_norm = 0.2  # fallback

    # Indices of dead features to fill
    dead_indices = dead_mask.nonzero(as_tuple=False).flatten()[:n_resample]

    for i, feat_idx in enumerate(dead_indices.tolist()):
        direction = new_directions[i]  # [d_input], unit norm
        # Decoder row [d_dict, d_input], so row feat_idx
        sae.W_dec.data[feat_idx] = direction
        # Encoder column [d_input, d_dict], scaled to match the alive average
        sae.W_enc.data[:, feat_idx] = direction * avg_enc_norm
        # Reset encoder bias
        sae.b_enc.data[feat_idx] = 0.0

    return n_resample


# ----------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print(f"RIBOSCOPE Phase 12/13: Production SAE training v3 (layer {LAYER})")
    print("=" * 70)

    if not ACT_FILE.exists():
        print(f"❌ Activations file not found: {ACT_FILE}")
        print(f"   Run the corresponding extraction script first.")
        sys.exit(1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- #
    # 1. Load activations
    # --------------------------------------------------------------- #
    print(f"[1/4] Loading activations from {ACT_FILE}...")
    print(f"      (this may take 30-60 seconds for a multi-GB file)")
    raw = load_file(str(ACT_FILE))

    layer_keys = sorted(k for k in raw.keys() if k.endswith(f"__layer{LAYER}"))
    if not layer_keys:
        print(f"❌ No layer-{LAYER} activations found.")
        sys.exit(1)
    print(f"      Sequences with layer-{LAYER}: {len(layer_keys)}")

    activations = torch.cat([raw[k] for k in layer_keys], dim=0).contiguous()
    n_tokens, d_input = activations.shape

    print(f"      Total tokens:    {n_tokens:,}")
    print(f"      Hidden dim:      {d_input}")
    print(f"      RAM footprint:   "
          f"{activations.element_size() * activations.numel() / 1e9:.2f} GB")

    # Phase 13b NEW: N-token masking. Build keep-mask while `raw` is still in
    # scope so we can read per-sequence shapes, then filter activations and
    # release raw.
    if MASK_N_TOKENS:
        if not FASTA_FILE.exists():
            print(f"❌ MASK_N_TOKENS=True but FASTA not found: {FASTA_FILE}")
            sys.exit(1)
        print(f"      Loading FASTA from {FASTA_FILE} for N-mask...")
        seq_meta = parse_fasta_with_metadata(FASTA_FILE)
        print(f"      FASTA sequences: {len(seq_meta):,}")
        keep_mask = build_n_token_mask(layer_keys, raw, seq_meta)
        n_dropped = int((~keep_mask).sum().item())
        del raw
        print(f"      N-mask: dropping {n_dropped:,} of {n_tokens:,} tokens "
              f"({100 * n_dropped / max(n_tokens, 1):.2f}%)")
        activations = activations[keep_mask].contiguous()
        n_tokens = activations.shape[0]
        print(f"      Tokens after N-mask: {n_tokens:,}")
    else:
        del raw

    # Train/eval split
    n_eval = int(n_tokens * EVAL_FRAC)
    n_train = n_tokens - n_eval
    train_acts = activations[:n_train]
    eval_acts_cpu = activations[n_train:].float()  # held on CPU, chunks moved to GPU

    print(f"      Train tokens:    {n_train:,}")
    print(f"      Eval tokens:     {n_eval:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device:          {device}")

    # --------------------------------------------------------------- #
    # 2. SAE
    # --------------------------------------------------------------- #
    print(f"[2/4] Initializing BatchTopK SAE...")
    sae = BatchTopKSAE(d_input=d_input, d_dict=DICT_SIZE, k=K_SPARSITY).to(device)
    n_params = sum(p.numel() for p in sae.parameters())
    print(f"      Dict size:       {DICT_SIZE} ({DICT_SIZE / d_input:.1f}x expansion)")
    print(f"      Target avg L0:   {K_SPARSITY}")
    print(f"      Parameters:      {n_params:,}")

    # --------------------------------------------------------------- #
    # 3. Train
    # --------------------------------------------------------------- #
    print(f"[3/4] Training for {N_STEPS} steps "
          f"(batch {BATCH_SIZE}, peak LR {LEARNING_RATE}, warmup {WARMUP_STEPS})...")
    optimizer = torch.optim.AdamW(sae.parameters(), lr=LEARNING_RATE)

    history: dict[str, list] = {
        "step": [], "lr": [],
        "train_recon_loss": [], "train_l0": [], "train_explained_var": [],
        "eval_step": [], "eval_recon_loss": [], "eval_explained_var": [],
        "dead_check_step": [], "dead_check_pct": [], "resampled_per_check": [],
    }

    total_resampled = 0

    sae.train()
    for step in range(N_STEPS):
        # LR schedule
        lr = lr_schedule(step, LEARNING_RATE, WARMUP_STEPS, N_STEPS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Random batch
        batch_idx = torch.randint(0, n_train, (BATCH_SIZE,))
        batch = train_acts[batch_idx].float().to(device, non_blocking=True)

        # Forward
        x_hat, h_sparse = sae(batch)
        recon_loss = ((x_hat - batch) ** 2).mean()

        # Backward
        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()
        sae.normalize_decoder()

        # Training log
        if step == 0 or (step + 1) % LOG_EVERY == 0:
            with torch.no_grad():
                l0 = (h_sparse > 0).float().sum(dim=1).mean().item()
                var_resid = (batch - x_hat).var().item()
                var_input = batch.var().item()
                exp_var = 1.0 - var_resid / max(var_input, 1e-12)

            history["step"].append(step)
            history["lr"].append(lr)
            history["train_recon_loss"].append(float(recon_loss.item()))
            history["train_l0"].append(float(l0))
            history["train_explained_var"].append(float(exp_var))

            print(
                f"      step {step + 1:5d}  "
                f"lr={lr:.2e}  "
                f"train_loss={recon_loss.item():7.4f}  "
                f"L0={l0:5.1f}  "
                f"train_ev={exp_var:.3f}"
            )

        # Periodic eval (chunked) — also drives the dead-feature check
        if (step + 1) % EVAL_EVERY == 0:
            e_loss, e_ev, e_pct_dead, _ = chunked_eval(
                sae, eval_acts_cpu, device, EVAL_CHUNK_SIZE
            )
            history["eval_step"].append(step + 1)
            history["eval_recon_loss"].append(float(e_loss))
            history["eval_explained_var"].append(float(e_ev))
            print(f"        [eval at step {step + 1}: "
                  f"loss={e_loss:.4f}  ev={e_ev:.3f}  dead_on_eval={e_pct_dead:.1f}%]")

        # Periodic dead-feature check + resample
        if (step + 1) % DEAD_FEATURE_CHECK_EVERY == 0:
            # Compute dead-on-eval and use the per-feature mask to drive resample
            _, _, pct_dead, ever_fired_eval = chunked_eval(
                sae, eval_acts_cpu, device, EVAL_CHUNK_SIZE
            )
            dead_mask = ~ever_fired_eval  # [d_dict] bool, True = dead on eval

            n_resampled = 0
            if RESAMPLING_STARTS_AT <= step + 1 <= RESAMPLING_ENDS_AT:
                # Use the current training batch as the source of high-loss tokens
                n_resampled = resample_dead_features(
                    sae, batch, dead_mask, max_resample=RESAMPLE_TOP_K
                )
                total_resampled += n_resampled

            history["dead_check_step"].append(step + 1)
            history["dead_check_pct"].append(float(pct_dead))
            history["resampled_per_check"].append(int(n_resampled))

            print(f"        [dead-feature check at step {step + 1}: "
                  f"{pct_dead:.1f}% dead on eval set "
                  f"({int(dead_mask.sum().item())}/{DICT_SIZE})]")
            if n_resampled > 0:
                print(f"        [resampled {n_resampled} dead features at step {step + 1}]")

        # Periodic checkpoint
        if (step + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f"sae_big_layer{LAYER}_step{step + 1}.safetensors"
            state_dict = {k: v.detach().cpu() for k, v in sae.state_dict().items()}
            save_file(state_dict, str(ckpt_path))
            print(f"        [checkpoint saved: {ckpt_path}]")

    # --------------------------------------------------------------- #
    # 4. Final eval + save
    # --------------------------------------------------------------- #
    print(f"[4/4] Final eval + save...")
    final_loss, final_ev, final_pct_dead, _ = chunked_eval(
        sae, eval_acts_cpu, device, EVAL_CHUNK_SIZE
    )

    state_dict = {k: v.detach().cpu() for k, v in sae.state_dict().items()}
    save_file(state_dict, str(SAE_FINAL))

    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)

    # Initial vs final train loss (for reduction %)
    initial_train_loss = history["train_recon_loss"][0]
    final_train_loss = history["train_recon_loss"][-1]
    reduction = (initial_train_loss - final_train_loss) / max(initial_train_loss, 1e-8)

    print()
    print("=" * 70)
    print("✅ Production SAE training v3 complete!")
    print("=" * 70)
    print(f"   Final train loss:    {final_train_loss:.4f}  "
          f"(reduction: {100 * reduction:.1f}%)")
    print(f"   Final eval loss:     {final_loss:.4f}")
    print(f"   Final eval EV:       {final_ev:.3f}")
    print(f"   Final L0:            {history['train_l0'][-1]:.1f}  (target: {K_SPARSITY})")
    print(f"   Dead on eval:        {final_pct_dead:.2f}% of {DICT_SIZE}")
    print(f"   Total resampled:     {total_resampled} reinitializations")
    print()
    print(f"   SAE weights:      {SAE_FINAL}")
    print(f"   Training history: {HISTORY_OUT}")
    print(f"   Checkpoints:      {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
