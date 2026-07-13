"""
sae_models.py — Sparse autoencoder models for RIBOSCOPE.

Currently implements:
- BatchTopKSAE: a sparse autoencoder with batch-level Top-K sparsity
  (Bussmann, Leask & Nanda, NeurIPS Mech-Interp Workshop 2024).

This module is imported by 04_train_sae.py and any future scripts that
need to construct or load SAE models.

Reference: https://hf.co/papers/2412.06410
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTopKSAE(nn.Module):
    """
    BatchTopK Sparse Autoencoder.

    Architecture (forward pass)
    ---------------------------
        x  ─→  encode (linear + ReLU)  ─→  BatchTopK  ─→  decode (linear)  ─→  x̂
        ↑                                                                       │
        └──────────────── reconstruction error ←────────────────────────────────┘

    Key idea
    --------
    BatchTopK applies the sparsity constraint at the *batch* level rather
    than per-sample. Given a batch of B samples and target sparsity k,
    we keep only the B*k largest activations across the entire batch; the
    rest are zeroed.

    Compared to TopK SAE (which keeps exactly k per sample), this gives:
    - Variable per-sample sparsity (some samples activate many features,
      others activate few — same average L0 = k)
    - Better reconstruction at equal average sparsity
    - No need for an explicit hyperparameter sweep over the L1 coefficient

    Args
    ----
    d_input:  dimension of the input activations (640 for RNA-FM,
              1280 for OmniGenome, 1280 for RiNALMo).
    d_dict:   number of dictionary features. Common choice: 8x or 16x of
              d_input. With d_input=640, 8x = 5120, 16x = 10240.
    k:        target average number of active features per sample.
              Common range: 16–128.
    """

    def __init__(self, d_input: int, d_dict: int, k: int):
        super().__init__()
        self.d_input = d_input
        self.d_dict = d_dict
        self.k = k

        # ------------------------------------------------------------- #
        # Encoder weights: maps d_input → d_dict
        # ------------------------------------------------------------- #
        # Initialize with Kaiming-uniform (a standard init for ReLU networks).
        # The shape is (d_input, d_dict) so that x @ W_enc has shape
        # (B, d_dict) — i.e., batch size B, then d_dict features.
        self.W_enc = nn.Parameter(torch.empty(d_input, d_dict))
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)

        # Encoder bias — added after the linear projection
        self.b_enc = nn.Parameter(torch.zeros(d_dict))

        # ------------------------------------------------------------- #
        # Decoder weights: maps d_dict → d_input
        # ------------------------------------------------------------- #
        # Standard SAE init: decoder = transpose of encoder, then renormalize
        # each row to unit L2 norm. This gives a sensible starting point
        # before the SAE diverges into different encoder/decoder roles.
        self.W_dec = nn.Parameter(self.W_enc.detach().clone().T.contiguous())
        self.normalize_decoder()

        # ------------------------------------------------------------- #
        # Pre-encoder bias (a.k.a. "decoder bias")
        # ------------------------------------------------------------- #
        # Subtracted from x before the encoder. Often called the "decoder
        # bias" in the SAE literature because it doubles as the
        # reconstruction's bias term — it's the SAE's learned estimate of
        # the activation distribution's center.
        self.b_dec = nn.Parameter(torch.zeros(d_input))

    # ----------------------------------------------------------------- #
    # Decoder normalization
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """
        Normalize each decoder row to have unit L2 norm.

        Why: SAE features have a scale ambiguity — you can multiply an
        encoder column by 2 and divide the corresponding decoder row by 2,
        and the network's input/output is identical but the feature
        magnitudes have changed. To fix the ambiguity, we constrain
        decoder rows to unit norm. This also empirically helps with
        feature interpretation.

        We call this after every gradient step (in the training loop).
        """
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    # ----------------------------------------------------------------- #
    # Encode / decode primitives
    # ----------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map x [B, d_input] → pre-sparse features [B, d_dict].

        Returns the post-ReLU activations, NOT yet sparsified.
        """
        # Subtract pre-encoder bias (centering the input)
        x_centered = x - self.b_dec
        return F.relu(x_centered @ self.W_enc + self.b_enc)

    def batch_topk(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply Top-K at the batch level.

        h:        post-ReLU activations of shape [B, d_dict]
        returns:  same shape, with all but the top B*k entries zeroed.
        """
        B, D = h.shape
        n_keep = B * self.k

        # Edge case: k is so large that every entry would be kept.
        # In that case, no need to compute anything — return as-is.
        if n_keep >= B * D:
            return h

        # Flatten across batch and feature dims, find top values, zero rest
        flat = h.flatten()
        # `sorted=False` is faster and we don't need ranking, just selection
        _, topk_idx = flat.topk(n_keep, sorted=False)

        mask = torch.zeros_like(flat)
        mask[topk_idx] = 1.0

        return (flat * mask).reshape(B, D)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """
        Map sparse features [B, d_dict] → reconstructed activations [B, d_input].
        """
        return h @ self.W_dec + self.b_dec

    # ----------------------------------------------------------------- #
    # Full forward pass
    # ----------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full SAE forward pass.

        Args:
            x: input activations [B, d_input]

        Returns:
            x_hat:    reconstruction [B, d_input]
            h_sparse: sparse feature activations [B, d_dict]
                      (after BatchTopK; most entries are zero)
        """
        h_pre = self.encode(x)
        h_sparse = self.batch_topk(h_pre)
        x_hat = self.decode(h_sparse)
        return x_hat, h_sparse
