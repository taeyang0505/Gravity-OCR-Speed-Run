"""Track 2 All-in-VRAM Dynamic Vocab Slicer & 1-Hop Halo Patch Pruner for SGLang glm_ocr.py.

Integrated into `GlmOcrForConditionalGeneration` via `tools/patch_sglang_glm_ocr.py`:
- Stage 2: Batch vision-vocab dot-product (Q * K^T) dynamic vocabulary slicing + Cauchy-Schwarz
           mathematical lossless gate & rescue (100% greedy parity with full 59,283-row lm_head).
- Stage 3: 1-Hop Halo preserving 28x28 blank paper patch pruning while retaining exact 3D RoPE
           `(t, row, col)` coordinates.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_VOCAB_SIZE = 59283
BLANK_SPREAD = int(os.environ.get("TRACK2_BLANK_SPREAD", "16"))
VISUAL_TOP_K = int(os.environ.get("TRACK2_VOCAB_TOPK", "2048"))


class Track2AllInVramAccelerator:
    """L40S 48GB VRAM-resident blank patch pruner & dynamic vocab slicer with Cauchy-Schwarz rescue."""

    def __init__(
        self,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        slice_dir: str | Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or lm_head_weight.device
        self.stage = int(os.environ.get("TRACK2_STAGE", "3"))
        self.enable_rescue = os.environ.get("TRACK2_RESCUE", "1") == "1"
        self.lm_head_weight = lm_head_weight  # [V, 1536] BF16, pinned in VRAM
        self.vocab_size = int(lm_head_weight.shape[0])

        resolved_dir = Path(
            slice_dir
            or os.environ.get("TRACK2_SLICE_DIR")
            or Path(__file__).resolve().parent.parent / "vocab_slices"
        )

        # 1. [Untimed precompute] Normalized embedding keys E_norm_T ([1536, V], BF16) + row L2 norms
        with torch.no_grad():
            e_norm = F.normalize(embed_weight.detach().to(torch.float32), p=2, dim=-1)
            self.e_norm_t = e_norm.to(embed_weight.dtype).t().contiguous()
            self.head_row_norms = torch.norm(lm_head_weight.detach().to(torch.float32), p=2, dim=-1)

        # 2. Task-specific Byte-BPE Safe core vocabulary indices pinned in VRAM
        self.core_ids: dict[str, torch.Tensor] = {}
        for task, fname, default_k in (
            ("table", "core_table_ids.npy", 16384),
            ("text", "core_text_ids.npy", 16384),
            ("formula", "core_formula_ids.npy", 8192),
        ):
            fpath = resolved_dir / fname
            if fpath.is_file():
                arr = np.load(fpath).astype(np.int64)
                arr = arr[(arr >= 0) & (arr < self.vocab_size)]
                self.core_ids[task] = torch.from_numpy(arr).to(device=self.device, dtype=torch.long)
            else:
                self.core_ids[task] = torch.arange(min(default_k, self.vocab_size), device=self.device, dtype=torch.long)

        # 3. Pre-allocated zero-cudaMalloc boolean mask buffer
        self.vocab_mask_buf = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        self.active_vocab_ids: torch.Tensor | None = None
        self.w_rest_max: float = float(self.head_row_norms.max().item())

    @torch.no_grad()
    def prune_blank_patches_with_1hop_halo(
        self,
        patch_pixels: torch.Tensor,
        grid_thw: tuple[int, int, int],
        pos_ids_3d: torch.Tensor,
        spread_thresh: int = BLANK_SPREAD,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[Stage 3] Drop blank 28x28 merge groups outside the 1-hop ink halo while keeping original 3D RoPE coords."""
        if self.stage < 3:
            return patch_pixels, pos_ids_3d

        _, h_grid, w_grid = grid_thw
        if h_grid % 2 != 0 or w_grid % 2 != 0:
            return patch_pixels, pos_ids_3d
        h_merge, w_merge = h_grid // 2, w_grid // 2
        n_groups = h_merge * w_merge
        if n_groups <= 32 or patch_pixels.shape[0] != n_groups * 4:
            return patch_pixels, pos_ids_3d

        # Spread per 28x28 merge group (works for both normalized and [0, 1] pixel tensors)
        grouped = patch_pixels.view(n_groups, -1)
        scale = 255.0 if float(grouped.abs().max().item()) <= 1.5 else 68.0
        spread = (grouped.max(dim=-1).values - grouped.min(dim=-1).values) * scale
        ink_grid = (spread > spread_thresh).view(1, 1, h_merge, w_merge).to(torch.float32)

        # 3x3 max-pool (1-hop dilation) protects glyph edges, formula dots, and thin table rules
        keep_halo = F.max_pool2d(ink_grid, kernel_size=3, stride=1, padding=1).view(-1) > 0.5
        keep_halo[0] = True
        keep_halo[-1] = True

        kept = int(keep_halo.sum().item())
        if kept >= int(n_groups * 0.92) or kept < 4:
            return patch_pixels, pos_ids_3d

        keep_patches = keep_halo.repeat_interleave(4)
        return patch_pixels[keep_patches], pos_ids_3d[keep_patches]

    @torch.no_grad()
    def update_batch_active_vocab(
        self,
        vision_embeds: torch.Tensor,
        tasks: str | Iterable[str] = "text",
        top_k: int = VISUAL_TOP_K,
    ) -> None:
        """[Stage 2+] Compute V_batch @ E_norm_T once and switch the active vocabulary slice in-place."""
        if self.stage < 2:
            self.active_vocab_ids = None
            return

        if isinstance(tasks, str):
            task_list = [tasks]
        else:
            task_list = list(tasks) or ["text"]

        # 1. Single cuBLAS GEMM: V_batch [M, 1536] @ E_norm_T [1536, V]
        v_flat = vision_embeds.reshape(-1, vision_embeds.shape[-1])
        v_norm = F.normalize(v_flat.to(torch.float32), p=2, dim=-1).to(self.e_norm_t.dtype)
        sim_scores = torch.matmul(v_norm, self.e_norm_t)
        max_sim_per_token = sim_scores.max(dim=0).values
        k = min(top_k, self.vocab_size)
        visual_topk_ids = torch.topk(max_sim_per_token, k=k, largest=True).indices

        # 2. Union of task core slices U visual Top-K into pre-allocated boolean buffer (zero alloc)
        self.vocab_mask_buf.zero_()
        for t in set(task_list):
            base_ids = self.core_ids.get(t, self.core_ids["text"])
            self.vocab_mask_buf.index_fill_(0, base_ids, True)
        self.vocab_mask_buf.index_fill_(0, visual_topk_ids, True)

        self.active_vocab_ids = torch.nonzero(self.vocab_mask_buf, as_tuple=False).squeeze(-1)

        # Update Cauchy-Schwarz upper bound W_rest_max over cut rows
        rest_mask = ~self.vocab_mask_buf
        if torch.any(rest_mask):
            self.w_rest_max = float(self.head_row_norms[rest_mask].max().item())
        else:
            self.w_rest_max = 0.0

    @torch.no_grad()
    def compute_logits_lossless(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fast sliced GEMM on V_active + Cauchy-Schwarz exact greedy rescue on uncertain rows (<5%)."""
        if self.stage < 2 or self.active_vocab_ids is None:
            return F.linear(hidden_states, self.lm_head_weight)

        active_ids = self.active_vocab_ids
        w_active = self.lm_head_weight.index_select(0, active_ids)
        sub_logits = F.linear(hidden_states, w_active)

        full_logits = torch.full(
            (*hidden_states.shape[:-1], self.vocab_size),
            float("-inf"),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        full_logits.index_copy_(-1, active_ids, sub_logits)

        if not self.enable_rescue or self.w_rest_max <= 0.0:
            return full_logits

        # Cauchy-Schwarz gate: L_best >= ||x||_2 * W_rest_max guarantees no cut row can win
        flat_x = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_full = full_logits.reshape(-1, self.vocab_size)
        l_best = sub_logits.reshape(-1, sub_logits.shape[-1]).max(dim=-1).values.to(torch.float32)
        x_norm = torch.norm(flat_x.to(torch.float32), p=2, dim=-1)

        uncertain_mask = l_best < (x_norm * self.w_rest_max)
        if torch.any(uncertain_mask):
            rescue_idx = torch.nonzero(uncertain_mask, as_tuple=False).squeeze(-1)
            x_rescue = flat_x.index_select(0, rescue_idx)
            flat_full.index_copy_(0, rescue_idx, F.linear(x_rescue, self.lm_head_weight))

        return full_logits
