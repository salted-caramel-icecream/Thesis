"""Mask sampling for JEPA-style pretraining on a hierarchical backbone.

Design (see docs/JEPA_GUIDE.md for the full rationale):

- The mask unit is one **stage-4 token** = 32x32 input pixels (224/32 = 7,
  so the mask lives on a 7x7 grid). Because every stage downsamples by 2x,
  a stage-4-aligned mask maps cleanly onto every intermediate grid — the
  Hiera "mask unit" idea applied to a conv pyramid.
- Masking is applied in **token space after the first patch embedding**
  (SimMIM-style learnable mask token), never by dropping tokens: PVT's conv
  stems, strided-conv SRA, and DWConv all need a contiguous 2D grid.
- Target selection follows I-JEPA multi-block sampling: a few contiguous
  rectangular blocks totalling roughly 40-60% of the image.
"""

from __future__ import annotations

import torch


def sample_block_mask(
    grid: int = 7,
    n_blocks: int = 4,
    block_area: tuple = (0.10, 0.20),
    aspect_ratio: tuple = (0.75, 1.5),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one multi-block mask on a ``grid x grid`` board.

    Returns a flattened bool tensor of shape ``(grid*grid,)`` — True = MASKED
    (prediction target). Guarantees at least one visible and one masked unit.
    """

    def _rand() -> float:
        return torch.rand((), generator=generator).item()

    def _randint(low: int, high: int) -> int:  # inclusive bounds
        if high <= low:
            return low
        return int(torch.randint(low, high + 1, (), generator=generator).item())

    mask = torch.zeros(grid, grid, dtype=torch.bool)
    for _ in range(n_blocks):
        area = (block_area[0] + _rand() * (block_area[1] - block_area[0])) * grid * grid
        ratio = aspect_ratio[0] + _rand() * (aspect_ratio[1] - aspect_ratio[0])
        h = max(1, min(grid, round((area * ratio) ** 0.5)))
        w = max(1, min(grid, round((area / ratio) ** 0.5)))
        top = _randint(0, grid - h)
        left = _randint(0, grid - w)
        mask[top : top + h, left : left + w] = True

    if mask.all():
        mask[0, 0] = False  # keep at least one visible unit
    if not mask.any():
        mask[grid // 2, grid // 2] = True  # keep at least one target unit
    return mask.flatten()


def sample_batch_masks(
    batch_size: int,
    grid: int = 7,
    n_blocks: int = 4,
    block_area: tuple = (0.10, 0.20),
    aspect_ratio: tuple = (0.75, 1.5),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Independent multi-block masks for a batch -> bool ``(B, grid*grid)``."""
    return torch.stack(
        [
            sample_block_mask(grid, n_blocks, block_area, aspect_ratio, generator)
            for _ in range(batch_size)
        ]
    )


def upsample_mask(mask: torch.Tensor, grid: int, target_hw: int) -> torch.Tensor:
    """Expand a ``(B, grid*grid)`` unit mask to a ``(B, target_hw*target_hw)``
    token mask (e.g. 7x7 units -> 56x56 stage-1 tokens; factor must divide)."""
    if target_hw % grid != 0:
        raise ValueError(f"target grid {target_hw} not divisible by mask grid {grid}")
    factor = target_hw // grid
    B = mask.shape[0]
    m = mask.view(B, grid, grid)
    m = m.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2)
    return m.reshape(B, target_hw * target_hw)
