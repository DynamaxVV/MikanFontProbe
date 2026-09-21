"""Small training-only helpers for the v13 font matching experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CategoryPrototypeBank(nn.Module):
    """Learnable normalized category centers used only as a training loss.

    The prototypes are deliberately not part of inference.  Their purpose is
    to make embeddings from different weights in one answer category occupy a
    compact region, while the saved encoder remains compatible with v11.
    """

    def __init__(self, category_count: int, embedding_dim: int = 64):
        super().__init__()
        if category_count < 2:
            raise ValueError("category_count must be at least 2")
        self.prototypes = nn.Parameter(torch.randn(category_count, embedding_dim))
        nn.init.normal_(self.prototypes, std=0.02)

    def logits(self, embedding: torch.Tensor, temperature: float = 0.12) -> torch.Tensor:
        centers = F.normalize(self.prototypes, p=2, dim=1)
        return embedding @ centers.T / temperature


def category_prototype_loss(bank: CategoryPrototypeBank,
                            embedding: torch.Tensor,
                            category: torch.Tensor,
                            temperature: float = 0.12) -> torch.Tensor:
    """Return cross entropy against normalized learned category centers."""
    return F.cross_entropy(bank.logits(embedding, temperature), category)
