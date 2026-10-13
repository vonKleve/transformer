from typing import Optional

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, d_embedding: int, d_head: int, block_size: int, is_causal: bool):
        super().__init__()

        self.d_head = d_head # d_q = d_k = d_v
        self.is_causal = is_causal
        self.causal_mask: Optional[torch.Tensor] = None

        self.query = nn.Linear(d_embedding, d_head, False)
        self.key = nn.Linear(d_embedding, d_head, False)
        self.value = nn.Linear(d_embedding, d_head, False)

        if is_causal:
            self.register_buffer("causal_mask", torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, block_size, _ = x.shape

        q: torch.Tensor = self.query(x) # (batch, block, d_head)
        k: torch.Tensor = self.key(x) # (batch, block, d_head)

        # (batch, block, d_head) x (batch, d_head, block) = (batch, block, block)
        attention: torch.Tensor = q @ k.transpose(-2, -1) * (self.d_head ** -0.5)

        if self.is_causal:
            assert self.causal_mask is not None
            
            # apply causal mask
            attention = attention.masked_fill(
                self.causal_mask[:block_size, :block_size] == 0,
                float("-inf")
            )

        # (batch, block, block)
        attention_scores = torch.softmax(attention, dim=-1) # for each query -> softmax
        # (batch, block, block) x (batch, block, d_head) = (batch, block, d_head)
        self_attention_scores = attention_scores @ self.value(x)

        return self_attention_scores
