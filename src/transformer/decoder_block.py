from typing import Optional

import torch
import torch.nn as nn

from .multi_attention_head import MultiAttentionHead
from .feed_forward_network import FeedForwardNetwork


class DecoderBlock(nn.Module):
    def __init__(
            self,
            d_embedding: int,
            num_heads: int,
            d_head: int,
            d_hidden: int,
            block_size: int
        ):
        super().__init__()

        self.d_embedding = d_embedding
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_hidden = d_hidden
        self.block_size = block_size

        self.ln1 = nn.LayerNorm(d_embedding)
        self.ln2 = nn.LayerNorm(d_embedding)
        self.ln3 = nn.LayerNorm(d_embedding)

        # causal self attention
        self.self_attention = MultiAttentionHead(
            d_embedding,
            num_heads,
            d_head,
            block_size,
            is_causal=True
        )

        # cross-attention
        self.cross_attention = MultiAttentionHead(
            d_embedding,
            num_heads,
            d_head,
            block_size,
            is_causal=False,
        )

        self.ffn = FeedForwardNetwork(d_embedding, d_hidden)


    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))

        if encoder_output is not None:
            x = x + self.cross_attention(self.ln2(x), k_arg=encoder_output, v_arg=encoder_output)

        x = x + self.ffn(self.ln3(x))

        return x
