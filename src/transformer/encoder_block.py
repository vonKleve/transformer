import torch
import torch.nn as nn

from .multi_attention_head import MultiAttentionHead
from .feed_forward_network import FeedForwardNetwork


class EncoderBlock(nn.Module):
    def __init__(self,
                 d_embedding: int,
                 num_heads: int,
                 d_head: int,
                 d_hidden: int,
                 block_size: int
                ):
        """
        Uses more modern take on Transformer architecture with pre-LN.
        """
        super().__init__()

        self.d_embedding = d_embedding
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_hidden = d_hidden
        self.block_size = block_size

        self.ln1 = nn.LayerNorm(d_embedding)
        self.ln2 = nn.LayerNorm(d_embedding)

        self.attention = MultiAttentionHead(
            d_embedding,
            num_heads,
            d_head,
            block_size,
            is_causal=False
        )

        self.ffn = FeedForwardNetwork(
            d_embedding,
            d_hidden
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

