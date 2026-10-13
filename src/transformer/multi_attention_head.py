from typing import Optional

import torch
import torch.nn as nn


class MultiAttentionHead(nn.Module):
    causal_mask: Optional[torch.Tensor]


    def __init__(self, d_embedding: int, num_heads: int, d_head: int, block_size: int, is_causal: bool):
        super().__init__()

        self.num_heads = num_heads
        self.d_head = d_head
        self.d_embedding = d_embedding
        self.is_causal = is_causal

        self.query = nn.Linear(d_embedding, num_heads * d_head, False)
        self.key = nn.Linear(d_embedding, num_heads * d_head, False)
        self.value = nn.Linear(d_embedding, num_heads * d_head, False)

        self.out_proj = nn.Linear(num_heads * d_head, d_embedding, False)

        if is_causal:
            self.register_buffer("causal_mask", torch.tril(torch.ones((block_size, block_size))))

    def forward(
            self,
            x: torch.Tensor,
            k_arg: Optional[torch.Tensor] = None,
            v_arg: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        if (k_arg is None) ^ (v_arg is None):
            raise ValueError("Either k_arg or v_arg is provided, both must be provided.")

        batch_size, seq_len, _ = x.shape

        # (batch, block, d_head)
        q: torch.Tensor = self.query(x)

        if (k_arg is not None) and (v_arg is not None):
            k = self.key(k_arg)
            v = self.value(v_arg)
            kv_seq_length = k_arg.shape[1]
        else:
            k = self.key(x)
            v = self.value(x)
            kv_seq_length = seq_len

        # (batch, seq_len, num_heads, d_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head)
        # could be (batch, kv_seq_len, num_heads, d_head)
        k = k.view(batch_size, kv_seq_length, self.num_heads, self.d_head)
        v = v.view(batch_size, kv_seq_length, self.num_heads, self.d_head)

        # (batch, num_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        # could be (batch, num_heads, kv_seq_len, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (batch, num_heads, seq_len, seq_len)
        # or (batch, num_heads, seq_len, kv_seq_len)
        attention = q @ k.transpose(-2, -1) * (self.d_head ** -0.5)

        # Apply causal mask only for self-attention (generation)
        if self.is_causal and k_arg is None:
            assert self.causal_mask is not None
            
            attention = attention.masked_fill(
                self.causal_mask[:seq_len, :seq_len] == 0,
                float("-inf")
            )
        # (batch, num_heads, seq_len, seq_len)
        # or (batch, num_heads, seq_len, kv_seq_len)
        attention_scores = torch.softmax(attention, dim=-1)
        # (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, d_head) =
        # = (batch, num_heads, seq_len, d_head)
        # (same result for the kv_seq_len)
        self_attention_scores = attention_scores @ v

        multihead_attention = self.out_proj(
            self_attention_scores
                .transpose(1, 2)
                .reshape(batch_size, seq_len, self.num_heads * self.d_head)
        )

        return multihead_attention
