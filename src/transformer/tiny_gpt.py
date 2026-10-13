import torch
import torch.nn as nn

from .decoder_block import DecoderBlock


class TinyGpt(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_embedding: int,
            num_heads: int,
            d_head: int,
            d_hidden: int,
            block_size: int,
            num_layers: int
        ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_embedding = d_embedding
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_hidden = d_hidden
        self.block_size = block_size
        self.num_layers = num_layers        

        self.token_embedding = nn.Embedding(vocab_size, d_embedding)
        self.pos_embedding = nn.Embedding(block_size, d_embedding)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_embedding, num_heads, d_head, d_hidden, block_size)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_embedding)

        self.lm_head = nn.Linear(d_embedding, vocab_size, False)


    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = idx.shape

        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0) # TODO: dig into this

        x = self.token_embedding(idx) + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits
        

