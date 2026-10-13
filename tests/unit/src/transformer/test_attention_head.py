import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.attention_head import AttentionHead


# --- Unit Tests ---

def test_output_shape():
    batch_size = 2
    block_size = 8
    d_embedding = 64  # Asymmetric to d_head to catch dimension mix-ups
    d_head = 16

    head = AttentionHead(d_embedding, d_head, block_size, is_causal=False)
    dummy_input = torch.randn(batch_size, block_size, d_embedding)
    
    output = head(dummy_input)
    expected_shape = (batch_size, block_size, d_head)
    
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"


def test_overfit_tiny_batch():
    d_embedding, d_head, block_size = 32, 16, 8
    head = AttentionHead(d_embedding, d_head, block_size, is_causal=False)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    
    x = torch.randn(2, block_size, d_embedding)
    target = torch.randn(2, block_size, d_head)
    
    initial_loss = None
    for step in range(30):
        optimizer.zero_grad()
        out = head(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
            
    final_loss = loss.item() # type: ignore
    assert initial_loss is not None, "Initial loss should not be None in this context."
    assert final_loss < initial_loss, "Model failed to learn on a tiny batch (gradients might be broken)."