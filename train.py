import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformer import TinyGpt

# TODO: this file was written by AI, rewrite it myself

# 1. Configuration & Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

batch_size = 32
block_size = 64
max_iters = 500
eval_interval = 100
learning_rate = 3e-4

# 2. Load Tokenizer & Dataset
tokenizer_path = "artifacts/tokenizer_f3a12f79.json"
data_path = "data/tiny-shakespeare.txt"

if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

tokenizer = Tokenizer.from_file(tokenizer_path)
vocab_size = tokenizer.get_vocab_size()

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Encode the entire text corpus into integer token IDs
encoded_tokens = tokenizer.encode(text).ids
data = torch.tensor(encoded_tokens, dtype=torch.long)
print(f"Dataset size: {len(data):,} tokens | Vocab size: {vocab_size}")

# Train/Validation split (90% train, 10% val)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Generates a small batch of inputs (x) and targets (y) shifted by 1."""
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# 3. Initialize Model
d_embedding = 128
num_heads = 4
d_head = 32
d_hidden = 256
num_layers = 3

model = TinyGpt(
    vocab_size=vocab_size,
    d_embedding=d_embedding,
    num_heads=num_heads,
    d_head=d_head,
    d_hidden=d_hidden,
    block_size=block_size,
    num_layers=num_layers
).to(device)

print(f"Model parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 4. Optimizer and Loss Function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 5. Training Loop
print("\n🚀 Starting training...")
model.train()

for step in range(max_iters):
    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass: get logits
    logits = model(xb)

    # Reshape for CrossEntropyLoss: (Batch * Seq_Len, Vocab_Size)
    loss = criterion(logits.view(-1, vocab_size), yb.view(-1))

    # Backward pass and optimization step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Print loss updates
    if step % eval_interval == 0 or step == max_iters - 1:
        print(f"Step {step:3d} | Loss: {loss.item():.4f}")

print("\n✨ Training finished successfully!")

import os

# Create an artifacts directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

# Save the model's state dictionary
model_path = "artifacts/tiny_gpt.pt"
torch.save(model.state_dict(), model_path)
print(f"\n💾 Model saved successfully to {model_path}")
