import torch
from tokenizers import Tokenizer
from transformer import TinyGpt


# TODO: this file was written by AI, rewrite it myself

# 1. Load Tokenizer
tokenizer_path = "artifacts/tokenizer_f3a12f79.json"
model_path = "artifacts/tiny_gpt.pt"

tokenizer = Tokenizer.from_file(tokenizer_path)

# 2. Configuration & Hyperparameters (Must match training!)
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = tokenizer.get_vocab_size()  # Adjust if your tokenizer vocab size differs
d_embedding = 128
num_heads = 4
d_head = 32
d_hidden = 256
block_size = 64
num_layers = 3

# Trained Model

model = TinyGpt(
    vocab_size=vocab_size,
    d_embedding=d_embedding,
    num_heads=num_heads,
    d_head=d_head,
    d_hidden=d_hidden,
    block_size=block_size,
    num_layers=num_layers
).to(device)

# Load the saved weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully for generation!\n")

# 3. Generation Function
@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8) -> str:
    # Encode the text prompt into token IDs
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's block_size
        idx_cond = idx[:, -block_size:]

        # Forward pass to get logits
        logits = model(idx_cond)

        # Focus on the very last time-step and scale by temperature
        logits = logits[:, -1, :] / temperature

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample from the probability distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the new token to the sequence
        idx = torch.cat((idx, next_token), dim=1)

    # Decode tokens back to a readable string
    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)

# 4. Prompt the Model
prompt_text = "ROMEO:"
print(f"Prompt: {prompt_text}\n" + "-"*40)

output_text = generate(model, tokenizer, prompt=prompt_text, max_new_tokens=150, temperature=0.8)
print(output_text)
