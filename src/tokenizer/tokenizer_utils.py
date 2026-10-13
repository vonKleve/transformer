import torch


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    # get random starting indexes
    ix = torch.randint(len(data) - batch_size, (batch_size,))

    inputs = torch.stack([data[i : i + block_size] for i in ix])
    outputs = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    return inputs, outputs
