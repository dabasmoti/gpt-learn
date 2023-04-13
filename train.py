import torch
from model import Gpt
from data_utils import (
    get_batch,
    device,
    vocab_size,
)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    # hyperparameters
    batch_size = 16  # how many independent sequences will we process in parallel?
    block_size = 32  # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 100
    learning_rate = 1e-3
    eval_iters = 200
    n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0
    # ------------
    torch.manual_seed(1337)
    model = Gpt(vocab_size, n_embd, n_layer, n_head, block_size, dropout).to(device)
    # train model
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for i in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if i % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
