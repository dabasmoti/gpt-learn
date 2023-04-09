import torch
import torch.nn as nn
from torch.nn import functional as F




class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.keys = nn.Linear(n_embed, head_size)
        self.queries = nn.Linear(n_embed, head_size)
        self.values = nn.Linear(n_embed, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.keyys(x)
        q = self.queries(x)

        wei = k @ q.transpose(-2, -1) * C**-0.5  # (B,T,C) @ (B,C,T) = (B,T,T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        v = self.values(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) = (B,T,C)
        return out


class NanoGpt(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, target=None):

        B, T = idx.shape

        tok_emb = self.token_embeddings_table(idx)  # (B,T,C)
        pos_emb = self.position_embeddings_table(torch.arange(T).to(DEVICE))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,embed_size)

        if target is not None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)

        return loss, logits




if __name__ == "__main__":
    from learn import chars, Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos, block_size, n_embed, vocab_size
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    with open("data/input.txt") as f:
        words = f.read()

    vocab_size = len(chars)
    block_size = 8
    n_embed = 64
    model = NanoGpt(vocab_size)
    print(model)
