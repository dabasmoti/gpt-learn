import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# # data loading
# def get_batch(split, batch_size=32, block_size=8):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == "train" else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i : i + block_size] for i in ix])
#     y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y


def get_batch(split, batch_size=32, block_size=8):
    data = train_ids if split == "train" else val_ids
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# # here are all the unique characters that occur in this text
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# # create a mapping from characters to integers
# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# # Train and test splits
# data = torch.tensor(encode(text), dtype=torch.long)
# n = int(0.9*len(data)) # first 90% will be train, rest val
# train_data = data[:n]
# val_data = data[n:]


n = len(text)
train_data = text[: int(n * 0.9)]
val_data = text[int(n * 0.9) :]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = np.array(enc.encode_ordinary(train_data))
val_ids = np.array(enc.encode_ordinary(val_data))
vocab_size = enc.max_token_value + 1
print(f"vocab size: {vocab_size}")
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
