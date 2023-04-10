import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open("data/input.txt") as f:
    words = f.read()



chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


def build_dataset(words):
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

# import random
# random.seed(42)
# random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))


# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


chars = sorted(list(set(chars)))
vocab_size = len(chars)
block_size = 8
n_embed = 64
