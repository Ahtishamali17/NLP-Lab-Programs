"""
Program 25 : Sentence Embeddings using BERT and GPT-2                      [CO3]
Aim        : Get ONE fixed size vector for a whole sentence from transformer hidden states.
Algorithm  : 1) tokenize  2) forward pass -> last_hidden_state (batch, tokens, 768)
             3) BERT : mean pooling over real tokens (attention mask) and the [CLS] vector
                GPT-2: mean pooling and the LAST token vector (causal model, last token has seen everything)
             4) cosine similarity matrix  5) compare both models on similar / dissimilar sentences
Install    : pip install torch transformers
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer
from common import banner

sentences = ["The cat is sleeping on the sofa.",
             "A kitten is resting on the couch.",
             "The stock market crashed today.",
             "Share prices fell sharply this morning.",
             "I love playing cricket on Sundays."]
names = [f"S{i+1}" for i in range(len(sentences))]
for n, s in zip(names, sentences): print(n, ":", s)

def embed(model_name, pooling):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:                       # GPT-2 has no pad token
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name).eval()
    enc = tok(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state                       # (batch, seq, hidden)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    mean = (h * mask).sum(1) / mask.sum(1)                       # mean pooling ignoring padding
    if pooling == "cls":
        special = h[:, 0, :]                                     # [CLS] token
    else:                                                        # last non-pad token (GPT-2 pads on the right)
        last = enc["attention_mask"].sum(1) - 1
        special = h[torch.arange(h.size(0)), last]
    return mean.numpy(), special.numpy()

def cos_matrix(E):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return pd.DataFrame((E @ E.T).round(3), index=names, columns=names)

results = {}
for name, pooling in [("bert-base-uncased", "cls"), ("gpt2", "last")]:
    banner(f"{name}")
    mean, special = embed(name, pooling)
    print("Embedding shape:", mean.shape)
    print("First 6 values of S1 (mean pooled):", mean[0][:6].round(3))
    print(f"\nCosine similarity - mean pooling\n{cos_matrix(mean)}")
    print(f"\nCosine similarity - {pooling} pooling\n{cos_matrix(special)}")
    M = cos_matrix(mean).values
    results[name] = (M[0, 1], M[2, 3], M[0, 2], M[0, 4])

banner("Summary (mean pooling)")
print(f"{'model':<20}{'cat~kitten':<13}{'stock~shares':<14}{'cat~stock':<12}{'cat~cricket'}")
for k, v in results.items():
    print(f"{k:<20}{v[0]:<13.3f}{v[1]:<14.3f}{v[2]:<12.3f}{v[3]:.3f}")
print("\nParaphrases (S1~S2, S3~S4) should score higher than unrelated pairs.")
print("Note: raw BERT / GPT-2 vectors are anisotropic (all similarities are high);")
print("      for best semantic search use SBERT (pip install sentence-transformers).")
