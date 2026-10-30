"""
Program 13 : GloVe Embeddings loading and vector representation           [CO2]
Aim        : Use pre-trained GloVe vectors (trained on Wikipedia + Gigaword, 50 dimensions).
Setup      : EITHER place glove.6B.50d.txt in data/  (https://nlp.stanford.edu/projects/glove/)
             OR let gensim download 'glove-wiki-gigaword-50' automatically (internet needed, ~66 MB).
Algorithm  : 1) load vectors  2) inspect a word vector  3) cosine similarity  4) nearest neighbours
             5) analogy king - man + woman  6) sentence vector = average of word vectors  7) PCA plot
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import re
import numpy as np
from common import banner, load_embeddings, OUTPUT_DIR

glove = load_embeddings(allow_fallback=False)
print(f"Loaded {len(glove.key_to_index)} word vectors of dimension {glove.vector_size}")

banner("Vector representation of the word 'computer'")
v = glove["computer"]
print("shape :", v.shape)
print("values:", v[:10].round(3), "...")
print("L2 norm:", round(float(np.linalg.norm(v)), 3))

banner("Cosine similarity")
for a, b in [("king", "queen"), ("india", "delhi"), ("cat", "dog"), ("good", "bad"), ("car", "banana")]:
    print(f"similarity({a:<6},{b:<7}) = {glove.similarity(a, b):.3f}")

banner("Nearest neighbours")
for w in ["india", "python", "doctor", "happy"]:
    print(f"{w:<8}->", [x for x, _ in glove.most_similar(w, topn=6)])

banner("Analogies  (a - b + c)")
print("king - man + woman   =", glove.most_similar(positive=["king", "woman"], negative=["man"], topn=3))
print("paris - france + india =", glove.most_similar(positive=["paris", "india"], negative=["france"], topn=3))

banner("Sentence vector (average of word vectors)")
def sent_vec(s):
    ws = [w for w in re.findall(r"[a-z]+", s.lower()) if w in glove]
    return np.mean([glove[w] for w in ws], axis=0)
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
s1, s2, s3 = "the cat sat on the mat", "a kitten rested on the rug", "stock prices fell sharply today"
print(f"'{s1}' vs '{s2}' : {cos(sent_vec(s1), sent_vec(s2)):.3f}")
print(f"'{s1}' vs '{s3}' : {cos(sent_vec(s1), sent_vec(s3)):.3f}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    words = ["king", "queen", "man", "woman", "india", "delhi", "france", "paris", "cat", "dog", "car", "bus", "happy", "sad"]
    pts = PCA(2).fit_transform([glove[w] for w in words])
    plt.figure(figsize=(8, 6)); plt.scatter(pts[:, 0], pts[:, 1])
    for w, (x, y) in zip(words, pts): plt.annotate(w, (x, y))
    plt.title("GloVe 50d - PCA projection"); plt.savefig(OUTPUT_DIR / "13_glove_pca.png", dpi=120)
    print("\nPlot saved to outputs/13_glove_pca.png")
except Exception as e:
    print("plot skipped:", e)
