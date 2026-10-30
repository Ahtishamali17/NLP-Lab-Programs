"""
Program 12 : Word2Vec word embeddings using gensim on a custom corpus      [CO2]
Aim        : Learn dense word vectors so that words used in similar contexts get similar vectors.
Algorithm  : 1) tokenize data/corpus.txt  2) train CBOW (sg=0) and Skip-gram (sg=1)
             3) inspect vectors, similarity, most_similar, odd-one-out  4) save model, PCA plot
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import banner, load_corpus, OUTPUT_DIR
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

sentences = [simple_preprocess(s) for s in load_corpus()]
print(f"{len(sentences)} sentences, e.g. {sentences[0]}")

params = dict(vector_size=50, window=4, min_count=1, workers=1, epochs=300, seed=42)
cbow = Word2Vec(sentences, sg=0, **params)
skip = Word2Vec(sentences, sg=1, **params)
print("Vocabulary size:", len(cbow.wv))

banner("Vector of the word 'king' (first 10 of 50 dimensions)")
print(cbow.wv["king"][:10].round(3))

for name, m in [("CBOW", cbow), ("Skip-gram", skip)]:
    banner(f"{name}: most similar words")
    for w in ["king", "cricket", "python", "doctor"]:
        print(f"{w:<8}->", [(x, round(s, 2)) for x, s in m.wv.most_similar(w, topn=4)])

banner("Similarity scores (Skip-gram)")
for a, b in [("king", "queen"), ("man", "woman"), ("cricket", "football"), ("king", "doctor"), ("python", "java")]:
    print(f"similarity({a}, {b}) = {skip.wv.similarity(a, b):.3f}")

banner("Odd one out")
print(skip.wv.doesnt_match(["king", "queen", "prince", "cricket"]))

banner("Analogy: king - man + woman")
print(skip.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3))
print("(a corpus of 40 sentences is tiny, so analogies are noisy - see Program 13 for real GloVe vectors)")

skip.save(str(OUTPUT_DIR / "word2vec_custom.model"))
print("\nModel saved to outputs/word2vec_custom.model")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    words = [w for w in ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl", "cricket", "football",
                         "players", "python", "java", "code", "doctor", "nurse", "hospital", "bank", "market", "shares"] if w in skip.wv]
    pts = PCA(n_components=2).fit_transform([skip.wv[w] for w in words])
    plt.figure(figsize=(8, 6)); plt.scatter(pts[:, 0], pts[:, 1])
    for w, (x, y) in zip(words, pts):
        plt.annotate(w, (x, y))
    plt.title("Word2Vec (skip-gram) embeddings - PCA projection"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_word2vec_pca.png", dpi=120); print("Plot saved to outputs/12_word2vec_pca.png")
except Exception as e:
    print("plot skipped:", e)
