"""
Program 18 : Topic Modeling using Latent Semantic Analysis (LSA)          [CO2]
Aim        : Reduce the TF-IDF matrix with truncated SVD:  A  ~  U . S . V^T  to find latent concepts.
Algorithm  : 1) TF-IDF matrix  2) TruncatedSVD with k components  3) explained variance
             4) top terms per concept  5) documents in concept space  6) query matching in latent space
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from common import banner, load_news
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

df = load_news()
tv = TfidfVectorizer(stop_words="english")
A = tv.fit_transform(df.text)
terms = tv.get_feature_names_out()
print("TF-IDF matrix A:", A.shape)

K = 5
svd = TruncatedSVD(n_components=K, random_state=42)
Z = svd.fit_transform(A)                       # = U . S  (documents in concept space)

banner("Singular values and explained variance")
print("Singular values:", svd.singular_values_.round(3))
print("Explained variance ratio:", svd.explained_variance_ratio_.round(3), "total =", round(svd.explained_variance_ratio_.sum(), 3))

banner(f"Top terms of the {K} latent concepts (rows of V^T)")
for k, comp in enumerate(svd.components_):
    idx = np.abs(comp).argsort()[::-1][:8]
    print(f"Concept {k}: " + ", ".join(f"{terms[i]}({comp[i]:+.2f})" for i in idx))

banner("Documents in latent space (first 5)")
print(pd.DataFrame(Z[:5].round(3), columns=[f"C{k}" for k in range(K)]))
df["concept"] = Z.argmax(axis=1)
print("\nDominant concept vs category:\n", pd.crosstab(df.concept, df.category))

banner("Query matching in latent space")
Zn = Normalizer().fit_transform(Z)
for q in ["cricket team won the match", "new vaccine for patients", "bank interest rates and shares"]:
    qv = Normalizer().fit_transform(svd.transform(tv.transform([q])))
    sims = cosine_similarity(qv, Zn)[0]
    best = sims.argsort()[::-1][:2]
    print(f"\nQuery: {q}")
    for b in best:
        print(f"   {sims[b]:.2f} [{df.category[b]}] {df.text[b]}")
print("\nLSA vs LDA: LSA is linear algebra (SVD, values can be negative); LDA is probabilistic (topics are distributions).")
