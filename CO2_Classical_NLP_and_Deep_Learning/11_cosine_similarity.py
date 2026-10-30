"""
Program 11 : Cosine Similarity computation between text documents         [CO2]
Aim        : Measure how similar two documents are: cos(theta) = (A . B) / (|A| |B|).
Algorithm  : 1) TF-IDF vectors  2) manual cosine with numpy  3) sklearn cosine_similarity matrix
             4) most similar pair  5) rank documents for a query
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from common import banner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = ["Machine learning is a method of data analysis that automates model building",
        "Deep learning is a subset of machine learning based on neural networks",
        "Cricket is a bat and ball game played between two teams of eleven players",
        "The Indian cricket team won the match against Australia in the final",
        "The stock market closed higher as investors bought bank shares"]
names = [f"D{i+1}" for i in range(len(docs))]
for n, d in zip(names, docs):
    print(n, ":", d)

tv = TfidfVectorizer(stop_words="english")
X = tv.fit_transform(docs).toarray()

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

banner("Manual cosine similarity")
print("cos(D1, D2) =", round(cosine(X[0], X[1]), 4), " (both about machine learning)")
print("cos(D3, D4) =", round(cosine(X[2], X[3]), 4), " (both about cricket)")
print("cos(D1, D3) =", round(cosine(X[0], X[2]), 4), " (different topics)")

banner("Similarity matrix (sklearn)")
S = cosine_similarity(X)
print(pd.DataFrame(S.round(3), index=names, columns=names))

banner("Most similar pair")
S2 = S.copy(); np.fill_diagonal(S2, -1)
i, j = np.unravel_index(S2.argmax(), S2.shape)
print(f"{names[i]} and {names[j]} with similarity {S2[i, j]:.3f}")

banner("Query ranking")
query = "neural networks and machine learning models"
scores = cosine_similarity(tv.transform([query]), tv.transform(docs))[0]
print("Query:", query)
for k in scores.argsort()[::-1]:
    print(f"  {names[k]}  score={scores[k]:.3f}  {docs[k][:60]}")
print("\nCosine distance = 1 - similarity ; value 1 -> same direction, 0 -> nothing in common.")
