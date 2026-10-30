"""
Program 14 : Text Similarity using Word Mover's Distance (WMD)             [CO2]
Aim        : WMD = minimum "travel cost" to move the words of document A onto the words of document B
             in embedding space (an Earth-Mover problem).  Works even when documents share NO words.
Algorithm  : 1) load embeddings  2) normalised bag of words (nBOW) of each document
             3) cost matrix = Euclidean distance between word vectors
             4) solve the transport linear program (scipy linprog)  5) compare with cosine on BoW
Note       : implemented from scratch with scipy, so no extra package (POT / pyemd) is required.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import re
import numpy as np
from scipy.optimize import linprog
from common import banner, load_embeddings
from nltk.corpus import stopwords
from common import ensure_nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ensure_nltk()
wv = load_embeddings(allow_fallback=True)
stop = set(stopwords.words("english"))

def prep(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop and w in wv]

def wmd(doc1, doc2):
    t1, t2 = prep(doc1), prep(doc2)
    u1, u2 = sorted(set(t1)), sorted(set(t2))
    d1 = np.array([t1.count(w) for w in u1], float); d1 /= d1.sum()
    d2 = np.array([t2.count(w) for w in u2], float); d2 /= d2.sum()
    C = np.array([[np.linalg.norm(wv.get_vector(a, norm=True) - wv.get_vector(b, norm=True)) for b in u2] for a in u1])   # cost matrix
    n, m = C.shape
    A_eq, b_eq = [], []
    for i in range(n):                                   # outgoing flow of word i = d1[i]
        row = np.zeros((n, m)); row[i, :] = 1; A_eq.append(row.ravel()); b_eq.append(d1[i])
    for j in range(m):                                   # incoming flow to word j = d2[j]
        row = np.zeros((n, m)); row[:, j] = 1; A_eq.append(row.ravel()); b_eq.append(d2[j])
    res = linprog(C.ravel(), A_eq=np.array(A_eq), b_eq=b_eq, bounds=(0, None), method="highs")
    return res.fun

s0 = "Obama speaks to the media in Illinois"
cands = ["The President greets the press in Chicago",     # same meaning, no common words
         "The band gave a concert in Japan",              # different meaning
         "Obama speaks to the media in Illinois",         # identical
         "Students write code in the laboratory"]

banner("Word Mover's Distance (lower = more similar)")
print("Reference:", s0)
scores = []
for c in cands:
    d = wmd(s0, c); scores.append(d)
    print(f"WMD = {d:7.4f}   {c}")
print("\nClosest document:", cands[int(np.argmin(scores))])

banner("Comparison with BoW cosine similarity (higher = more similar)")
bow = CountVectorizer(stop_words="english").fit([s0] + cands)
cs = cosine_similarity(bow.transform([s0]), bow.transform(cands))[0]
for c, s in zip(cands, cs):
    print(f"cosine = {s:.3f}   {c}")
print("-> Cosine on BoW gives 0 to the paraphrase (no shared words) but WMD still sees it as closer than the unrelated sentences.")

banner("Cross-check with gensim's built-in wmdistance (needs POT)")
try:
    print("gensim WMD(s0, paraphrase) =", round(wv.wmdistance(prep(s0), prep(cands[0])), 4))
except Exception as e:
    print("skipped (pip install POT to enable):", type(e).__name__)
