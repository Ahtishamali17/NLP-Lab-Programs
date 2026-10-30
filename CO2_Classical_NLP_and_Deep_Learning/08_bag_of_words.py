"""
Program 8 : Bag-of-Words (BoW) vectorization and representation           [CO2]
Aim       : Represent documents as word-count vectors.
Algorithm : 1) tokenize + build vocabulary  2) count words per document (manual)
            3) verify with sklearn CountVectorizer  4) binary BoW, n-gram BoW  5) sparsity
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import re
import numpy as np, pandas as pd
from common import banner, OUTPUT_DIR
from sklearn.feature_extraction.text import CountVectorizer

docs = ["NLP is fun and NLP is useful",
        "Machine learning is fun",
        "Deep learning is a part of machine learning",
        "Students love NLP lab programs"]

banner("Manual Bag-of-Words")
tok = lambda s: re.findall(r"[a-z]+", s.lower())
vocab = sorted({w for d in docs for w in tok(d)})
print("Vocabulary (%d words):" % len(vocab), vocab)
manual = np.array([[tok(d).count(w) for w in vocab] for d in docs])
print(pd.DataFrame(manual, columns=vocab, index=[f"D{i+1}" for i in range(len(docs))]))

banner("sklearn CountVectorizer")
cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")   # keep 1-letter words like "a"
X = cv.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out(), index=[f"D{i+1}" for i in range(len(docs))])
print(df)
print("\nManual result equals sklearn result:", (manual == df[vocab].values).all())

banner("Binary BoW (presence / absence)")
print(pd.DataFrame(CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b").fit_transform(docs).toarray(), columns=cv.get_feature_names_out()))

banner("Bag of Bigrams (ngram_range=(1,2))")
cv2 = CountVectorizer(ngram_range=(1, 2), stop_words="english")
X2 = cv2.fit_transform(docs)
print("Feature count:", X2.shape[1]); print(cv2.get_feature_names_out())

banner("Encoding a NEW sentence with the learned vocabulary")
new = ["NLP lab is fun and the lab is useful"]
print(dict(zip(cv.get_feature_names_out(), cv.transform(new).toarray()[0])))

banner("Sparsity")
print(f"Matrix shape {X.shape}, non-zero cells {X.nnz}, sparsity = {100*(1 - X.nnz/np.prod(X.shape)):.1f}%")
df.to_csv(OUTPUT_DIR / "08_bow_matrix.csv"); print("Saved outputs/08_bow_matrix.csv")
