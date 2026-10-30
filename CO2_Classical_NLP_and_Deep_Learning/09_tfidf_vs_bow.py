"""
Program 9 : TF-IDF Implementation and comparison with BoW                  [CO2]
Aim       : Weight words by importance: TF-IDF = TF x IDF, so common words get low weights.
Algorithm : 1) tf = count/len(doc)   2) idf = log(N / df)   3) tfidf = tf*idf (manual)
            4) sklearn TfidfVectorizer (smooth idf + l2 norm)   5) compare with raw BoW counts
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import re
import numpy as np, pandas as pd
from common import banner, OUTPUT_DIR
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat",
        "the dog sat on the log",
        "the cat chased the dog",
        "machine learning is the future of the world"]
tok = lambda s: re.findall(r"[a-z]+", s.lower())
tokens = [tok(d) for d in docs]
vocab = sorted({w for t in tokens for w in t})
N = len(docs)
names = [f"D{i+1}" for i in range(N)]

banner("Manual TF-IDF")
tf = np.array([[t.count(w) / len(t) for w in vocab] for t in tokens])
df_ = np.array([sum(w in t for t in tokens) for w in vocab])
idf = np.log(N / df_)
tfidf_manual = tf * idf
print("IDF values:", dict(zip(vocab, idf.round(3))))
print(pd.DataFrame(tfidf_manual.round(3), columns=vocab, index=names))

banner("sklearn TfidfVectorizer (smooth idf, l2 norm)")
tv = TfidfVectorizer()
T = tv.fit_transform(docs)
tdf = pd.DataFrame(T.toarray().round(3), columns=tv.get_feature_names_out(), index=names)
print(tdf)

banner("Bag of Words counts for comparison")
cv = CountVectorizer()
B = cv.fit_transform(docs)
bdf = pd.DataFrame(B.toarray(), columns=cv.get_feature_names_out(), index=names)
print(bdf)

banner("BoW vs TF-IDF for document D1 :  'the cat sat on the mat'")
print(f"{'WORD':<10}{'BoW count':<12}{'TF-IDF':<10}")
for w in sorted(set(tok(docs[0]))):
    print(f"{w:<10}{bdf.loc['D1', w]:<12}{tdf.loc['D1', w]:<10}")
print("\n-> 'the' has the HIGHEST count in BoW but a LOW TF-IDF weight (it occurs in every document),")
print("   while rare words like 'mat' get higher weight: TF-IDF highlights discriminative terms.")

banner("Top-3 words per document by TF-IDF")
for d in names:
    print(d, tdf.loc[d].sort_values(ascending=False).head(3).to_dict())

tdf.to_csv(OUTPUT_DIR / "09_tfidf_matrix.csv"); print("\nSaved outputs/09_tfidf_matrix.csv")
