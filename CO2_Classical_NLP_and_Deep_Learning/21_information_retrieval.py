"""
Program 21 : Information Retrieval system with ranking using TF-IDF        [CO2]
Aim        : A mini search engine: inverted index + TF-IDF weighting + cosine ranking.
Algorithm  : 1) preprocess (lowercase, stop-word removal, Porter stemming)
             2) build inverted index  term -> {doc: tf}   3) idf = log(N/df)
             4) query vector  5) score = cosine(query, doc)  6) top-k results + Precision@k
Run        : python 21_information_retrieval.py              (demo queries)
             python 21_information_retrieval.py --interactive (type your own queries)
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import math, re
from collections import Counter, defaultdict
from common import banner, ensure_nltk, load_news
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ensure_nltk()
stop, stem = set(stopwords.words("english")), PorterStemmer().stem
df = load_news()
docs = df.text.tolist()

def preprocess(t):
    return [stem(w) for w in re.findall(r"[a-z]+", t.lower()) if w not in stop]

banner("Building the inverted index")
index = defaultdict(dict)                       # term -> {doc_id: term frequency}
doc_len = {}
for i, d in enumerate(docs):
    toks = preprocess(d); doc_len[i] = len(toks)
    for term, f in Counter(toks).items():
        index[term][i] = f
N = len(docs)
idf = {t: math.log(N / len(p)) + 1 for t, p in index.items()}
norms = defaultdict(float)
for t, post in index.items():
    for i, f in post.items():
        norms[i] += ((f / doc_len[i]) * idf[t]) ** 2
norms = {i: math.sqrt(v) for i, v in norms.items()}
print(f"{N} documents, {len(index)} unique terms")
print("Example posting list  'cricket':", dict(list(index[stem('cricket')].items())[:5]))

def search(query, k=5):
    q = Counter(preprocess(query))
    qw = {t: (f / sum(q.values())) * idf[t] for t, f in q.items() if t in idf}
    qn = math.sqrt(sum(w * w for w in qw.values())) or 1
    scores = defaultdict(float)
    for t, w in qw.items():                                     # only visit docs containing a query term
        for i, f in index[t].items():
            scores[i] += w * (f / doc_len[i]) * idf[t]
    ranked = [(i, s / (qn * norms[i])) for i, s in scores.items()]
    return sorted(ranked, key=lambda x: -x[1])[:k]

def show(query, expected=None, k=5):
    print(f"\nQUERY: {query}")
    res = search(query, k)
    for r, (i, s) in enumerate(res, 1):
        print(f"  {r}. score={s:.3f} [{df.category[i]:<10}] {docs[i]}")
    if expected:
        p = sum(df.category[i] == expected for i, _ in res) / max(len(res), 1)
        print(f"  Precision@{k} for category '{expected}' = {p:.2f}")
        return p

if "--interactive" in sys.argv:
    while True:
        q = input("\nsearch> ").strip()
        if not q: break
        show(q)
else:
    banner("Search results")
    tests = [("cricket team won the final match", "sports"), ("new vaccine and cancer treatment", "health"),
             ("machine learning platform for cloud", "technology"), ("election and national budget", "politics"),
             ("bank interest rates and profits", "business")]
    ps = [show(q, c) for q, c in tests]
    banner("Evaluation")
    print(f"Mean Precision@5 over {len(tests)} queries = {sum(ps)/len(ps):.2f}")
