"""
Program 10 : N-Gram Model (uni-, bi-, tri-gram) generation from corpus     [CO2]
Aim        : Build n-gram frequency tables, probabilities, next-word prediction and text generation.
Algorithm  : 1) tokenize corpus (<s> </s> markers)  2) count 1,2,3-grams  3) P(w|prev) with Laplace smoothing
             4) predict next word  5) generate sentences  6) perplexity of a test sentence
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import math, random, re
from collections import Counter, defaultdict
from common import banner, load_corpus
from nltk.util import ngrams

corpus = load_corpus()
sents = [["<s>"] + re.findall(r"[a-z]+", s.lower()) + ["</s>"] for s in corpus]
words = [w for s in sents for w in s]
V = len(set(words))
print(f"Corpus: {len(corpus)} sentences, {len(words)} tokens, vocabulary size {V}")

uni = Counter(words)
bi = Counter(g for s in sents for g in ngrams(s, 2))
tri = Counter(g for s in sents for g in ngrams(s, 3))

banner("Top unigrams");  print(uni.most_common(10))
banner("Top bigrams");   print(bi.most_common(10))
banner("Top trigrams");  print(tri.most_common(8))

def p_bigram(prev, w):                    # Laplace (add-one) smoothing
    return (bi[(prev, w)] + 1) / (uni[prev] + V)

def p_trigram(a, b, w):
    return (tri[(a, b, w)] + 1) / (bi[(a, b)] + V)

banner("Bigram probabilities")
for prev, w in [("the", "king"), ("the", "queen"), ("machine", "learning"), ("the", "zebra")]:
    print(f"P({w:<9}| {prev:<8}) = {p_bigram(prev, w):.4f}")

follow = defaultdict(Counter)
for (a, b), c in bi.items():
    follow[a][b] = c
tri_follow = defaultdict(Counter)
for (a, b, c3), c in tri.items():
    tri_follow[(a, b)][c3] = c

banner("Next word prediction")
for w in ["the", "machine", "natural", "students"]:
    print(f"after '{w}':", follow[w].most_common(4))
print("after 'the king' (trigram):", tri_follow[("the", "king")].most_common(3))

def generate(n=2, seed=None, max_len=15):
    rnd = random.Random(seed)
    out = ["<s>"]
    for _ in range(max_len):
        table = tri_follow.get(tuple(out[-2:])) if n == 3 and len(out) >= 2 else None
        if not table:
            table = follow[out[-1]]
        if not table:
            break
        nxt = rnd.choices(list(table), weights=list(table.values()))[0]
        if nxt == "</s>":
            break
        out.append(nxt)
    return " ".join(out[1:])

banner("Generated sentences")
for n in (2, 3):
    for s in range(3):
        print(f"{n}-gram model: {generate(n, seed=s)}")

banner("Perplexity (bigram, Laplace)")
for test in ["the king rules the kingdom", "kingdom the rules king the", "students write python code in the lab"]:
    t = ["<s>"] + test.split() + ["</s>"]
    lp = sum(math.log(p_bigram(a, b)) for a, b in zip(t, t[1:]))
    print(f"{test:<42} perplexity = {math.exp(-lp/(len(t)-1)):.1f}")
print("(lower perplexity = sentence is more probable under the model)")
