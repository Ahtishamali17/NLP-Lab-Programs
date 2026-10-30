"""
Program 32 : Hybrid Probabilistic + FSM-Based Sequence Modeling for document boundary detection   [CO3]
Aim        : Decide, for every page of a scanned batch, whether it BEGINS a document (B) or CONTINUES one (I),
             and of which type - by combining
               * a PROBABILISTIC page model   P(state | page text)   (TF-IDF + Logistic Regression)
               * a FINITE STATE MACHINE        that forbids impossible transitions
               * HMM-style Viterbi decoding    over the whole page sequence.
FSM        : states  B_t  (first page of a type-t document)  and  I_t  (continuation page of type t)
             start  -> only B_t allowed          B_t -> I_t | B_any          I_t -> I_t | B_any
             (an I_t page can never follow a page of another type -> hard zero in the transition matrix)
Algorithm  : 1) train the page classifier on 8 states  2) estimate transition probabilities from training
                sequences, then mask with the FSM  3) emission score = log P(state|x) - log P(state)
             4) Viterbi decoding  5) compare with a greedy (no FSM) baseline: boundary precision / recall / F1
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from common import banner, make_page_stream, DOC_TYPES

TYPES = sorted(DOC_TYPES)
STATES = [f"{k}_{t}" for k in ("B", "I") for t in TYPES]
S = {s: i for i, s in enumerate(STATES)}
nS = len(STATES)

state_of = lambda p: ("B_" if p["is_first"] else "I_") + p["type"]
NOISE = (8, 16)          # many words from other document types leak into every page -> a hard, noisy batch
train = make_page_stream(n_docs=150, seed=1, noise=NOISE)
valid = make_page_stream(n_docs=60, seed=3, noise=NOISE)      # used only to tune the emission weight
test = make_page_stream(n_docs=60, seed=2, noise=NOISE)

banner("1. FSM definition")
allowed = np.zeros((nS, nS), bool)                       # allowed[i, j] : transition i -> j is legal
for a in STATES:
    for b in STATES:
        if b.startswith("B_"):
            allowed[S[a], S[b]] = True                   # a new document may start after any page
        else:                                            # I_t only after B_t or I_t of the SAME type
            allowed[S[a], S[b]] = a.split("_", 1)[1] == b.split("_", 1)[1]
start_allowed = np.array([s.startswith("B_") for s in STATES])
print("States:", STATES)
print(f"Legal transitions: {allowed.sum()} of {nS*nS}   |   legal start states: {[s for s in STATES if s.startswith('B_')]}")

banner("2. Probabilistic part: P(state | page text), priors and transition probabilities")
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=3000, C=10))
clf.fit([p["text"] for p in train], [state_of(p) for p in train])
order = [S[c] for c in clf.classes_]                     # map classifier column -> state index

counts = np.ones((nS, nS))                               # Laplace smoothing
prior = np.ones(nS)
for a, b in zip(train, train[1:]):
    counts[S[state_of(a)], S[state_of(b)]] += 1
for p in train:
    prior[S[state_of(p)]] += 1
prior /= prior.sum()
counts = counts * allowed                                # FSM mask: illegal transitions get probability 0
trans = counts / counts.sum(1, keepdims=True)
print("Prior P(state):", {k: round(float(v), 3) for k, v in zip(STATES, prior)})
print("P(I_invoice | B_invoice) =", trans[S["B_invoice"], S["I_invoice"]].round(3),
      "  P(I_invoice | B_resume) =", trans[S["B_resume"], S["I_invoice"]])

def emissions(pages):
    post = np.zeros((len(pages), nS))
    post[:, order] = clf.predict_proba([p["text"] for p in pages])
    return np.log(post + 1e-12) - np.log(prior)           # scaled likelihood  P(x|s) ~ P(s|x)/P(s)

def viterbi(E):
    T = len(E)
    with np.errstate(divide="ignore"):
        logT = np.log(trans); logS = np.where(start_allowed, np.log(prior), -np.inf)
    delta = np.full((T, nS), -np.inf); back = np.zeros((T, nS), int)
    delta[0] = logS + E[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + logT              # (from, to)
        back[t] = cand.argmax(0); delta[t] = cand.max(0) + E[t]
    path = [int(delta[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(back[t][path[-1]])
    return path[::-1]

banner("3. Decoding the test stream")
# emission weight w (like the 'acoustic scale' in HMM-DNN hybrids) balances the page model against the FSM transitions
Ev, truth_v = emissions(valid), [S[state_of(p)] for p in valid]
scores = {w: np.mean(np.array(viterbi(Ev * w)) == truth_v) for w in (1, 2, 3, 5)}
W = max(scores, key=scores.get)
print("Validation accuracy for emission weights:", {w: round(float(a), 3) for w, a in scores.items()}, "-> chosen w =", W)

E = emissions(test)
hybrid = [STATES[i] for i in viterbi(E * W)]
greedy = [STATES[i] for i in E.argmax(1)]                # baseline: best state per page, no sequence model
truth = [state_of(p) for p in test]
print(f"{'page':<5}{'truth':<18}{'greedy':<18}{'hybrid (FSM+Viterbi)'}")
shown = 0
for i in range(len(test)):
    if greedy[i] == truth[i] and hybrid[i] == truth[i] and i > 5:
        continue
    if shown >= 14: break
    shown += 1
    flag = "" if hybrid[i] == truth[i] else "   <-- error"
    print(f"{i:<5}{truth[i]:<18}{greedy[i]:<18}{hybrid[i]}{flag}")

banner("4. Evaluation")
def report(name, pred):
    yb = [t.startswith("B_") for t in truth]; pb = [s.startswith("B_") for s in pred]
    pr, rc, f1, _ = precision_recall_fscore_support(yb, pb, average="binary", zero_division=0)
    acc = np.mean([a == b for a, b in zip(truth, pred)])
    illegal = sum(1 for a, b in zip(pred, pred[1:]) if not allowed[S[a], S[b]])
    print(f"{name:<22} boundary P={pr:.3f} R={rc:.3f} F1={f1:.3f} | state accuracy={acc:.3f} | illegal transitions={illegal}")
report("Greedy (no FSM)", greedy)
report("Hybrid FSM + Viterbi", hybrid)

banner("5. Documents recovered by the hybrid model")
docs, cur = [], []
for i, s in enumerate(hybrid):
    if s.startswith("B_") and cur:
        docs.append(cur); cur = []
    cur.append(i)
docs.append(cur)
true_n = len(set(p["doc_id"] for p in test))
print(f"Predicted {len(docs)} documents; ground truth {true_n}")
for n, d in enumerate(docs[:6], 1):
    print(f"  doc {n}: pages {d[0]}-{d[-1]}  type = {hybrid[d[0]].split('_',1)[1]}")
print("\nThe FSM guarantees a legal output (never an 'I' page of a different type than its document); Viterbi finds the")
print("globally most probable legal sequence instead of trusting every noisy per-page decision.")
