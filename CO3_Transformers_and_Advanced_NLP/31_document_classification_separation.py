"""
Program 31 : Automatic Document Classification and Separation using ML     [CO3]
Aim        : A scanned batch contains many documents (invoices, resumes, papers, letters) stacked together as
             single pages. (1) classify every page, (2) detect where a NEW document starts, (3) separate the
             stream into individual documents and save them.
Algorithm  : 1) synthetic page stream (common.make_page_stream)  2) TF-IDF + Logistic Regression -> document TYPE
             3) TF-IDF + Logistic Regression -> "is this the FIRST page?"  (boundary detector)
             4) rule: new document if first-page prob > 0.5  OR  predicted type changes
             5) group pages, majority vote of type per document  6) evaluate + write files to outputs/
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from common import banner, make_page_stream, OUTPUT_DIR

train = make_page_stream(n_docs=150, seed=1)
test = make_page_stream(n_docs=40, seed=2)
print(f"Training pages: {len(train)}   Test pages: {len(test)} (from {len(set(p['doc_id'] for p in test))} documents)")

def fit(labels_key):
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000, C=10))
    return pipe.fit([p["text"] for p in train], [p[labels_key] for p in train])

type_clf = fit("type")                                   # model 1: page -> document type
first_clf = fit("is_first")                              # model 2: page -> first page of a document?

banner("1. Page level document classification")
X_test = [p["text"] for p in test]
pred_type = type_clf.predict(X_test)
print("Accuracy:", round(accuracy_score([p["type"] for p in test], pred_type), 3))
print(classification_report([p["type"] for p in test], pred_type, zero_division=0))

banner("2. Boundary (first page) detection")
p_first = first_clf.predict_proba(X_test)[:, list(first_clf.classes_).index(True)]
is_first_pred = p_first > 0.5
y_first = [p["is_first"] for p in test]
pr, rc, f1, _ = precision_recall_fscore_support(y_first, is_first_pred, average="binary", zero_division=0)
print(f"first-page precision = {pr:.3f}   recall = {rc:.3f}   F1 = {f1:.3f}")

banner("3. Separation of the page stream into documents")
boundary = [True]                                        # page 0 always starts a document
for i in range(1, len(test)):
    boundary.append(bool(is_first_pred[i]) or pred_type[i] != pred_type[i - 1])
docs, cur = [], []
for i, b in enumerate(boundary):
    if b and cur:
        docs.append(cur); cur = []
    cur.append(i)
docs.append(cur)
true_docs = len(set(p["doc_id"] for p in test))
print(f"Predicted {len(docs)} documents (true number = {true_docs})")

# exact match: a predicted document is correct if its pages are exactly one true document
correct = sum(len({test[i]["doc_id"] for i in d}) == 1 and
              sum(1 for p in test if p["doc_id"] == test[d[0]]["doc_id"]) == len(d) for d in docs)
print(f"Perfectly separated documents: {correct}/{true_docs} ({100*correct/true_docs:.1f}%)")

banner("4. First 8 predicted documents")
out_dir = OUTPUT_DIR / "31_separated_documents"; out_dir.mkdir(exist_ok=True)
for f in out_dir.glob("*.txt"): f.unlink()
for n, d in enumerate(docs, 1):
    dtype = Counter(pred_type[i] for i in d).most_common(1)[0][0]    # majority vote
    true_t = Counter(test[i]["type"] for i in d).most_common(1)[0][0]
    if n <= 8:
        print(f"doc_{n:03d}  type={dtype:<15} pages={len(d)}  (true type: {true_t})")
    with open(out_dir / f"doc_{n:03d}_{dtype}.txt", "w", encoding="utf-8") as f:
        for k, i in enumerate(d, 1):
            f.write(f"--- page {k} ---\n{test[i]['text']}\n\n")
print(f"\n{len(docs)} separated documents written to {out_dir}")

banner("5. Predict a brand new batch")
new_pages = ["invoice number 4411 bill to customer amount due payment terms item quantity unit price total tax",
             "item quantity unit price subtotal discount shipping total amount rupees page 2 of 2",
             "curriculum vitae career objective email phone experience skills python education project"]
for t in new_pages:
    print(f"{type_clf.predict([t])[0]:<15} first_page={first_clf.predict([t])[0]!s:<6} <- {t[:70]}...")
