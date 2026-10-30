"""
common.py - helpers shared by all 32 NLP Lab programs.
Keeps every program short: dataset loading, resource download, sample text,
embedding loader and a synthetic "scanned page stream" generator (Programs 31, 32).
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_TEXT = (
    "Dr. A. P. J. Abdul Kalam was born in Rameswaram, Tamil Nadu on 15 October 1931. "
    "He served as the 11th President of India from 2002 to 2007. "
    "Natural Language Processing (NLP) helps computers understand human language! "
    "Isn't it amazing? Google and Microsoft invest billions of dollars in AI research, "
    "and NIET in Greater Noida teaches it too."
)


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def ensure_nltk(extra=()):
    """Download the NLTK resources used by the lab (silently, only if missing)."""
    import nltk
    pkgs = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
            "vader_lexicon", "universal_tagset"]
    for p in list(pkgs) + list(extra):
        try:
            nltk.download(p, quiet=True)
        except Exception as e:  # offline etc.
            print(f"[warn] could not download {p}: {e}")


def load_spacy(model="en_core_web_sm"):
    import spacy
    try:
        return spacy.load(model)
    except OSError:
        print(f"[info] downloading spaCy model {model} ...")
        from spacy.cli import download
        download(model)
        return spacy.load(model)


def read_text(name):
    return (DATA_DIR / name).read_text(encoding="utf-8")


def load_reviews():
    import pandas as pd
    return pd.read_csv(DATA_DIR / "reviews.csv")      # columns: review, sentiment


def load_news():
    import pandas as pd
    return pd.read_csv(DATA_DIR / "news.csv")          # columns: text, category


def load_corpus():
    """Small custom corpus (one sentence per line) used by n-gram / Word2Vec / WMD programs."""
    return [l.strip() for l in read_text("corpus.txt").splitlines() if l.strip()]


def load_embeddings(allow_fallback=True):
    """
    Return gensim KeyedVectors.
      1. data/glove.6B.50d.txt (if you downloaded it manually), else
      2. gensim-data 'glove-wiki-gigaword-50' (auto download, ~66 MB), else
      3. (fallback) Word2Vec trained on the lab corpus - only if allow_fallback=True.
    """
    from gensim.models import KeyedVectors
    local = DATA_DIR / "glove.6B.50d.txt"
    if local.exists():
        print("[info] loading local GloVe file ...")
        return KeyedVectors.load_word2vec_format(str(local), binary=False, no_header=True)
    try:
        import gensim.downloader as api
        print("[info] loading glove-wiki-gigaword-50 via gensim downloader ...")
        return api.load("glove-wiki-gigaword-50")
    except Exception as e:
        if not allow_fallback:
            raise SystemExit(f"Could not load GloVe ({e}). Put glove.6B.50d.txt inside data/.")
        print(f"[warn] GloVe unavailable ({e}); training Word2Vec on lab corpus instead.")
        from gensim.models import Word2Vec
        from gensim.utils import simple_preprocess
        sents = [simple_preprocess(s) for s in load_corpus()]
        return Word2Vec(sents, vector_size=50, window=4, min_count=1, epochs=200, seed=1).wv


# ----------------------------------------------------------------------------
# Synthetic stream of scanned pages (for document classification / separation)
# ----------------------------------------------------------------------------
DOC_TYPES = {
    "invoice": dict(
        first=["invoice", "invoice number", "bill to", "amount due", "payment terms", "gst number"],
        body=["item", "quantity", "unit price", "total", "tax", "subtotal", "discount",
              "payment", "amount", "rupees", "qty", "shipping", "balance", "order"]),
    "resume": dict(
        first=["curriculum vitae", "career objective", "personal details", "email", "phone", "address"],
        body=["experience", "skills", "python", "education", "project", "university", "internship",
              "certification", "degree", "team", "developed", "achievements", "languages", "cgpa"]),
    "research_paper": dict(
        first=["abstract", "keywords", "introduction", "authors", "department", "proceedings"],
        body=["method", "results", "dataset", "accuracy", "model", "experiment", "analysis",
              "conclusion", "references", "figure", "table", "proposed", "baseline", "training"]),
    "letter": dict(
        first=["dear sir", "date", "subject", "respected madam", "to the principal", "ref no"],
        body=["kindly", "request", "regarding", "yours sincerely", "faithfully", "grateful",
              "permission", "consider", "application", "thank you", "enclosed", "attention", "leave", "matter"]),
}
COMMON_WORDS = ["the", "of", "and", "to", "in", "for", "is", "with", "this", "that", "on", "as", "by", "a"]


def make_page_stream(n_docs=40, seed=0, noise=(1, 4)):
    """Return list of page dicts: doc_id, page_no, type, is_first, text (a scanned-batch simulation).
    noise=(lo, hi): how many words from OTHER document types leak into a page (OCR / vocabulary noise)."""
    rng = random.Random(seed)
    pages = []
    for d in range(n_docs):
        t = rng.choice(list(DOC_TYPES))
        spec = DOC_TYPES[t]
        n_pages = rng.randint(1, 4)
        for p in range(1, n_pages + 1):
            words = rng.choices(COMMON_WORDS, k=14) + rng.choices(spec["body"], k=rng.randint(14, 24))
            other = rng.choice([x for x in DOC_TYPES if x != t])          # OCR / vocabulary noise
            words += rng.choices(DOC_TYPES[other]["body"], k=rng.randint(*noise))
            rng.shuffle(words)
            if p == 1:
                words = rng.sample(spec["first"], k=3) + words
            if rng.random() < 0.5:
                words.append(f"page {p} of {n_pages}")
            pages.append(dict(doc_id=d, page_no=p, type=t, is_first=(p == 1), text=" ".join(words)))
    return pages
