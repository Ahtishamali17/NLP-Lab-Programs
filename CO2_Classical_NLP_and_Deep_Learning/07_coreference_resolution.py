"""
Program 7 : Coreference Resolution (neuralcoref or similar)               [CO2]
Aim       : Link pronouns ("he", "it", "they") to the entity they refer to.
Note      : `neuralcoref` is unmaintained and does not install on spaCy 3 / Python 3.9+.
            So this program implements a transparent RULE BASED resolver on top of spaCy
            (gender / number agreement + recency + binding rule) and, if the package `fastcoref`
            is installed, also runs a neural resolver ("or similar") for comparison.
Algorithm : 1) find candidate mentions (PERSON/ORG/GPE entities + noun chunks)
            2) for every pronoun, filter candidates by gender & number
            3) drop same-clause subject for object pronouns (binding)  4) choose the most recent / subject
            5) build coreference clusters and rewrite the text
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from collections import defaultdict
from common import banner, load_spacy

MALE_NAMES = {"rahul", "sundar", "john", "amit", "raj", "arjun", "kalam", "narendra", "ravi", "mohan"}
FEMALE_NAMES = {"priya", "anita", "ruchika", "mary", "sita", "neha", "kavita", "pooja"}
MALE_P, FEMALE_P = {"he", "him", "his", "himself"}, {"she", "her", "hers", "herself"}
NEUTER_P, PLURAL_P = {"it", "its", "itself"}, {"they", "them", "their", "theirs", "themselves"}
OBJ_DEPS = {"dobj", "obj", "dative", "iobj", "pobj"}


def gender_of(span):
    first = span.text.split()[0].lower()
    if first in MALE_NAMES or first in {"mr", "mr."}:
        return "m"
    if first in FEMALE_NAMES or first in {"mrs", "ms", "mrs.", "ms."}:
        return "f"
    return "?"


def candidates(doc):
    """Return list of (span, kind) where kind in {person, org, thing, plural}."""
    cands, used = [], set()
    for ent in doc.ents:
        kind = "person" if ent.label_ == "PERSON" else "org" if ent.label_ in ("ORG", "GPE", "NORP") else None
        if kind:
            cands.append((ent, kind)); used.update(range(ent.start, ent.end))
    for nc in doc.noun_chunks:
        if nc.root.pos_ == "PRON" or any(i in used for i in range(nc.start, nc.end)):
            continue
        if nc.root.pos_ == "PROPN" and nc.root.text.lower() in MALE_NAMES | FEMALE_NAMES:
            kind = "person"                       # names the NER model missed (e.g. Priya, Rahul)
        elif nc.root.tag_ in ("NNS", "NNPS"):
            kind = "plural"
        else:
            kind = "thing"
        cands.append((nc, kind))
    return sorted(cands, key=lambda c: c[0].start)


def resolve(doc):
    """Map pronoun token index -> antecedent span."""
    cands = candidates(doc)
    sent_no = {s.start: n for n, s in enumerate(doc.sents)}
    sid = lambda span: sent_no[span.sent.start]
    links = {}
    for tok in doc:
        w = tok.text.lower()
        if tok.pos_ != "PRON" or w not in MALE_P | FEMALE_P | NEUTER_P | PLURAL_P:
            continue
        ok = []
        for span, kind in cands:
            if span.end > tok.i or sid(tok.sent) - sid(span) > 2:   # must precede, max 2 sentences back
                continue
            g = gender_of(span) if kind == "person" else None
            if w in MALE_P and not (kind == "person" and g in ("m", "?")): continue
            if w in FEMALE_P and not (kind == "person" and g in ("f", "?")): continue
            if w in NEUTER_P and kind not in ("org", "thing"): continue
            if w in PLURAL_P and kind not in ("org", "plural"): continue
            # binding rule: an object pronoun cannot refer to the subject of its own verb
            reflexive = w.endswith("self") or w.endswith("selves")
            if tok.dep_ in OBJ_DEPS and not reflexive and span.root.dep_ == "nsubj" and span.root.head == tok.head:
                continue
            ok.append(span)
        if ok:
            subjects = [sp for sp in ok if sp.root.dep_ in ("nsubj", "nsubjpass")]
            links[tok.i] = (subjects or ok)[-1]        # prefer the most recent subject, else most recent mention
    return links


def rewrite(doc, links):
    out = []
    for tok in doc:
        if tok.i in links:
            ante = links[tok.i].text
            if tok.dep_ == "poss":
                ante += "'" if ante.endswith("s") else "'s"
            out.append(ante + tok.whitespace_)
        else:
            out.append(tok.text_with_ws)
    return "".join(out)


if __name__ == "__main__":
    text = ("Sundar Pichai is the CEO of Google. He joined the company in 2004. "
            "Priya met Rahul at NIET. She gave him a book because he helped her. "
            "Microsoft released a new model. It outperformed the old system, and the engineers were proud of their work.")
    nlp = load_spacy()
    doc = nlp(text)
    links = resolve(doc)

    banner("Pronoun -> antecedent links")
    for i, span in links.items():
        print(f"{doc[i].text:<8} (sentence {list(doc.sents).index(doc[i].sent)+1})  ->  {span.text}")

    banner("Coreference clusters")
    clusters = defaultdict(list)
    for i, span in links.items():
        clusters[span.text].append(doc[i].text)
    for k, v in clusters.items():
        print(f"{k}: mentions = {[k] + v}")

    banner("Original text");  print(text)
    banner("Resolved text");  print(rewrite(doc, links))

    banner("Optional neural resolver (fastcoref)")
    try:
        from fastcoref import spacy_component  # noqa: F401
        nlp2 = load_spacy()
        nlp2.add_pipe("fastcoref")
        d2 = nlp2(text, component_cfg={"fastcoref": {"resolve_text": True}})
        print(d2._.resolved_text)
    except Exception as e:
        print("fastcoref not installed (pip install fastcoref) -> skipped:", type(e).__name__)
