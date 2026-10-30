"""
Program 5 : Parsing and Chunking using regex and spaCy                     [CO1]
Aim       : Find phrase structure (chunks) and grammatical relations (parse) in a sentence.
Algorithm : 1) POS tag  2) regex chunk grammar (NLTK RegexpParser) for NP / VP / PP
            3) full parse with a small CFG (chart parser)  4) spaCy noun chunks + dependency parse
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import banner, ensure_nltk, load_spacy
import nltk
from nltk.tokenize import word_tokenize

ensure_nltk()
sentence = "The smart student wrote a clean program in the lab"

banner("1. Chunking with regular expressions (NLTK RegexpParser)")
grammar = r"""
  NP: {<DT>?<JJ.*>*<NN.*>+}      # determiner, adjectives, nouns
  PP: {<IN><NP>}                 # preposition + noun phrase
  VP: {<VB.*><NP|PP>+}           # verb + (NP or PP)
"""
tagged = nltk.pos_tag(word_tokenize(sentence))
print("Tagged:", tagged)
tree = nltk.RegexpParser(grammar).parse(tagged)
print(tree)
tree.pretty_print()
print("Noun phrases:", [" ".join(w for w, _ in st.leaves()) for st in tree.subtrees(lambda t: t.label() == "NP")])

banner("2. Parsing with a Context-Free Grammar (chart parser)")
cfg = nltk.CFG.fromstring("""
  S  -> NP VP
  NP -> Det N | Det Adj N | NP PP
  VP -> V NP | VP PP
  PP -> P NP
  Det -> 'the' | 'a'
  Adj -> 'smart' | 'clean'
  N  -> 'student' | 'program' | 'lab'
  V  -> 'wrote'
  P  -> 'in'
""")
parser = nltk.ChartParser(cfg)
trees = list(parser.parse(sentence.lower().split()))
print("Number of valid parse trees (structural ambiguity):", len(trees))
for t in trees:
    t.pretty_print()

banner("3. spaCy noun chunks")
nlp = load_spacy()
doc = nlp(sentence)
for c in doc.noun_chunks:
    print(f"{c.text:<22} root={c.root.text:<9} role={c.root.dep_}")

banner("4. spaCy dependency parse")
print(f"{'TOKEN':<10}{'DEP':<8}{'HEAD':<10}{'CHILDREN'}")
for t in doc:
    print(f"{t.text:<10}{t.dep_:<8}{t.head.text:<10}{[c.text for c in t.children]}")

banner("5. Subject - Verb - Object extraction")
for t in doc:
    if t.pos_ == "VERB":
        subj = [c.text for c in t.children if c.dep_ in ("nsubj", "nsubjpass")]
        obj = [c.text for c in t.children if c.dep_ in ("dobj", "obj")]
        print("Subject:", subj, "| Verb:", t.text, "| Object:", obj)
