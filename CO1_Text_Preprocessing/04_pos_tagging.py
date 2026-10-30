"""
Program 4 : Part-of-Speech (POS) Tagging of a given sentence               [CO1]
Aim       : Assign grammatical categories (noun, verb, ...) to every word.
Algorithm : 1) tokenize  2) NLTK averaged-perceptron tagger (Penn + Universal tags)
            3) spaCy tagger  4) count tag frequencies  5) compare the two taggers
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from collections import Counter
from common import banner, ensure_nltk, load_spacy
import nltk
from nltk.tokenize import word_tokenize

ensure_nltk()
sentence = "The quick brown fox jumps over the lazy dog while students are learning natural language processing in Noida."
print("Sentence:", sentence)

banner("NLTK POS tagging (Penn Treebank tags)")
tokens = word_tokenize(sentence)
penn = nltk.pos_tag(tokens)
print(penn)

banner("NLTK POS tagging (Universal tagset)")
try:
    print(nltk.pos_tag(tokens, tagset="universal"))
except Exception as e:
    print("universal tagset unavailable:", e)

banner("spaCy POS tagging")
import spacy
nlp = load_spacy()
doc = nlp(sentence)
print(f"{'TOKEN':<14}{'UPOS':<8}{'FINE TAG':<9}{'MEANING'}")
for t in doc:
    print(f"{t.text:<14}{t.pos_:<8}{t.tag_:<9}{spacy.explain(t.tag_)}")

banner("Tag frequency")
print("NLTK :", Counter(tag for _, tag in penn).most_common())
print("spaCy:", Counter(t.pos_ for t in doc).most_common())

banner("Ambiguity example - same word, different tags")
for s in ["I will book a ticket.", "I read a book.", "They watch a movie.", "My watch is new."]:
    print(f"{s:<28}", [(w, t) for w, t in nltk.pos_tag(word_tokenize(s)) if w.lower() in ("book", "watch")])
