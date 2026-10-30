"""
Program 2 : Stemming and Lemmatization on sample text                      [CO1]
Aim       : Reduce words to their root form using rule based stemmers and dictionary based lemmatizers.
Algorithm : 1) tokenize  2) Porter / Lancaster / Snowball stemming
            3) WordNet lemmatization (with POS tag)  4) spaCy lemmatization  5) compare
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import banner, ensure_nltk, load_spacy
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

ensure_nltk()
text = "The children were running faster and studying better programs while the mice are eating cheeses"
tokens = word_tokenize(text.lower())

porter, lancaster, snowball = PorterStemmer(), LancasterStemmer(), SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def wn_pos(tag):
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)

pos_tags = nltk.pos_tag(tokens)
nlp = load_spacy()
spacy_lemmas = {t.text: t.lemma_ for t in nlp(text.lower())}

banner("Stemming vs Lemmatization")
print(f"{'WORD':<12}{'PORTER':<11}{'LANCASTER':<11}{'SNOWBALL':<11}{'WN-LEMMA':<11}{'WN-LEMMA(POS)':<15}{'SPACY'}")
for w, tag in pos_tags:
    print(f"{w:<12}{porter.stem(w):<11}{lancaster.stem(w):<11}{snowball.stem(w):<11}"
          f"{lemmatizer.lemmatize(w):<11}{lemmatizer.lemmatize(w, wn_pos(tag)):<15}{spacy_lemmas.get(w, '-')}")

banner("Special cases")
for w in ["better", "mice", "geese", "was", "studies", "caring", "happily"]:
    print(f"{w:<10} stem={porter.stem(w):<9} lemma(noun)={lemmatizer.lemmatize(w):<9} lemma(verb)={lemmatizer.lemmatize(w,'v'):<9} lemma(adj)={lemmatizer.lemmatize(w,'a')}")

banner("Observation")
print("* Stemming is fast but may give non-words (studies -> studi).")
print("* Lemmatization returns valid dictionary words and needs the POS tag (better -> good only as adjective).")
