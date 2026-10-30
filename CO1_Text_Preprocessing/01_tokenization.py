"""
Program 1 : Tokenization of Sentences and Words using NLTK and spaCy      [CO1]
Aim       : Split raw text into sentences and words (tokens).
Algorithm : 1) load text  2) NLTK sent_tokenize / word_tokenize  3) other NLTK tokenizers
            4) spaCy pipeline -> doc.sents and tokens  5) compare both libraries
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import banner, ensure_nltk, load_spacy, SAMPLE_TEXT
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer, TweetTokenizer, RegexpTokenizer

ensure_nltk()
text = SAMPLE_TEXT
print("INPUT TEXT:\n", text)

banner("NLTK - sentence tokenization")
nltk_sents = sent_tokenize(text)
for i, s in enumerate(nltk_sents, 1):
    print(f"{i}. {s}")

banner("NLTK - word tokenization")
nltk_words = word_tokenize(text)
print(nltk_words)

banner("Other NLTK tokenizers")
print("WordPunct :", WordPunctTokenizer().tokenize("Don't stop-believing, NLP's fun!"))
print("Tweet     :", TweetTokenizer().tokenize("Loving #NLP lab :) @ruchika http://niet.co.in"))
print("Regexp    :", RegexpTokenizer(r"[A-Za-z]+").tokenize(text)[:12], "...")

banner("spaCy - sentence tokenization")
nlp = load_spacy()
doc = nlp(text)
spacy_sents = list(doc.sents)
for i, s in enumerate(spacy_sents, 1):
    print(f"{i}. {s.text}")

banner("spaCy - word tokenization (with token attributes)")
print(f"{'TOKEN':<14}{'IS_ALPHA':<10}{'IS_PUNCT':<10}{'LIKE_NUM':<10}")
for t in doc[:20]:
    print(f"{t.text:<14}{str(t.is_alpha):<10}{str(t.is_punct):<10}{str(t.like_num):<10}")

banner("Comparison")
print(f"NLTK  : {len(nltk_sents)} sentences, {len(nltk_words)} tokens")
print(f"spaCy : {len(spacy_sents)} sentences, {len(doc)} tokens")
print("Note: NLTK splits \"Isn't\" -> ['Is', \"n't\"]; spaCy does the same, but they differ on 'Dr.' and abbreviations.")
