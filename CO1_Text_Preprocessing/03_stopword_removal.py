"""
Program 3 : Stop-word Removal from a document                              [CO1]
Aim       : Remove high frequency, low information words from a document.
Algorithm : 1) read data/sample_document.txt  2) tokenize  3) remove NLTK stop words
            4) remove with spaCy  5) custom stop-word list  6) report reduction, save cleaned text
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from collections import Counter
from common import banner, ensure_nltk, load_spacy, read_text, OUTPUT_DIR
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ensure_nltk()
doc_text = read_text("sample_document.txt")
tokens = [t for t in word_tokenize(doc_text) if t.isalpha()]
print("Total word tokens in document:", len(tokens))

banner("1. NLTK stop-word removal")
nltk_sw = set(stopwords.words("english"))
print("Number of NLTK English stop words:", len(nltk_sw))
nltk_clean = [t for t in tokens if t.lower() not in nltk_sw]
print("After removal:", len(nltk_clean), "tokens")
print(nltk_clean[:25])

banner("2. spaCy stop-word removal")
nlp = load_spacy()
doc = nlp(doc_text)
spacy_clean = [t.text for t in doc if t.is_alpha and not t.is_stop]
print("After removal:", len(spacy_clean), "tokens (spaCy has", len(nlp.Defaults.stop_words), "stop words)")
print(spacy_clean[:25])

banner("3. Custom stop-word list (domain specific)")
custom_sw = nltk_sw | {"india", "indian", "also", "many", "every"}
custom_clean = [t for t in tokens if t.lower() not in custom_sw]
print("After removal:", len(custom_clean), "tokens")

banner("4. Most frequent words before / after removal")
print("Before:", Counter(t.lower() for t in tokens).most_common(8))
print("After :", Counter(t.lower() for t in nltk_clean).most_common(8))

banner("5. Summary")
for name, res in [("NLTK", nltk_clean), ("spaCy", spacy_clean), ("Custom", custom_clean)]:
    print(f"{name:<7}: {len(res):>4} tokens left  ({100*(1-len(res)/len(tokens)):.1f}% reduction)")

out = OUTPUT_DIR / "03_cleaned_document.txt"
out.write_text(" ".join(nltk_clean), encoding="utf-8")
print("\nCleaned document saved to", out)
