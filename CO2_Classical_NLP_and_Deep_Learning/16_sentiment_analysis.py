"""
Program 16 : Sentiment Analysis using TextBlob and VADER                   [CO2]
Aim        : Find the polarity (positive / negative / neutral) of text with two lexicon based tools.
Algorithm  : 1) TextBlob -> polarity [-1,1] and subjectivity [0,1]
             2) VADER -> neg/neu/pos/compound score (handles emojis, CAPS, negation, "very")
             3) evaluate both on data/reviews.csv against the true labels  4) compare
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from common import banner, load_reviews
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()
samples = ["I love this phone, the camera is amazing!",
           "This is the worst laptop I have ever bought.",
           "The delivery was okay.",
           "The movie was not good.",
           "The food was VERY GOOD!!! :)",
           "The battery is good but the screen is terrible."]

banner("Sentence level scores")
print(f"{'SENTENCE':<50}{'TB pol':>8}{'TB subj':>9}{'VADER':>8}")
for s in samples:
    tb = TextBlob(s).sentiment
    print(f"{s:<50}{tb.polarity:>8.2f}{tb.subjectivity:>9.2f}{vader.polarity_scores(s)['compound']:>8.2f}")

banner("VADER full output for one sentence")
print(vader.polarity_scores("The food was VERY GOOD!!! :)"))

def tb_label(t):
    p = TextBlob(t).sentiment.polarity
    return "positive" if p >= 0 else "negative"

def vader_label(t):
    c = vader.polarity_scores(t)["compound"]
    return "positive" if c >= 0.05 else "negative"

df = load_reviews()
df["textblob"] = df.review.apply(tb_label)
df["vader"] = df.review.apply(vader_label)

for tool in ["textblob", "vader"]:
    banner(f"Evaluation of {tool.upper()} on {len(df)} reviews")
    print("Accuracy:", round(accuracy_score(df.sentiment, df[tool]), 3))
    print(classification_report(df.sentiment, df[tool], zero_division=0))
    print(pd.DataFrame(confusion_matrix(df.sentiment, df[tool], labels=["positive", "negative"]),
                       index=["true_pos", "true_neg"], columns=["pred_pos", "pred_neg"]))

banner("Reviews where the tools disagree")
dis = df[df.textblob != df.vader][["review", "sentiment", "textblob", "vader"]]
print(dis.head(5).to_string(index=False) if len(dis) else "None - the two tools agree on every review (the dataset is clean, templated text).")
