"""
Program 15 : Text Classification using Naive Bayes / SVM with TF-IDF       [CO2]
Aim        : Classify news sentences into sports / technology / health / politics / business.
Algorithm  : 1) load data/news.csv  2) train/test split  3) Pipeline(TF-IDF -> classifier)
             4) Multinomial Naive Bayes, Linear SVM, Logistic Regression  5) accuracy, report, confusion matrix
             6) cross validation  7) predict new text, save model
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from common import banner, load_news, OUTPUT_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

df = load_news()
print(df.head(), "\n\nClass distribution:\n", df.category.value_counts())
X_train, X_test, y_train, y_test = train_test_split(df.text, df.category, test_size=0.25, random_state=42, stratify=df.category)
print(f"\nTrain size = {len(X_train)}, Test size = {len(X_test)}")

models = {"Naive Bayes": MultinomialNB(),
          "Linear SVM": LinearSVC(),
          "Logistic Regression": LogisticRegression(max_iter=1000)}
results, fitted = {}, {}
for name, clf in models.items():
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), stop_words="english"), clf)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(pipe, df.text, df.category, cv=5).mean()
    results[name] = (acc, cv); fitted[name] = pipe
    banner(f"{name}")
    print(f"Test accuracy = {acc:.3f}   5-fold CV accuracy = {cv:.3f}")
    print(classification_report(y_test, pred, zero_division=0))
    labels = sorted(df.category.unique())
    print("Confusion matrix:\n", pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels))

banner("Model comparison")
for n, (a, c) in results.items():
    print(f"{n:<22} test={a:.3f}  cv={c:.3f}")

banner("Predicting unseen sentences (SVM)")
for s in ["The batsman hit a century in the cricket final",
          "Doctors discovered a new treatment for cancer patients",
          "The software startup released a machine learning platform for cloud customers",
          "Shares and bank profits rose in the market"]:
    print(f"{fitted['Linear SVM'].predict([s])[0]:<12} <- {s}")

joblib.dump(fitted["Linear SVM"], OUTPUT_DIR / "15_news_classifier.joblib")
print("\nModel saved to outputs/15_news_classifier.joblib")
