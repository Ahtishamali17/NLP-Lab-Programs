"""
Program 22 : Sequence Classification using RNN / LSTM (Keras / TensorFlow) [CO2]
Aim        : Classify review sequences as positive / negative with recurrent networks.
Algorithm  : 1) TextVectorization (words -> integer ids, padded)  2) Embedding layer
             3) SimpleRNN, LSTM and Bidirectional-LSTM models  4) train + validate
             5) compare accuracy  6) predict new reviews
Install    : pip install tensorflow
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from common import banner, load_reviews, OUTPUT_DIR

tf.random.set_seed(42); np.random.seed(42)
df = load_reviews()
y = (df.sentiment == "positive").astype("float32").values
X_train, X_test, y_train, y_test = train_test_split(df.review.values, y, test_size=0.25, random_state=42, stratify=y)
print(f"train = {len(X_train)}   test = {len(X_test)}")

MAX_WORDS, SEQ_LEN = 1000, 30
vectorizer = layers.TextVectorization(max_tokens=MAX_WORDS, output_sequence_length=SEQ_LEN)
vectorizer.adapt(X_train)
Xtr, Xte = vectorizer(X_train).numpy(), vectorizer(X_test).numpy()
print("Vocabulary size:", len(vectorizer.get_vocabulary()))
print("Example :", X_train[0], "\n->", Xtr[0])

def build(kind):
    m = keras.Sequential([layers.Input(shape=(SEQ_LEN,)),
                          layers.Embedding(MAX_WORDS, 32, mask_zero=True)])
    if kind == "SimpleRNN":
        m.add(layers.SimpleRNN(32))
    elif kind == "LSTM":
        m.add(layers.LSTM(32))
    else:
        m.add(layers.Bidirectional(layers.LSTM(32)))
    m.add(layers.Dropout(0.3)); m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

results, models = {}, {}
for kind in ["SimpleRNN", "LSTM", "BiLSTM"]:
    banner(f"Training {kind}")
    m = build(kind)
    if kind == "LSTM":
        m.summary()
    hist = m.fit(Xtr, y_train, epochs=25, batch_size=8, validation_split=0.2, verbose=0)
    loss, acc = m.evaluate(Xte, y_test, verbose=0)
    results[kind] = acc; models[kind] = m
    print(f"final train acc = {hist.history['accuracy'][-1]:.3f}   val acc = {hist.history['val_accuracy'][-1]:.3f}   TEST acc = {acc:.3f}")

banner("Comparison")
for k, v in results.items():
    print(f"{k:<10} test accuracy = {v:.3f}")

best = max(results, key=results.get)
banner(f"Classification report of best model ({best})")
pred = (models[best].predict(Xte, verbose=0).ravel() > 0.5).astype(int)
print(classification_report(y_test, pred, target_names=["negative", "positive"], zero_division=0))

banner("Predictions on new reviews")
for s in ["The camera is excellent and I love it", "Terrible battery, a complete waste of money", "Worst delivery and very poor service"]:
    p = float(models[best].predict(vectorizer([s]).numpy(), verbose=0)[0, 0])
    print(f"{'positive' if p > 0.5 else 'negative':<9} (p={p:.2f})  {s}")
models[best].save(OUTPUT_DIR / "22_best_rnn_model.keras")
print("\nModel saved to outputs/22_best_rnn_model.keras")
