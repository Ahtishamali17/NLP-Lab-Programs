"""
Program 23 : Implementing Attention Mechanism (basic custom model)         [CO2]
Aim        : Build a custom Bahdanau (additive) attention layer from scratch and put it on top of a BiLSTM.
Theory     : for hidden states h_1..h_T :   e_t = v^T tanh(W h_t)      (score)
                                            a_t = softmax(e_t)         (attention weights, sum to 1)
                                            c   = sum_t a_t * h_t      (context vector -> classifier)
Algorithm  : 1) vectorize text  2) Embedding -> BiLSTM(return_sequences)  3) custom Attention layer
             4) Dense sigmoid  5) train  6) display which words the model attends to
Install    : pip install tensorflow
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from common import banner, load_reviews, OUTPUT_DIR

tf.random.set_seed(1); np.random.seed(1)


class BahdanauAttention(layers.Layer):
    """Additive attention over the time axis. Returns (context_vector, attention_weights)."""
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.supports_masking = True
        self.W = layers.Dense(units, activation="tanh")      # W h_t
        self.v = layers.Dense(1, use_bias=False)             # v^T (...)

    def call(self, hidden, mask=None):                       # hidden: (batch, T, features)
        score = tf.squeeze(self.v(self.W(hidden)), axis=-1)  # (batch, T)
        if mask is not None:                                 # ignore padding positions
            score += (1.0 - tf.cast(mask, score.dtype)) * -1e9
        alpha = tf.nn.softmax(score, axis=-1)                # attention weights
        context = tf.reduce_sum(hidden * tf.expand_dims(alpha, -1), axis=1)
        return context, alpha

    def compute_mask(self, inputs, mask=None):
        return None


df = load_reviews()
y = (df.sentiment == "positive").astype("float32").values
X_train, X_test, y_train, y_test = train_test_split(df.review.values, y, test_size=0.25, random_state=42, stratify=y)

SEQ_LEN = 30
vectorizer = layers.TextVectorization(max_tokens=1000, output_sequence_length=SEQ_LEN)
vectorizer.adapt(X_train)
vocab = vectorizer.get_vocabulary()
Xtr, Xte = vectorizer(X_train).numpy(), vectorizer(X_test).numpy()

inp = keras.Input(shape=(SEQ_LEN,), dtype="int32")
x = layers.Embedding(1000, 32, mask_zero=True)(inp)
x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
context, alpha = BahdanauAttention(32, name="attention")(x)
out = layers.Dense(1, activation="sigmoid")(layers.Dropout(0.3)(context))

model = keras.Model(inp, out)
attention_model = keras.Model(inp, alpha)          # shares the layers -> gives attention weights
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
banner("Model architecture"); model.summary()

banner("Training")
model.fit(Xtr, y_train, epochs=25, batch_size=8, validation_split=0.2, verbose=0)
loss, acc = model.evaluate(Xte, y_test, verbose=0)
print(f"Test accuracy with attention = {acc:.3f}")

banner("What does the model look at?  (attention weight of every word)")
for s in ["The camera is terrible and disappointing", "Amazing battery life, totally worth the money"]:
    ids = vectorizer([s]).numpy()
    w = attention_model.predict(ids, verbose=0)[0]
    toks = [vocab[i] for i in ids[0]]
    n = len([t for t in toks if t != ""])
    p = float(model.predict(ids, verbose=0)[0, 0])
    print(f"\n'{s}'  ->  {'positive' if p > 0.5 else 'negative'} (p={p:.2f})")
    for t, a in sorted(zip(toks[:n], w[:n]), key=lambda z: -z[1])[:5]:
        print(f"   {t:<14}{a:.3f}  {'#' * int(a * 60)}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    s = "Worst camera ever, a complete waste of money"
    ids = vectorizer([s]).numpy(); w = attention_model.predict(ids, verbose=0)[0]
    toks = [vocab[i] for i in ids[0] if i != 0]
    plt.figure(figsize=(8, 2)); plt.imshow(w[: len(toks)][None, :], cmap="Reds", aspect="auto")
    plt.xticks(range(len(toks)), toks, rotation=45, ha="right"); plt.yticks([]); plt.title("Attention heat-map")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "23_attention_heatmap.png", dpi=120)
    print("\nHeat-map saved to outputs/23_attention_heatmap.png")
except Exception as e:
    print("plot skipped:", e)
