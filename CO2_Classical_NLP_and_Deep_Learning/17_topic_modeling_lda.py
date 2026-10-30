"""
Program 17 : Topic Modeling using Latent Dirichlet Allocation (LDA)        [CO2]
Aim        : Discover hidden topics: every document is a mixture of topics, every topic a distribution over words.
Algorithm  : 1) clean text + CountVectorizer  2) LDA (sklearn, variational Bayes) with k = 5
             3) top words per topic  4) document-topic distribution  5) infer topic of new text
             6) (optional) same with gensim LdaModel
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from common import banner, load_news
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

df = load_news()
cv = CountVectorizer(stop_words="english", max_df=0.8, min_df=2)
X = cv.fit_transform(df.text)
terms = cv.get_feature_names_out()
print("Document-term matrix:", X.shape)

K = 5
lda = LatentDirichletAllocation(n_components=K, random_state=42, learning_method="batch", max_iter=50)
doc_topic = lda.fit_transform(X)

banner(f"Top words of the {K} discovered topics")
for k, comp in enumerate(lda.components_):
    top = comp.argsort()[::-1][:8]
    print(f"Topic {k}: " + ", ".join(terms[top]))

banner("Document-topic distribution (first 6 documents)")
print(pd.DataFrame(doc_topic[:6].round(2), columns=[f"T{k}" for k in range(K)]))
print(df.category.head(6).tolist())

banner("Topic vs original category (how well topics match labels)")
df["topic"] = doc_topic.argmax(axis=1)
print(pd.crosstab(df.topic, df.category))

banner("Infer topic of unseen text")
for s in ["the football team scored a goal in the final", "doctors started a vaccine trial in the hospital"]:
    d = lda.transform(cv.transform([s]))[0]
    print(f"Topic {d.argmax()} (p={d.max():.2f}) <- {s}")
print(f"\nModel perplexity: {lda.perplexity(X):.1f} (lower is better)")

banner("Optional: gensim LDA")
try:
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.utils import simple_preprocess
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
    texts = [[w for w in simple_preprocess(t) if w not in SW] for t in df.text]
    dic = corpora.Dictionary(texts)
    bow = [dic.doc2bow(t) for t in texts]
    g = LdaModel(bow, num_topics=K, id2word=dic, passes=15, random_state=1)
    for t in g.print_topics(num_words=5):
        print(t)
except Exception as e:
    print("skipped:", e)
