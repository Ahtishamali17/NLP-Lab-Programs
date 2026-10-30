# Natural Language Processing (NLP) Lab - Practical File

**Department of Artificial Intelligence, Noida Institute of Engineering and Technology (NIET), Greater Noida**
**Course faculty:** Ms. Ruchika (Assistant Professor)  |  **Submission deadline:** 31 October 2026

| | |
|---|---|
| **Name** | `<your name>` |
| **Roll No.** | `<your roll number>` |
| **Section / Year** | `<section>` / `<year>` |
| **GitHub** | `<your repository link>` |

This repository contains all **32 programs** of the NLP Lab, organised by Course Outcome (CO). Every program is a
single, self-contained Python file that starts with an *Aim* and an *Algorithm*, prints readable output, and can be
run directly from VS Code or the terminal.

---

## 1. Repository structure

```
NLP-Lab-Programs/
├── README.md                    <- this practical file
├── requirements.txt             <- all Python dependencies
├── setup_resources.py           <- one-time download of NLTK data, spaCy model, GloVe (and HF models)
├── run_all.py                   <- runs programs and saves console output into outputs/
├── common.py                    <- shared helpers (dataset loaders, embeddings loader, page-stream generator)
├── data/                        <- datasets used by the programs
│   ├── reviews.csv              (88 product reviews + sentiment label)        Programs 16, 19, 22, 23, 24
│   ├── news.csv                 (60 news sentences, 5 categories)             Programs 15, 17, 18, 21
│   ├── corpus.txt               (40-sentence custom corpus)                   Programs 10, 12, 14
│   ├── sample_document.txt      (3-paragraph article)                         Programs 3, 29
│   ├── gpt2_custom.txt          (custom text for GPT-2 fine-tuning)           Program 26
│   └── generate_datasets.py     (re-creates reviews.csv and news.csv, seeded)
├── outputs/                     <- saved console output (NN_name.txt), plots, csv, json, html
├── CO1_Text_Preprocessing/                      Programs 1-6
├── CO2_Classical_NLP_and_Deep_Learning/         Programs 7-23
└── CO3_Transformers_and_Advanced_NLP/           Programs 24-32
```

## 2. List of programs

| S.No | Program | CO | File | Main libraries / technique |
|---|---|---|---|---|
| 1 | Tokenization of sentences and words (NLTK, spaCy) | CO1 | `CO1_Text_Preprocessing/01_tokenization.py` | NLTK, spaCy |
| 2 | Stemming and lemmatization | CO1 | `.../02_stemming_lemmatization.py` | Porter, Lancaster, Snowball, WordNet, spaCy |
| 3 | Stop-word removal from a document | CO1 | `.../03_stopword_removal.py` | NLTK, spaCy, custom list |
| 4 | Part-of-Speech tagging | CO1 | `.../04_pos_tagging.py` | NLTK perceptron tagger, spaCy |
| 5 | Parsing and chunking (regex, spaCy) | CO1 | `.../05_parsing_chunking.py` | RegexpParser, CFG chart parser, dependency parse |
| 6 | Named Entity Recognition with spaCy | CO1 | `.../06_named_entity_recognition.py` | spaCy, displaCy HTML |
| 7 | Coreference resolution | CO2 | `CO2_Classical_NLP_and_Deep_Learning/07_coreference_resolution.py` | rule-based resolver on spaCy (+ optional fastcoref) |
| 8 | Bag-of-Words | CO2 | `.../08_bag_of_words.py` | manual + CountVectorizer |
| 9 | TF-IDF and comparison with BoW | CO2 | `.../09_tfidf_vs_bow.py` | manual + TfidfVectorizer |
| 10 | N-gram models (uni/bi/tri) | CO2 | `.../10_ngram_model.py` | Counter, Laplace smoothing, perplexity, generation |
| 11 | Cosine similarity between documents | CO2 | `.../11_cosine_similarity.py` | numpy, scikit-learn |
| 12 | Word2Vec on a custom corpus | CO2 | `.../12_word2vec_gensim.py` | gensim (CBOW and Skip-gram) |
| 13 | GloVe embeddings loading and representation | CO2 | `.../13_glove_embeddings.py` | gensim KeyedVectors |
| 14 | Text similarity with Word Mover's Distance | CO2 | `.../14_word_movers_distance.py` | GloVe + scipy linear programming |
| 15 | Text classification (Naive Bayes / SVM / LR + TF-IDF) | CO2 | `.../15_text_classification.py` | scikit-learn pipelines |
| 16 | Sentiment analysis (TextBlob, VADER) | CO2 | `.../16_sentiment_analysis.py` | TextBlob, vaderSentiment |
| 17 | Topic modeling - LDA | CO2 | `.../17_topic_modeling_lda.py` | scikit-learn LDA (+ gensim) |
| 18 | Topic modeling - LSA | CO2 | `.../18_topic_modeling_lsa.py` | TF-IDF + TruncatedSVD |
| 19 | Opinion mining on reviews | CO2 | `.../19_opinion_mining.py` | aspect-based sentiment (VADER + spaCy) |
| 20 | Information extraction | CO2 | `.../20_information_extraction.py` | regex, spaCy NER, SVO triples, JSON/CSV/form parsing |
| 21 | Information retrieval with TF-IDF ranking | CO2 | `.../21_information_retrieval.py` | inverted index, cosine ranking, Precision@k |
| 22 | Sequence classification with RNN / LSTM | CO2 | `.../22_sequence_classification_rnn_lstm.py` | Keras: SimpleRNN, LSTM, BiLSTM |
| 23 | Attention mechanism (custom model) | CO2 | `.../23_attention_mechanism.py` | custom Bahdanau attention layer in Keras |
| 24 | Fine-tuning BERT for text classification | CO3 | `CO3_Transformers_and_Advanced_NLP/24_finetune_bert_classification.py` | Hugging Face + PyTorch |
| 25 | Sentence embeddings using BERT and GPT-2 | CO3 | `.../25_sentence_embeddings_bert_gpt2.py` | mean / CLS / last-token pooling |
| 26 | Text generation with GPT-2 on a custom dataset | CO3 | `.../26_gpt2_text_generation.py` | GPT-2 fine-tuning + top-k/top-p sampling |
| 27 | NER using Transformers (BERT) | CO3 | `.../27_ner_with_bert.py` | `dslim/bert-base-NER` |
| 28 | Machine translation (MarianMT / T5) | CO3 | `.../28_machine_translation.py` | en to hi / fr / de, BLEU |
| 29 | Text summarization (BART / T5) | CO3 | `.../29_text_summarization.py` | DistilBART, T5, ROUGE |
| 30 | Chatbot with a pre-trained transformer | CO3 | `.../30_chatbot_dialogpt.py` | DialoGPT with chat history |
| 31 | Automatic document classification and separation | CO3 | `.../31_document_classification_separation.py` | TF-IDF + Logistic Regression, boundary detection |
| 32 | Hybrid probabilistic + FSM sequence modeling for document boundary detection | CO3 | `.../32_hybrid_probabilistic_fsm.py` | classifier + FSM + Viterbi decoding |

## 3. Setup (VS Code / terminal)

Python **3.10 - 3.12** is recommended.

```bash
# 1. clone and enter the repository
git clone https://github.com/<your-username>/NLP-Lab-Programs.git
cd NLP-Lab-Programs

# 2. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS

# 3. install dependencies
pip install -r requirements.txt

# 4. one-time download of NLTK data, spaCy model and GloVe
python setup_resources.py          # add  --models  to pre-download the Hugging Face models for Programs 24-30
```

**In VS Code:** open the folder (`File > Open Folder`), press `Ctrl+Shift+P` -> *Python: Select Interpreter* ->
choose `.venv`, then open any program and press the **Run** button (or `F5`).
The programs find `common.py` and `data/` automatically, so they can be run from any working directory.

## 4. How to run

```bash
python CO1_Text_Preprocessing/01_tokenization.py                      # a single program
python run_all.py                                                     # all 32 programs, console output saved in outputs/
python run_all.py 1 5 12                                              # only some programs
python run_all.py --skip-heavy                                        # skip the deep-learning programs 22-30
python CO3_Transformers_and_Advanced_NLP/30_chatbot_dialogpt.py --interactive   # chat with the bot
python CO2_Classical_NLP_and_Deep_Learning/21_information_retrieval.py --interactive
```

`run_all.py` writes `outputs/NN_program_name.txt` for each program. Together with the generated plots / csv / json / html
files these are the **outputs** to show in the practical file.

## 5. Notes on specific programs

* **Program 7 (coreference):** the `neuralcoref` package is no longer maintained and does not install on spaCy 3 /
  Python 3.9+. The "or similar" option is used: a transparent rule-based resolver (gender and number agreement,
  recency, subject preference, binding rule) built on spaCy. If `fastcoref` is installed, a neural resolver is also run.
* **Program 13 / 14:** GloVe is loaded from `data/glove.6B.50d.txt` if present, otherwise downloaded through
  `gensim.downloader` (about 66 MB, internet needed once). WMD is implemented from scratch with `scipy.optimize.linprog`
  and its value matches gensim's `wmdistance`.
* **Programs 22 / 23:** need TensorFlow (`pip install tensorflow`). Program 23 prints which words the model attends to.
* **Programs 24-30:** need PyTorch + Hugging Face `transformers` and an internet connection on first run to download the
  pre-trained models (distilbert 260 MB, gpt2 500 MB, Marian ~300 MB each, distilbart ~1.2 GB, DialoGPT-small ~350 MB).
  They run on CPU; a GPU makes Programs 24 and 26 faster. Use `--epochs`, `--model` etc. (see each file's docstring).
* **Programs 31 / 32:** documents are simulated as a stream of scanned pages (`common.make_page_stream`).
  Program 32 shows the benefit of the FSM: on a noisy stream the plain per-page classifier makes 25 illegal state
  transitions, the hybrid FSM + Viterbi decoder makes 0 and raises state accuracy from 0.89 to 0.98.
* **Datasets are small and synthetic/templated** so that every program runs in seconds on a laptop. Because of this,
  some classifiers reach ~100% accuracy; with real data (IMDB, 20 Newsgroups, ...) the same code gives realistic scores.
  To use a different dataset, replace the CSV in `data/` keeping the same column names.

## 6. Verification status

* Programs **1-23, 31 and 32** were executed end to end (Python 3.12, spaCy 3.8, NLTK 3.9, gensim 4.x, scikit-learn,
  TensorFlow 2.21 / Keras 3) and finished without errors; their outputs are in `outputs/`.
* Programs **24-30** were written against the current Hugging Face API and checked statically (syntax and undefined
  names), but could not be run in the authoring environment because it had no access to the Hugging Face model hub.
  **Run them once on your own machine (`python run_all.py 24 25 26 27 28 29 30`) before submitting, and keep the
  generated `outputs/*.txt` files** - that confirms they work with your installed library versions.

## 7. Uploading to GitHub

```bash
git init
git add .
git commit -m "Add all 32 NLP lab programs, datasets, outputs and README"
git branch -M main
git remote add origin https://github.com/<your-username>/NLP-Lab-Programs.git
git push -u origin main
```

Before submitting, check that: (1) the repository is **public** (or shared with the faculty), (2) the `outputs/` folder
is committed, (3) the repository opens in the browser and the README renders, (4) you keep the repository link ready
for verification. Large model folders (`outputs/24_finetuned_model`, `outputs/26_gpt2_finetuned`) are excluded by
`.gitignore` on purpose.

## 8. Course outcomes covered

* **CO1** - text pre-processing and linguistic analysis (tokenization, stemming, lemmatization, stop words, POS, parsing, chunking, NER).
* **CO2** - text representation and classical / neural NLP (BoW, TF-IDF, n-grams, embeddings, classification, topic models, IE, IR, RNN/LSTM, attention).
* **CO3** - transformer based NLP and applications (BERT, GPT-2, NER, translation, summarization, chatbots, document separation).

---
*Prepared for the NLP Lab, Department of Artificial Intelligence, NIET Greater Noida.*
