# Sentiment Analysis — Understanding Notes

Goal of the project: do sentiment analysis on product reviews (e.g. Flipkart).
Approach: **build understanding first, then climb from easy → hard implementations**,
measuring at each step so we can justify our final choice.

---

## 1. What "sentiment analysis" means

Given a text like *"Battery drains fast but the camera is amazing"*, output a
sentiment: **positive / negative / neutral** (sometimes a score, or a predicted
star rating).

### Granularity — how detailed do we want to be?

| Level | Output | Example |
|---|---|---|
| Document-level | one label for the whole review | "mixed / neutral" |
| Sentence-level | label per sentence | s1 = neg, s2 = pos |
| Aspect-based (ABSA) | sentiment per *feature* | battery = neg, camera = pos |

Aspect-based is the most valuable for reviews but the hardest.

---

## 2. The families of approaches (our staircase)

| Step | Approach | Library | Idea | Trade-off |
|---|---|---|---|---|
| 1 | Lexicon / rule-based (VADER, TextBlob) | NLTK | dictionary of word scores + rules | zero training, explainable; weak on sarcasm/slang |
| 2 | Classical ML (TF-IDF + LogReg/NB/SVM) | scikit-learn | turn text into numbers, train a classifier | good baseline; needs labeled data |
| 3 | Deep learning (LSTM/CNN + embeddings) | — | captures word order/context | mostly superseded by transformers |
| 4 | Transformers (BERT, RoBERTa, DistilBERT) | Hugging Face | fine-tune a pretrained language model | SOTA accuracy; heavier compute |
| 5 | LLM (Claude/GPT via API) | API | just *ask* the model | near-zero setup, aspect-aware; per-call cost |

Effort/cost and accuracy both rise left → right. Climbing gives us **a baseline to
beat** and lets us *show* the progression (e.g. VADER 70% → TF-IDF 82% → BERT 88%).

Realistically, easy → hard usually buys **~5–15% accuracy** for a lot more
complexity. For an assignment, the *progression itself* is often what's being judged.

### Key libraries (vocabulary)
- **NLTK** = Natural Language Toolkit — text prep (tokenizing, stopwords,
  stemming/lemmatization, POS tagging) + ships VADER.
- **scikit-learn (sklearn)** — classical ML.
- **Hugging Face Transformers** — BERT-family models.
- **pandas** — load/handle review data (spreadsheets in code).

### Flipkart-specific note
Reviews come **with a star rating**, which we can use as *free labels*:
4–5★ = positive, 1–2★ = negative, 3★ = neutral. Solves "where do I get labels?".

---

## 3. How we judge / compare models (evaluation)

Two modes:
- **Qualitative** — eyeball cherry-picked examples. Good for intuition,
  doesn't scale, subjective.
- **Quantitative** — a number over lots of data. Scalable, objective, comparable.
  This is what lets us say "Model B beats Model A".

### Prerequisite: ground truth
Every quantitative metric needs the **correct answer (labels)** to compare
predictions against. (Flipkart stars give this for free.)

### The confusion matrix (positive vs negative)

|  | Model says Positive | Model says Negative |
|---|---|---|
| Actually Positive | True Positive (TP) | False Negative (FN) |
| Actually Negative | False Positive (FP) | True Negative (TN) |

### The metrics
- **Accuracy** = (TP+TN) / all — "fraction correct". Has a trap (below).
- **Precision** = TP / (TP+FP) — "when it says positive, how often right?" (few false alarms)
- **Recall** = TP / (TP+FN) — "of all real positives, how many caught?" (misses little)
- **F1** = 2·(P·R)/(P+R) — balance of precision & recall. **Usually the headline number.**

### The accuracy trap (why F1 exists)
If 90% of reviews are positive, a model that *always* says "positive" gets
**90% accuracy** but **0% recall on negatives** — useless. This is **class
imbalance**, and it's why we report **F1, not just accuracy**.

### Multi-class (pos/neg/neutral)
Compute precision/recall/F1 **per class**, then average:
- **Macro-average** — each class equal (good when the rare negative class matters).
- **Weighted-average** — weight by class size.
`sklearn.classification_report` prints all of this in one call.

### The sacred rule: train/test split
Always measure on data the model **never saw**. Split ~80% train / ~20% test.
Testing on training data = fake score ("overfitting"). Lexicon models (VADER)
have no training but we still evaluate on a held-out set so it's comparable.

**Bottom line:** compare on **F1-score** (report accuracy alongside + the
confusion matrix for diagnosis), on a **held-out test set**, vs **ground truth**.

---

## 4. Why we start with VADER

See [02_vader.md](02_vader.md).
