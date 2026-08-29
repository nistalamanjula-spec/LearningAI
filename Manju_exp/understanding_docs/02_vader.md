# VADER — Step 1 (lexicon + rules)

**VADER = Valence Aware Dictionary and sEntiment Reasoner**
(Hutto & Gilbert, 2014). Rule-based, no training. Ships inside NLTK; also a
standalone `vaderSentiment` package (what we use here).

## How it works

**1. The lexicon** — ~7,500 words each pre-scored from **−4 (very negative)** to
**+4 (very positive)**: e.g. `good +1.9`, `great +3.1`, `bad −2.5`, `okay +0.9`.
Base idea: look up each word, add scores.

**2. The rules ("reasoner")** — 5 heuristics that adjust for context:
- **Negation** — "not good" flips + dampens → negative. *(handles our test cases)*
- **Intensifiers** — "very good" > "good"; "marginally good" < "good".
- **Punctuation** — "good!!!" > "good".
- **Capitalization** — "GOOD" > "good".
- **Contrastive "but"** — weights the clause after "but" more.

## Output

`polarity_scores(text)` returns 4 numbers:
```
"The product is not bad" -> {neg: 0.0, neu: 0.58, pos: 0.42, compound: 0.43}
```
**`compound`** is the single normalized score, **−1 … +1**. Label thresholds:
```
compound >=  0.05  -> positive
compound <= -0.05  -> negative
otherwise          -> neutral
```

## Why we use it first
- Built-in **negation handling** — exactly the "not bad / not good" nuance we test.
- Gives a **continuous score**, so we *see* connotation gradients, not just labels.
- **Zero training, ~5 lines, instant.**

## Weaknesses (the ceiling that pushes us to Step 2)
Rule-based, not *learned* → blind to **sarcasm**, **domain slang**
("this phone is sick" = good), and struggles on long mixed reviews.

---

## Our first experiment

Script: `vader_exp/scripts/run_vader.py`
Input: `understanding_docs/sample_cases.txt` (connotation/negation probes)

Run:
```bash
uv run vader_exp/scripts/run_vader.py
# or on a custom file:
uv run vader_exp/scripts/run_vader.py path/to/cases.txt
```

### Results (2026-07-04)

| Sentence | Label | Compound | Note |
|---|---|---|---|
| the product is good | positive | +0.440 | baseline positive |
| product is average | neutral | 0.000 | "average" not in lexicon → dead neutral |
| product is not bad | positive | +0.431 | ✅ negation flipped "bad" → mildly positive |
| product is not good | negative | −0.341 | ✅ negation flipped "good" → mildly negative |
| product is bad | negative | −0.542 | baseline negative |

**Two things to savor:**
- `not bad` (+0.431) is slightly *less* positive than `good` (+0.440) — litotes softening.
- `not good` (−0.341) is *milder* than `bad` (−0.542) — correct, "not good" < "bad" in severity.

**Watch-out:** `average` scoring exactly 0 shows a lexicon limit — a word humans
read as "meh/neutral-ish" is simply *absent* from VADER's dictionary, so it
contributes nothing. Fine here, but a hint of why learned models eventually win.

---

## Environment
- Package manager: **uv** (`uv add vaderSentiment`), venv at `.venv/`.
- Dependency: `vadersentiment==3.3.2`.
