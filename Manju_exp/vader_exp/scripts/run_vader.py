"""
Step 1 of our sentiment-analysis staircase: VADER (lexicon + rules).

Reads sentences from a text file (one per line, an optional leading "- " or
numbering is stripped) and prints VADER's sentiment for each, so we can play
with connotation / negation cases like "not bad" vs "not good".

Run:
    uv run vader_exp/scripts/run_vader.py
    uv run vader_exp/scripts/run_vader.py path/to/other_cases.txt
"""

import sys
import re
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Default file to read if none is passed on the command line.
DEFAULT_CASES = (
    Path(__file__).resolve().parents[2]
    / "understanding_docs"
    / "sample_cases.txt"
)


def label_for(compound: float) -> str:
    """Turn VADER's compound score (-1..+1) into a label using the
    standard thresholds recommended by VADER's authors."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def clean(line: str) -> str:
    """Strip list markers like '- ' or '1. ' and surrounding whitespace."""
    line = line.strip()
    line = re.sub(r"^[-*\d.)\s]+", "", line)  # drop leading bullets / numbers
    return line.strip()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CASES
    if not path.exists():
        sys.exit(f"File not found: {path}")

    analyzer = SentimentIntensityAnalyzer()

    print(f"Analyzing: {path}\n")
    header = f"{'sentence':40} | {'label':8} | {'compound':>8} | {'pos':>4} {'neu':>4} {'neg':>4}"
    print(header)
    print("-" * len(header))

    for raw in path.read_text(encoding="utf-8").splitlines():
        sentence = clean(raw)
        if not sentence or sentence.lower() == "etc":
            continue

        scores = analyzer.polarity_scores(sentence)
        label = label_for(scores["compound"])
        print(
            f"{sentence:40} | {label:8} | {scores['compound']:+8.3f} | "
            f"{scores['pos']:.2f} {scores['neu']:.2f} {scores['neg']:.2f}"
        )


if __name__ == "__main__":
    main()
