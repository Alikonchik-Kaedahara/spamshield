"""Make disguised copies of spam emails to test how easily spam filters are fooled.

Only spam rows (label == 1) are changed. Ham rows stay exactly the same.
Every test email is repeated once per attack, with an "attack" column:

    original  - unchanged email
    leet      - spam words get letters swapped for digits:   free -> fr3e, cash -> c45h
    spacing   - spam words get spaces between letters:      cash -> c a s h
    padding   - normal "ham" words are added to the end:    ... meeting project thanks ...
    combined  - leet or spacing on each spam word, plus padding

Use from the command line:
    python disguise_spam.py test.csv disguised.csv
    python disguise_spam.py test.csv disguised.csv --words-file top_spam_words.txt

Or from a notebook:
    from disguise_spam import make_disguised
    disguised = make_disguised(test_df, text_col="text", label_col="label")
"""
import argparse
import random
import re

import pandas as pd

# Common spam words. Better: pass the top spam words from your own
# Logistic Regression model with --words-file (one word per line).
DEFAULT_SPAM_WORDS = [
    "free", "cash", "money", "win", "winner", "won", "prize", "click", "offer",
    "buy", "cheap", "price", "credit", "loan", "deal", "discount", "limited",
    "urgent", "guarantee", "guaranteed", "bonus", "earn", "income", "save",
    "order", "viagra", "pills", "meds", "pharmacy", "rolex", "replica",
    "casino", "lottery", "claim", "reward", "investment", "stock", "dollars",
    "unsubscribe", "remove", "online", "now", "today", "act",
]

# Words typical of normal emails. Better: pass the top ham words from your
# own model with --ham-words-file.
DEFAULT_HAM_WORDS = [
    "meeting", "project", "thanks", "schedule", "attached", "report", "team",
    "tomorrow", "please", "review", "notes", "lecture", "office", "update",
    "agenda", "call", "monday", "draft", "budget", "regards",
]

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

ATTACKS = ["original", "leet", "spacing", "padding", "combined"]


def _word_pattern(words):
    words = sorted({w.strip().lower() for w in words if w.strip()}, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)


def leet(word, rng):
    """Swap about half of the swappable letters for digits (at least one)."""
    chars = list(word)
    spots = [i for i, c in enumerate(chars) if c.lower() in LEET]
    if not spots:
        return word
    for i in rng.sample(spots, k=max(1, len(spots) // 2)):
        chars[i] = LEET[chars[i].lower()]
    return "".join(chars)


def spacing(word, rng):
    return " ".join(word)


def either(word, rng):
    return leet(word, rng) if rng.random() < 0.5 else spacing(word, rng)


def _disguise_words(text, pattern, change, rng, rate):
    return pattern.sub(
        lambda m: change(m.group(0), rng) if rng.random() < rate else m.group(0), text
    )


def _pad(text, rng, ham_words, n):
    return text + " " + " ".join(rng.choices(ham_words, k=n))


def disguise(text, attack, pattern, rng, rate=1.0, ham_words=None, pad_words=30):
    ham_words = ham_words or DEFAULT_HAM_WORDS
    if attack == "original":
        return text
    if attack == "leet":
        return _disguise_words(text, pattern, leet, rng, rate)
    if attack == "spacing":
        return _disguise_words(text, pattern, spacing, rng, rate)
    if attack == "padding":
        return _pad(text, rng, ham_words, pad_words)
    if attack == "combined":
        return _pad(_disguise_words(text, pattern, either, rng, rate), rng, ham_words, pad_words)
    raise ValueError(f"unknown attack: {attack}")


def make_disguised(df, text_col="text", label_col="label", spam_words=None,
                   ham_words=None, rate=1.0, pad_words=30, attacks=ATTACKS, seed=42):
    """Return one copy of df per attack, stacked, with an "attack" column."""
    pattern = _word_pattern(spam_words or DEFAULT_SPAM_WORDS)
    rng = random.Random(seed)
    parts = []
    for attack in attacks:
        part = df.copy()
        is_spam = part[label_col] == 1
        part.loc[is_spam, text_col] = [
            disguise(str(t), attack, pattern, rng, rate, ham_words, pad_words)
            for t in part.loc[is_spam, text_col]
        ]
        part["attack"] = attack
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _read_words(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description="Create disguised spam test sets.")
    p.add_argument("input_csv", help="test set CSV with a text column and a 0/1 label column")
    p.add_argument("output_csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--words-file", help="spam words to disguise, one per line")
    p.add_argument("--ham-words-file", help="normal words used for padding, one per line")
    p.add_argument("--rate", type=float, default=1.0,
                   help="share of spam words to disguise, 0-1 (default 1 = all)")
    p.add_argument("--pad-words", type=int, default=30,
                   help="how many normal words to add in padding attacks (default 30)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    df = pd.read_csv(a.input_csv)
    out = make_disguised(
        df, a.text_col, a.label_col,
        spam_words=_read_words(a.words_file) if a.words_file else None,
        ham_words=_read_words(a.ham_words_file) if a.ham_words_file else None,
        rate=a.rate, pad_words=a.pad_words, seed=a.seed,
    )
    out.to_csv(a.output_csv, index=False)
    n_spam = int((df[a.label_col] == 1).sum())
    print(f"Wrote {len(out)} rows ({len(ATTACKS)} attacks x {len(df)} emails, "
          f"{n_spam} spam changed per attack) to {a.output_csv}")


if __name__ == "__main__":
    main()
