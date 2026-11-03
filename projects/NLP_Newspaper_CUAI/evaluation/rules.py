"""Rule-based metrics: number and name preservation, readability, speech level, and output defects."""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import textstat
from wordfreq import get_frequency_dict

from . import korean

RARE_ZIPF = 3.0
_FREQ: dict[str, float] | None = None


def _zipf(word: str) -> float | None:
    global _FREQ
    if _FREQ is None:
        _FREQ = get_frequency_dict("ko", wordlist="best")
    freq = _FREQ.get(word)
    return math.log10(freq * 1e9) if freq else None


def preservation(source: str, output: str) -> dict[str, float]:
    src_nums, out_nums = korean.numbers(source), korean.numbers(output)
    names = set(korean.nouns(source, tag="NNP"))
    kept_names = [n for n in names if n in output]
    return {
        "num_src_count": len(src_nums),
        "num_recall": len(src_nums & out_nums) / len(src_nums) if src_nums else np.nan,
        "num_all_kept": float(src_nums <= out_nums) if src_nums else np.nan,
        "num_novel": float(bool(out_nums - src_nums)),
        "name_src_count": len(names),
        "name_recall": len(kept_names) / len(names) if names else np.nan,
    }


def readability(text: str) -> dict[str, float]:
    sents = korean.sentences(text)
    n_sent = max(len(sents), 1)
    nouns = korean.nouns(text)
    zipfs = [_zipf(n) for n in nouns]
    registers = Counter(korean.sentence_register(s) for s in sents)
    out = {
        "chars": len(text),
        "sentences": len(sents),
        "eojeol_per_sent": len(text.split()) / n_sent,
        "morph_per_sent": len(korean.morphemes(text)) / n_sent,
        "rare_noun_ratio": sum(1 for z in zipfs if z is None or z < RARE_ZIPF) / len(zipfs) if zipfs else np.nan,
        "fkgl_textstat": textstat.flesch_kincaid_grade(text),
        "textstat_words_per_sent": textstat.lexicon_count(text) / max(textstat.sentence_count(text), 1),
    }
    for reg in korean.REGISTERS:
        out[f"reg_{reg}"] = registers.get(reg, 0) / n_sent
    return out


_HANJA = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def defects(source: str, output: str) -> dict[str, float]:
    eojeols = output.split()
    trigrams = Counter(tuple(eojeols[i : i + 3]) for i in range(len(eojeols) - 2))
    return {
        "identical_to_source": float(output.strip() == source.strip()),
        "copy_similarity": SequenceMatcher(None, source, output).ratio(),
        "unfinished": float(korean.looks_truncated(output)),
        "repetition_loop": float(bool(trigrams) and max(trigrams.values()) >= 3),
        "hanja_chars": float(len(_HANJA.findall(output))),
        "latin_ratio": len(_LATIN.findall(output)) / max(len(output), 1),
    }


def per_sample(sources: list[str], outputs: list[str]) -> pd.DataFrame:
    rows = []
    for src, out in zip(sources, outputs):
        rows.append({**preservation(src, out), **readability(out), **defects(src, out)})
    return pd.DataFrame(rows)
