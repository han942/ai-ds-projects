"""SARI with its keep/delete/add components.

The n-gram scoring follows the Hugging Face `evaluate` SARI metric (Apache-2.0),
itself based on Xu et al. (2016) and tensor2tensor, so the default whitespace +
13a tokenization reproduces the score reported in the original notebook.
"""

from __future__ import annotations

from collections import Counter

import sacrebleu

_TOKENIZE_13A = sacrebleu.metrics.bleu._get_tokenizer("13a")()


def normalize(sentence: str) -> str:
    return _TOKENIZE_13A(sentence.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _sari_ngram(sgrams, cgrams, rgramslist, numref):
    rgramcounter = Counter(g for rgrams in rgramslist for g in rgrams)
    sgramcounter_rep = Counter({g: c * numref for g, c in Counter(sgrams).items()})
    cgramcounter_rep = Counter({g: c * numref for g, c in Counter(cgrams).items()})

    keep_rep = sgramcounter_rep & cgramcounter_rep
    keep_good = keep_rep & rgramcounter
    keep_all = sgramcounter_rep & rgramcounter
    keep_p = sum(keep_good[g] / keep_rep[g] for g in keep_good) / len(keep_rep) if keep_rep else 1
    keep_r = sum(keep_good.values()) / sum(keep_all.values()) if keep_all else 1
    keep = 2 * keep_p * keep_r / (keep_p + keep_r) if (keep_p or keep_r) else 0

    del_rep = sgramcounter_rep - cgramcounter_rep
    del_good = del_rep - rgramcounter
    delete = sum(del_good[g] / del_rep[g] for g in del_good) / len(del_rep) if del_rep else 1

    added = set(cgrams) - set(sgrams)
    added_good = added & set(rgramcounter)
    added_all = set(rgramcounter) - set(sgrams)
    add_p = len(added_good) / len(added) if added else 1
    add_r = len(added_good) / len(added_all) if added_all else 1
    add = 2 * add_p * add_r / (add_p + add_r) if (add_p or add_r) else 0
    return keep, delete, add


def sari_sentence(source: str, prediction: str, references: list[str]) -> dict[str, float]:
    """Sentence SARI (0-100) and its components. Inputs are tokenized by whitespace after `normalize`."""
    s1 = normalize(source).split(" ")
    c1 = normalize(prediction).split(" ")
    r1 = [normalize(r).split(" ") for r in references]
    keep, delete, add = [], [], []
    for n in range(1, 5):
        k, d, a = _sari_ngram(_ngrams(s1, n), _ngrams(c1, n), [_ngrams(r, n) for r in r1], len(r1))
        keep.append(k), delete.append(d), add.append(a)
    keep_s, del_s, add_s = (100 * sum(x) / 4 for x in (keep, delete, add))
    return {"sari": (keep_s + del_s + add_s) / 3, "sari_keep": keep_s, "sari_del": del_s, "sari_add": add_s}
