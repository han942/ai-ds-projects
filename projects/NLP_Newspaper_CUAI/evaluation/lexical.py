"""Reference-based lexical metrics: SARI, BLEU, chrF++, and ROUGE.

`*_ws` metrics reuse the original notebook's whitespace tokenization so the
reported numbers can be reproduced. `*_morph` metrics tokenize with Kiwi
morphemes, which suits agglutinative Korean better than whitespace eojeols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rouge import Rouge
from sacrebleu.metrics import BLEU, CHRF

from .korean import morph_string
from .sari import sari_sentence

_BLEU = BLEU(tokenize="none", effective_order=True)
_CHRF = CHRF(word_order=2)


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(prediction: list[str], reference: list[str]) -> float:
    lcs = _lcs(prediction, reference)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(prediction), lcs / len(reference)
    return 2 * p * r / (p + r)


def per_sample(sources: list[str], predictions: list[str], references: list[str]) -> pd.DataFrame:
    rows = []
    rouge = Rouge()
    for src, pred, ref in zip(sources, predictions, references):
        src_m, pred_m, ref_m = morph_string(src), morph_string(pred), morph_string(ref)
        ws = sari_sentence(src, pred, [ref])
        mo = sari_sentence(src_m, pred_m, [ref_m])
        rows.append({
            "sari_ws": ws["sari"],
            "sari_morph": mo["sari"],
            "sari_morph_keep": mo["sari_keep"],
            "sari_morph_del": mo["sari_del"],
            "sari_morph_add": mo["sari_add"],
            "bleu_morph": _BLEU.sentence_score(pred_m, [ref_m]).score,
            "chrf_pp": _CHRF.sentence_score(pred, [ref]).score,
            "rouge_l_ws": rouge.get_scores(pred, ref)[0]["rouge-l"]["f"] if pred.strip() and ref.strip() else 0.0,
            "rouge_l_morph": rouge_l_f1(pred_m.split(), ref_m.split()),
        })
    return pd.DataFrame(rows)


def corpus(sources: list[str], predictions: list[str], references: list[str]) -> dict[str, float]:
    preds_m = [morph_string(p) for p in predictions]
    refs_m = [morph_string(r) for r in references]
    return {
        "bleu_morph_corpus": BLEU(tokenize="none").corpus_score(preds_m, [refs_m]).score,
        "chrf_pp_corpus": CHRF(word_order=2).corpus_score(predictions, [references]).score,
    }


def notebook_reproduction(sources: list[str], predictions: list[str], references: list[str]) -> dict[str, float]:
    """The original notebook's metrics, computed the same way it did."""
    import evaluate

    scores = Rouge().get_scores(predictions, references, avg=True)
    bleu = evaluate.load("bleu")
    return {
        "rouge1_ws": scores["rouge-1"]["f"],
        "rouge2_ws": scores["rouge-2"]["f"],
        "rougeL_ws": scores["rouge-l"]["f"],
        "bleu1_vs_source": bleu.compute(predictions=predictions, references=[[s] for s in sources], max_order=1, smooth=True)["bleu"],
        "bleu1_vs_reference": bleu.compute(predictions=predictions, references=[[r] for r in references], max_order=1, smooth=True)["bleu"],
        "sari_ws": float(np.mean([sari_sentence(s, p, [r])["sari"] for s, p, r in zip(sources, predictions, references)])),
    }
