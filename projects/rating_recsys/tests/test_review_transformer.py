"""Leakage, missing text, frozen inputs and recommendation execution contracts."""
from dataclasses import replace
from datetime import date
import hashlib

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.retrieval.review_embeddings import E5_MODEL, ProfileFormatter, model_embedding_config
from rating_recsys.retrieval.review_transformer import (
    ReviewAttentionNetwork, ReviewBank, ReviewStore, ReviewTransformer,
    ReviewTransformerConfig, review_blocks,
)


class Cache:
    usage = {"api_requests": 0}
    def missing_count(self, docs):
        return len(set(docs))
    def embed(self, docs):
        values = {}
        for doc in docs:
            seed = int.from_bytes(hashlib.sha256(doc.encode()).digest()[:4], "little")
            value = np.random.default_rng(seed).normal(size=384).astype(np.float32)
            values[doc] = value / np.linalg.norm(value)
        return values


def row(rid, user, item, month, rating=5):
    return Interaction(rid, user, item, date(2025, month, 10), rating, "exact", "synthetic", "region")


def setup(tmp_path):
    cfg = ReviewTransformerConfig(width=16, heads=2, layers=1, max_user_reviews=3,
                                  max_item_reviews=4, max_blocks=2, block_tokens=64,
                                  epochs=2, threads=1, batch_size=4, negative_samples=2,
                                  candidate_k=5, dropout=0)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    embedding = replace(model_embedding_config(E5_MODEL, str(tmp_path / "unused")), max_document_tokens=64)
    formatter = ProfileFormatter(embedding, tmp_path, tokenizer=tokenizer)
    rows = [row(i, i, i * 10, 1) for i in range(1, 7)]
    rows += [row(7, 1, 20, 4), row(8, 1, 30, 4, 3), row(9, 1, 40, 4, 1),
             row(10, 2, 50, 4), row(11, 3, 60, 4), row(12, 4, 10, 4),
             row(13, 5, 20, 4), row(14, 6, 30, 4)]
    texts = {r.review_id: f"review-{r.review_id} 맛있었다. 가격은 비쌌다." for r in rows}
    texts[4] = ""
    exp = ExperimentConfig(ranker_training_mode="window", satisfaction_mode="history-aware")
    store = ReviewStore.prepare(rows, texts, formatter, Cache(), cfg)
    return cfg, formatter, rows, texts, store, exp


@pytest.mark.parametrize("kwargs", [{"width": 7}, {"epochs": 0}, {"dropout": 1},
                                    {"block_tokens": 513}, {"temperature": float("nan")},
                                    {"window_months": 5}, {"negative_samples": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ReviewTransformerConfig(**kwargs)


def test_blocks_preserve_contrast_and_audit_capped_long_review(tmp_path):
    cfg, formatter, *_ = setup(tmp_path)
    text = "국물은 좋았지만 고기는 질겼다. 가격은 비쌌다. 직원은 친절했다."
    blocks, audit = review_blocks(text, formatter, cfg)
    assert len(blocks) == 1
    assert "좋았지만" in blocks[0]["text"] and "질겼다" in blocks[0]["text"]
    text = " ".join(f"단어{i}" for i in range(200))
    blocks, audit = review_blocks(text, formatter, cfg)
    assert len(blocks) == 2 and audit["available_blocks"] > 2
    assert audit["retained_chars"] < audit["source_chars"]
    assert blocks[-1]["end"] == len(text)
    for block in blocks:
        assert block["text"] == text[block["start"]:block["end"]]
        assert len(formatter.tokenizer.encode(formatter.format(block["text"], "document")).ids) <= cfg.block_tokens


def test_bank_is_cutoff_safe_on_both_sides_and_keeps_low_rating(tmp_path):
    cfg, _, rows, texts, store, _ = setup(tmp_path)
    cutoff = date(2025, 3, 31)
    first = ReviewBank.build(rows, texts, store, [1, 4], cutoff, cfg)
    assert set(first.audit["selected_review_ids"]) <= set(range(1, 7))
    texts = {**texts, 7: "changed future review"}
    second = ReviewBank.build(rows + [row(99, 1, 999, 7)], texts, store, [1, 4], cutoff, cfg)
    assert torch.equal(first.user_reviews, second.user_reviews)
    assert torch.equal(first.item_reviews, second.item_reviews)
    assert torch.equal(first.user_metadata, second.user_metadata)
    assert not first.user_reviews[1].any()
    later = ReviewBank.build(rows, texts, store, [1], date(2025, 4, 30), cfg)
    assert 9 in later.audit["selected_review_ids"]
    assert later.user_metadata[0, 0, 0] == -1


def test_all_padding_and_unknown_ids_are_finite_and_ignore_padding_values(tmp_path):
    cfg, _, rows, texts, store, _ = setup(tmp_path)
    torch.set_num_threads(1)
    network = ReviewAttentionNetwork(6, 6, cfg).eval()
    bank = ReviewBank.build(rows[:6], texts, store, [4, 999], date(2025, 3, 31), cfg)
    refs, meta = bank.user_reviews, bank.user_metadata
    with torch.no_grad():
        u = network.encode("user", torch.zeros(2, dtype=torch.long), refs, meta, store)
        v = network.encode("user", torch.zeros(2, dtype=torch.long), refs, torch.full_like(meta, 999), store)
        item = network.encode("item", torch.tensor([0, 0]), bank.item_reviews[:2], bank.item_metadata[:2], store)
        scores = network.match(u, item)
    assert all(torch.isfinite(x).all() for x in (*u, scores))
    assert torch.equal(u[0], v[0]) and torch.equal(u[-1], v[-1])
    assert not u[0].any()


def test_training_negative_contract_frozen_vectors_and_save_load(tmp_path, monkeypatch):
    cfg, _, rows, texts, store, exp = setup(tmp_path)
    import rating_recsys.retrieval.review_transformer as module
    original = module._sample_negatives
    draws = []
    def checked(rng, users, items, forbidden):
        sampled = original(rng, users, items, forbidden)
        assert not np.isin(users * items + sampled, forbidden).any()
        # First user's seen=10, positive=20/30, low-grade observed=40.
        assert not set(sampled[users == 0]) & {0, 1, 2, 3}
        draws.append(sampled)
        return sampled
    monkeypatch.setattr(module, "_sample_negatives", checked)
    before = store.vectors.clone()
    model = ReviewTransformer(cfg, store, exp).fit(rows, texts, cutoff=date(2025, 4, 30), eval_users=[1, 4, 999])
    assert draws and torch.equal(before, store.vectors) and not store.vectors.requires_grad
    assert model.history[-1]["observed_loss"] > 0
    assert all(a["latest_history_date"] <= a["cutoff"] for a in model.training_audit)
    retrieval, ranked = model.rankings([1, 4, 999], {1: [10]}, 5)
    assert 10 not in ranked[1] and ranked[999]
    assert all(len(v) == len(set(v)) for v in ranked.values())
    assert all(set(ranked[u]) <= set(retrieval[u]) for u in ranked)
    path = tmp_path / "model.pt"
    model.save(path)
    restored = ReviewTransformer.load(path)
    assert restored.rankings([1, 4, 999], {1: [10]}, 5) == (retrieval, ranked)
    assert restored.metadata["api_requests"] == 0


def test_fully_blocked_query_is_skipped_without_negative_sampling(tmp_path):
    cfg, formatter, _, _, _, exp = setup(tmp_path)
    rows = [row(1, 1, 10, 1), row(2, 2, 20, 1), row(3, 1, 20, 4)]
    texts = {r.review_id: "맛있었다" for r in rows}
    store = ReviewStore.prepare(rows, texts, formatter, Cache(), cfg)
    with pytest.raises(ValueError, match="No eligible"):
        ReviewTransformer(cfg, store, exp).fit(rows, texts, cutoff=date(2025, 4, 30), eval_users=[1])


def test_nearest_observed_lower_grade_and_token_budget_contract(tmp_path):
    cfg, formatter, rows, texts, store, exp = setup(tmp_path)
    windows = ReviewTransformer(cfg, store, exp).build_windows(rows, texts)
    w = windows[0]
    indices = np.where(w.users == 0)[0]
    assert list(w.grades[indices]) == [2, 1]
    assert list(w.lower_grades[indices]) == [1, 0]
    with pytest.raises(ValueError, match="token budget"):
        ReviewStore.prepare(rows, texts, formatter, Cache(), replace(cfg, block_tokens=128))


def test_cli_rejects_too_small_candidate_pool_before_loading_data():
    from rating_recsys.experiments.review_transformer_cli import main
    with pytest.raises(ValueError, match="at least 100"):
        main(["--candidate-k", "99"])


def test_stopping_waits_for_minimum_epochs_and_resets_on_improvement():
    from rating_recsys.experiments.review_transformer_cli import should_stop
    assert not should_stop(7, 2, min_epochs=10, patience=5)
    assert should_stop(10, 2, min_epochs=10, patience=5)
    assert not should_stop(10, 9, min_epochs=10, patience=5)
    assert not should_stop(20, 2, min_epochs=1, patience=None)


@pytest.mark.parametrize("args", [["--patience", "0"], ["--min-epochs", "4", "--epochs", "3"]])
def test_cli_rejects_invalid_stopping_before_loading_data(args):
    from rating_recsys.experiments.review_transformer_cli import main
    with pytest.raises(ValueError):
        main(args)


def test_resume_preserves_optimizer_negatives_and_dropout_sequence(tmp_path):
    cfg, _, rows, texts, store, exp = setup(tmp_path)
    cfg = replace(cfg, epochs=3, dropout=.1, id_dropout=.2)
    kwargs = {"cutoff":date(2025,4,30), "eval_users":[1,4,999]}
    full = ReviewTransformer(cfg,store,exp).fit(rows,texts,**kwargs)
    last = tmp_path / "last.pt"
    first = ReviewTransformer(replace(cfg,epochs=1),store,exp).fit(
        rows,texts,training_checkpoint=last,**kwargs)
    resumed = ReviewTransformer(cfg,store,exp).fit(rows,texts,resume_from=last,**kwargs)
    assert len(first.history) == 1 and len(resumed.history) == 3
    for name,tensor in full.network.state_dict().items():
        assert torch.equal(tensor,resumed.network.state_dict()[name]), name
    for a,b in zip(full.history,resumed.history):
        assert {k:v for k,v in a.items() if k != "seconds"} == {k:v for k,v in b.items() if k != "seconds"}
    assert full.rankings([1,4,999],{1:[10]},5) == resumed.rankings([1,4,999],{1:[10]},5)
    with pytest.raises(ValueError,match="contract differs"):
        ReviewTransformer(replace(cfg,learning_rate=.002),store,exp).fit(rows,texts,resume_from=last,**kwargs)
    changed_vectors = store.vectors.clone()
    changed_vectors[1,0] += .001
    changed_store = replace(store,vectors=changed_vectors)
    with pytest.raises(ValueError,match="contract differs"):
        ReviewTransformer(cfg,changed_store,exp).fit(rows,texts,resume_from=last,**kwargs)


def test_interrupted_evaluation_restores_last_committed_state(tmp_path):
    from rating_recsys.experiments.review_transformer_cli import resume_state,alias_file
    cfg, _, rows, texts, store, exp = setup(tmp_path)
    cfg = replace(cfg,epochs=3,dropout=.1)
    kwargs = {"cutoff":date(2025,4,30), "eval_users":[1,4,999]}
    full = ReviewTransformer(cfg,store,exp).fit(rows,texts,**kwargs)
    state = {"epochs":[], "selected_epoch":1, "candidates_file":"epoch1.jsonl"}
    original = tmp_path / "epoch1.jsonl"
    original.write_text('{"committed":true}\n')
    alias_file(original,tmp_path / "candidates_validation.jsonl")
    def evaluate(stats,model):
        state["epochs"].append(stats)
        if stats["epoch"] == 2:
            state["selected_epoch"] = 2
            (tmp_path / "epoch2.jsonl.tmp").write_text('{"partial":')
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        ReviewTransformer(cfg,store,exp).fit(rows,texts,callback=evaluate,
            training_checkpoint=tmp_path / "training_last.pt",checkpoint_context=lambda:state,**kwargs)
    committed,epoch = resume_state(tmp_path)
    assert epoch == 1 and len(committed["epochs"]) == 1 and committed["selected_epoch"] == 1
    assert (tmp_path / committed["candidates_file"]).read_text() == original.read_text()
    resumed = ReviewTransformer(cfg,store,exp).fit(rows,texts,resume_from=tmp_path / "training_last.pt",**kwargs)
    for name,tensor in full.network.state_dict().items():
        assert torch.equal(tensor,resumed.network.state_dict()[name]), name
