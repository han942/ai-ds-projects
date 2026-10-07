import hashlib
import json

import pytest

from rating_recsys.experiments.review_ltr_diagnostics import diagnose_rankings


def row(query, grades, candidates, baseline=None, review=None):
    return {"query_id": query, "user_id": 1, "relevance_by_item": grades,
            "c5_candidates": candidates, "baseline": baseline or candidates,
            "review_ltr": review or candidates}


def test_global_idcg_differs_without_changing_candidate_pool():
    result = diagnose_rankings([row("a", {1: 2, 9: 2}, [2, 1], review=[1, 2]),
                               row("b", {9: 1}, [1, 2])], cutoff=2, samples=20)
    assert result["recoverable_queries"] == 1
    assert result["unrecoverable_query_fraction"] == 0.5
    assert result["early_stopping_diagnostic"]["different_idcg_queries"] == 1
    assert result["metrics"]["oracle"]["recall"] == 0.25
    assert result["metrics"]["review_ltr"]["ndcg"] < 0.5
    assert result["early_stopping_diagnostic"]["candidate_normalized_ndcg"]["review_ltr"] == 1


def test_observed_negative_is_distinct_from_unknown_and_empty_positive_query():
    result = diagnose_rankings([row("a", {1: 2, 2: 0}, [1, 2, 3]),
                               row("b", {4: 0}, [4, 5])], samples=20)
    assert result["excluded_no_relevant"] == 1
    assert result["observed_candidate_rows"] == 3
    assert result["unobserved_candidate_rows"] == 2
    assert result["observed_low_grade_rows"] == 2
    assert result["retrieved_positive_rows"] == 1


def test_changed_or_duplicated_candidate_pool_is_rejected():
    with pytest.raises(ValueError, match="candidate pool"):
        diagnose_rankings([row("a", {1: 2}, [1, 2], review=[1, 3])])
    with pytest.raises(ValueError, match="Duplicate candidates"):
        diagnose_rankings([row("a", {1: 2}, [1, 1])])


def test_no_relevance_and_duplicate_query_are_rejected():
    with pytest.raises(ValueError, match="No relevant"):
        diagnose_rankings([row("a", {1: 0}, [1, 2])])
    with pytest.raises(ValueError, match="Duplicate query"):
        diagnose_rankings([row("a", {1: 2}, [1, 2])] * 2)


def test_paired_comparison_is_reproducible_and_oracle_bounds_ndcg():
    records = [row("a", {1: 2}, [1, 2], baseline=[2, 1]),
               row("b", {2: 1}, [1, 2], review=[2, 1])]
    result = diagnose_rankings(records, samples=20)
    assert result == diagnose_rankings(records, samples=20)
    assert result["metrics"]["oracle"]["ndcg"] >= result["metrics"]["review_ltr"]["ndcg"]


def test_cli_defaults_to_validation_without_loading_test(tmp_path):
    from rating_recsys.experiments.review_ltr_diagnostics import main

    records = [row("a", {1: 2}, [1, 2])]
    metrics = diagnose_rankings(records, samples=2)["metrics"]
    expected = {"validation": {arm: {f"{name}_at_10": value for name, value in metrics[arm].items()}
                               for arm in ("baseline", "review_ltr")}}
    (tmp_path / "metrics.json").write_text(json.dumps(expected))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "fixture", "config": {"ranking_k": 10, "random_seed": 42}}))
    (tmp_path / "status.json").write_text('{"status":"complete"}')
    (tmp_path / "recommendations_validation.jsonl").write_text(json.dumps(records[0]) + "\n")
    output = tmp_path / "diagnostic.json"
    assert main(["--run-dir", str(tmp_path), "--output", str(output), "--samples", "2"]) == 0
    assert set(json.loads(output.read_text())["phases"]) == {"validation"}


def test_cli_rejects_changed_test_c5_order_even_if_ranker_metrics_match(tmp_path):
    from rating_recsys.experiments.review_ltr_diagnostics import main

    records = [row("a", {1: 2}, [1, 2])]
    metrics = diagnose_rankings(records, samples=2)["metrics"]
    arms = {arm: {f"{name}_at_10": value for name, value in metrics[arm].items()}
            for arm in ("baseline", "review_ltr")}
    expected = {"validation": arms, "test": arms,
                "r0": {f"{name}_at_10": value for name, value in metrics["c5_candidates"].items()}}
    (tmp_path / "metrics.json").write_text(json.dumps(expected))
    order_hash = hashlib.sha256(json.dumps({"a": [1, 2]}, sort_keys=True).encode()).hexdigest()
    manifest = {"run_id": "fixture", "config": {"ranking_k": 10, "random_seed": 42},
                "shared_test_candidates_sha256": order_hash}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "status.json").write_text('{"status":"complete"}')
    (tmp_path / "recommendations_validation.jsonl").write_text(json.dumps(records[0]) + "\n")
    tampered = {**records[0], "c5_candidates": [2, 1]}
    (tmp_path / "recommendations_test.jsonl").write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValueError, match="C5 order differs"):
        main(["--run-dir", str(tmp_path), "--output", str(tmp_path / "output.json"),
              "--samples", "2", "--include-test"])


def test_review_signal_reads_cache_only_and_matches_actual_key(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from rating_recsys.experiments import snapshot as sources
    from rating_recsys.experiments.review_ltr_diagnostics import diagnose_review_signal
    from rating_recsys.retrieval.review_embeddings import (
        E5_MODEL, LocalEmbeddingCache, ProfileFormatter, model_embedding_config,
    )
    from support import _interaction

    rows = [_interaction(1, 1, 100, 1), _interaction(2, 2, 200, 1)]
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(tokenizer.to_str())
    cache_path = tmp_path / "cache.sqlite"
    embedding = model_embedding_config(E5_MODEL, str(cache_path))
    formatter = ProfileFormatter(embedding, tmp_path, tokenizer=tokenizer)
    docs = [formatter.format("alpha", "query"), formatter.format("alpha", "document"),
            formatter.format("beta", "document")]
    vector = [1.0] + [0.0] * 383
    with LocalEmbeddingCache(cache_path, embedding) as cache:
        cache.db.executemany("INSERT INTO vectors VALUES (?,?,?,?)", [
            (cache._key(doc), json.dumps(vector), embedding.model, 384) for doc in docs])
        cache.db.commit()
    before = cache_path.read_bytes()
    monkeypatch.setattr(sources, "load_snapshot", lambda _: rows)
    monkeypatch.setattr(sources, "load_review_texts", lambda *_: ({1: "alpha", 2: "beta"},
                                                              {"artifact_sha256": "texts"}))
    import rating_recsys.datasets.split as splitting
    monkeypatch.setattr(splitting, "build_global_temporal_split", lambda *_args, **_kwargs:
                        SimpleNamespace(train=rows, train_cutoff=rows[0].event_date))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Diagnostic must not load an encoder")
    monkeypatch.setattr(LocalEmbeddingCache, "_load_encoder", forbidden)
    manifest = {"embedding": embedding.to_dict(), "config": {"train_fraction": .8, "validation_fraction": .1},
                "snapshot": {"dataset_snapshot_id": sources.dataset_digest(rows)},
                "review_texts": {"artifact_sha256": "texts"},
                "preprocessing": {"tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()}}
    records = [{**row("validation:u1", {200: 2}, [100, 200]),
                "cutoff": rows[0].event_date.isoformat()}]
    result = diagnose_review_signal(manifest, records, snapshot=tmp_path / "snapshot.jsonl",
                                   review_cache=cache_path, tokenizer_path=tokenizer_path)
    assert result["within_query_pairwise_auc_macro"] == .5
    assert result["pair_coverage"] == 1
    assert result["encoder_loaded"] is False
    assert cache_path.read_bytes() == before
    bad = {**manifest, "snapshot": {"dataset_snapshot_id": "other"}}
    with pytest.raises(ValueError, match="Snapshot differs"):
        diagnose_review_signal(bad, records, snapshot=tmp_path / "snapshot.jsonl",
                               review_cache=cache_path, tokenizer_path=tokenizer_path)
