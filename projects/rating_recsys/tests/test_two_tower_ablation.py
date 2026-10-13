from types import SimpleNamespace

import pytest

from rating_recsys.experiments.two_tower_ablation_cli import conditional_seed_mean_change


def test_seed_averaged_bootstrap_resamples_users_not_seed_rows():
    queries = [SimpleNamespace(query_id=str(i), user_id=i, relevant=True, relevance_by_item={10: 2})
               for i in (1, 2)]
    bad = {"1": (), "2": ()}
    good = {"1": (10,), "2": (10,)}
    result = conditional_seed_mean_change(queries, [(bad, good), (good, bad)], 100, "recall", samples=30)
    assert result["paired_users"] == 2
    assert result["training_seeds"] == 2
    assert result["mean_difference"] == pytest.approx(0)
    assert result["bootstrap_95pct"] == [0, 0]
