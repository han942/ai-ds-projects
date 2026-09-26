from __future__ import annotations

from datetime import date

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.compare_lightgcn import quarter_start
from rating_recsys.experiments.models import RecommendationQuery
from rating_recsys.retrieval.baselines import BaselineCandidateGenerator
from rating_recsys.retrieval.lightgcn import LightGCN, LightGCNConfig, LightGCNRRF


def interaction(review_id: int, user: int, item: int) -> Interaction:
    return Interaction(review_id, user, item, date(2025, 1, 2), 5.0,
                       "exact", f"item-{item}", "서울")


def test_lightgcn_is_deterministic_and_ranks_learned_item_above_negative() -> None:
    graph = tuple(
        interaction(index, user, item)
        for index, (user, item) in enumerate(
            ((1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 4), (4, 3), (4, 4)), 1
        )
    )
    config = LightGCNConfig(dimension=8, epochs=80, seed=13)
    first = LightGCN(config).fit(graph)
    second = LightGCN(config).fit(graph)
    scores = first.score(1, (2, 3))
    assert scores == second.score(1, (2, 3))
    assert scores[2] > scores[3]
    assert first.final_loss < 0.3


def test_rrf_includes_lightgcn_and_excludes_seen_items() -> None:
    history = interaction(1, 1, 1)
    target = interaction(20, 1, 3)
    query = RecommendationQuery("q", "test", 1, target.event_date, (history,), target, 2)
    graph = (history, interaction(2, 1, 2), interaction(3, 2, 2),
             interaction(4, 2, 3), interaction(5, 3, 3), interaction(6, 3, 4))
    model = LightGCN(LightGCNConfig(dimension=8, epochs=10)).fit(graph)
    result, context = BaselineCandidateGenerator(candidate_k=3).retrieve(query, graph)
    fused, graph_ids = LightGCNRRF(candidate_k=3).fuse(query, result, context, model)
    assert graph_ids
    assert 1 not in graph_ids
    assert 1 not in {candidate.restaurant_id for candidate in fused}
    assert any("lightgcn" in candidate.candidate_sources for candidate in fused)


def test_quarter_checkpoint_precedes_query_date() -> None:
    assert quarter_start(date(2025, 9, 30)) == date(2025, 7, 1)
    assert quarter_start(date(2025, 10, 1)) == date(2025, 10, 1)
