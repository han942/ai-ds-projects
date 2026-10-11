from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from rating_recsys.retrieval.aspect_graph import (
    CHANNELS, AspectGraphLightGCN, build_opinion_profiles, mean_opinion_profiles, opinion_input,
)
from rating_recsys.retrieval.lightgcn import LightGCN, LightGCNConfig
from rating_recsys.retrieval.rlmrec import RLMRecConfig


def test_opinions_preserve_channels_and_missing_neutral():
    rows = [SimpleNamespace(review_id=1, user_id=1, restaurant_id=10, event_date=date(2025, 1, 1)),
            SimpleNamespace(review_id=2, user_id=1, restaurant_id=20, event_date=date(2025, 1, 2))]
    plus = {"category": "taste", "polarity": "positive", "target": "국물", "opinion": "좋다", "evidence": "국물 좋다"}
    minus = {**plus, "polarity": "negative", "target": "고기", "opinion": "질기다", "evidence": "고기 질기다"}
    neutral = {**plus, "category": "portion", "polarity": "neutral", "target": "양", "opinion": "많다", "evidence": "양 많다"}
    vectors = {opinion_input(record, role): np.eye(384, dtype=np.float32)[j]
               for j, record in enumerate((plus, minus, neutral)) for role in ("query", "passage")}
    users, items, audit = build_opinion_profiles(rows, date(2025, 1, 2), {1: 'x', 2: 'y'},
                                               {1: [plus, minus], 2: [neutral]}, vectors)
    channels = users[1].reshape(len(CHANNELS), 384)
    assert channels[0, 0] == 1 and channels[1, 1] == 1
    assert np.count_nonzero(channels[2:]) == 0
    assert 20 not in items
    assert np.count_nonzero(mean_opinion_profiles(users)[1]) == 2
    with pytest.raises(ValueError, match="Future"):
        build_opinion_profiles(rows, date(2025, 1, 1), {}, {}, {})


def fixture():
    rows = [SimpleNamespace(user_id=u, restaurant_id=i) for u, i in ((1,10),(1,20),(2,20),(2,30),(3,30),(3,10))]
    cfg = LightGCNConfig(dimension=4, layers=1, epochs=2, batch_size=3, seed=7)
    rng = np.random.default_rng(3)
    profiles = {key: rng.normal(size=(len(CHANNELS),384)).astype(np.float32).reshape(-1) for key in (1,2,3,10,20,30)}
    return rows, cfg, {k:profiles[k] for k in (1,2,3)}, {k:profiles[k] for k in (10,20,30)}


def test_zero_weight_exactly_preserves_graph_rng_and_updates():
    rows, cfg, users, items = fixture()
    plain = LightGCN(cfg).fit(rows)
    aspect = AspectGraphLightGCN(cfg, RLMRecConfig(weight=0), users, items).fit(rows)
    np.testing.assert_array_equal(plain.final_embeddings, aspect.final_embeddings)
    assert aspect.projector is None


def test_alignment_heads_update_graph_and_reload_reproduces(tmp_path):
    pytest.importorskip('torch')
    rows, cfg, users, items = fixture()
    aspect = AspectGraphLightGCN(cfg, RLMRecConfig(weight=.01, threads=1), users, items).fit(rows)
    plain = LightGCN(cfg).fit(rows)
    assert not np.array_equal(plain.final_embeddings, aspect.final_embeddings)
    assert aspect.channel_steps.sum() > 0
    assert len(aspect.projector['graph_heads']) == len(CHANNELS)
    path = tmp_path/'aspect.npz'
    aspect.save(path)
    loaded = AspectGraphLightGCN.load(path)
    np.testing.assert_array_equal(loaded.final_embeddings, aspect.final_embeddings)
    assert loaded.recommend([1,2,3], {}, 3) == aspect.recommend([1,2,3], {}, 3)
    with pytest.raises(ValueError, match='inference-only'):
        loaded.fit(rows)


def test_missing_channels_and_singletons_do_not_create_alignment():
    rows, cfg, users, items = fixture()
    for values in (users, items):
        for key in values:
            values[key][:] = 0
    aspect = AspectGraphLightGCN(cfg, RLMRecConfig(weight=.01), users, items).fit(rows)
    plain = LightGCN(cfg).fit(rows)
    np.testing.assert_array_equal(plain.final_embeddings, aspect.final_embeddings)
    assert aspect.channel_steps.sum() == 0
