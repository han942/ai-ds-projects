"""Frozen review embeddings as cutoff-safe features of existing candidates."""

from collections import defaultdict

import numpy as np

from rating_recsys.retrieval.review_embeddings import profile_api_inputs, _unit
from rating_recsys.retrieval.review_profiles import _selected_reviews


REVIEW_FEATURE_NAMES = (
    "review_cosine", "review_user_present", "review_item_present",
    "review_pair_present", "review_user_count", "review_item_count",
)


def review_feature_matrix(queries, candidates, users, items, user_counts, item_counts):
    """Align features to row IDs, including legitimate zero/negative cosine.

    Missing vectors use zero plus explicit masks. Counts refer to selected
    positive nonempty review events, before concatenation/token truncation.
    """
    users = {uid: np.asarray(_unit(v), dtype=np.float32) for uid, v in users.items()}
    items = {iid: np.asarray(_unit(v), dtype=np.float32) for iid, v in items.items()}
    widths = {len(v) for v in (*users.values(), *items.values())}
    if len(widths) > 1:
        raise ValueError("Review user and item vector dimensions differ")
    result = np.zeros((len(candidates.row_restaurant_ids), len(REVIEW_FEATURE_NAMES)), dtype=np.float32)
    offset = 0
    for query, size in zip(queries, candidates.group_sizes, strict=True):
        user = users.get(query.user_id)
        for row in range(offset, offset + size):
            iid = int(candidates.row_restaurant_ids[row])
            item = items.get(iid)
            present = user is not None and item is not None
            result[row] = (
                float(np.clip(user @ item, -1, 1)) if present else 0.0,
                float(user is not None), float(item is not None), float(present),
                user_counts.get(query.user_id, 0), item_counts.get(iid, 0),
            )
        offset += size
    if offset != len(result):
        raise ValueError("Review feature groups do not match candidate rows")
    return result


class ReviewFeatureBuilder:
    """Build profiles exclusively from the supplied window's past history."""

    def __init__(self, texts, config, formatter, cache, *, cache_only=True):
        if config.aggregation != "concat" or config.user_profile != "own_reviews":
            raise ValueError("The initial LTR comparison holds concat/own_reviews fixed")
        self.texts, self.config, self.formatter, self.cache = texts, config, formatter, cache
        self.cache_only = cache_only
        self.audit = []

    def __call__(self, history, queries, candidates):
        cutoff = queries[0].cutoff
        if any(q.cutoff != cutoff for q in queries) or any(r.event_date > cutoff for r in history):
            raise ValueError("Review feature history exceeds its query cutoff")
        users, items, _, docs = profile_api_inputs(
            history, self.texts, [q.user_id for q in queries], self.config, self.formatter,
        )
        misses = self.cache.missing_count(docs)
        if misses and self.cache_only:
            raise RuntimeError(f"Missing {misses} frozen review embeddings at {cutoff}; run --embed-only first")
        vectors = self.cache.embed(docs)
        by_user, by_item = defaultdict(list), defaultdict(list)
        for r in history:
            by_user[r.user_id].append(r)
            by_item[r.restaurant_id].append(r)
        counts = lambda grouped, limit: {
            key: len(_selected_reviews(rows, self.texts, limit, self.config))
            for key, rows in grouped.items()
        }
        extra = review_feature_matrix(
            queries, candidates,
            {uid: vectors[doc] for uid, doc in users.items()},
            {iid: vectors[doc] for iid, doc in items.items()},
            counts(by_user, self.config.max_user_reviews), counts(by_item, self.config.max_item_reviews),
        )
        self.audit.append({
            "phase": queries[0].phase, "cutoff": cutoff.isoformat(),
            "latest_history_date": max((r.event_date for r in history), default=None).isoformat() if history else None,
            "queries": len(queries), "rows": len(extra), "profile_inputs": len(set(docs)),
            "pair_coverage": float(extra[:, 3].mean()) if len(extra) else 0.0,
        })
        return np.concatenate([candidates.features, extra], axis=1)
