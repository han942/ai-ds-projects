"""LightGBM LambdaRank adapter with deterministic baseline settings."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from rating_recsys.experiments.models import FeatureRow, RankedCandidate
from rating_recsys.ranking.features import FEATURE_NAMES


class LightGBMLambdaRanker:
    def __init__(
        self,
        *,
        random_seed: int = 42,
        ranking_k: int = 10,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "LightGBM is required for R1; install the experiment extra with "
                "`pip install -e '.[experiment]'` and ensure the OpenMP runtime "
                "(libgomp on Linux) is available"
            ) from exc

        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[0, 1, 3],
            lambdarank_truncation_level=ranking_k + 3,
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            reg_lambda=1.0,
            random_state=random_seed,
            bagging_seed=random_seed,
            feature_fraction_seed=random_seed,
            data_random_seed=random_seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )
        self._feature_names = feature_names
        self._fitted = False

    def fit(self, rows: Iterable[FeatureRow]) -> None:
        grouped = _group_rows(rows)
        usable = [
            query_rows
            for query_rows in grouped
            if any(row.relevance > 0 for row in query_rows)
            and any(row.relevance == 0 for row in query_rows)
        ]
        if not usable:
            raise ValueError("LambdaRank requires at least one usable query group")
        flattened = [row for query_rows in usable for row in query_rows]
        features = [_feature_vector(row, self._feature_names) for row in flattened]
        labels = [row.relevance for row in flattened]
        groups = [len(query_rows) for query_rows in usable]
        self._model.fit(
            features,
            labels,
            group=groups,
        )
        self._fitted = True

    def rank(self, rows: Iterable[FeatureRow]) -> tuple[RankedCandidate, ...]:
        if not self._fitted:
            raise RuntimeError("Ranker must be fitted before rank()")
        grouped = _group_rows(rows)
        ranked: list[RankedCandidate] = []
        for query_rows in grouped:
            scores = self._model.predict(
                [_feature_vector(row, self._feature_names) for row in query_rows]
            )
            ordered = sorted(
                zip(query_rows, scores, strict=True),
                key=lambda pair: (-float(pair[1]), pair[0].restaurant_id),
            )
            ranked.extend(
                RankedCandidate(
                    row=row,
                    ranking_score=float(score),
                    final_rank=rank,
                )
                for rank, (row, score) in enumerate(ordered, start=1)
            )
        return tuple(ranked)

    def feature_importance(self) -> dict[str, float]:
        if not self._fitted:
            return {}
        return {
            name: float(value)
            for name, value in zip(
                self._feature_names,
                self._model.feature_importances_,
                strict=True,
            )
        }

    def save(self, destination: Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted ranker")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(destination))


def _group_rows(rows: Iterable[FeatureRow]) -> list[list[FeatureRow]]:
    grouped: defaultdict[str, list[FeatureRow]] = defaultdict(list)
    for row in rows:
        grouped[row.query_id].append(row)
    return [
        sorted(grouped[query_id], key=lambda row: row.restaurant_id)
        for query_id in sorted(grouped)
    ]


def _feature_vector(row: FeatureRow, feature_names: tuple[str, ...]) -> list[float]:
    return [float(row.features[name]) for name in feature_names]
