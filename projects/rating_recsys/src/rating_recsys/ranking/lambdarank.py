"""LightGBM LambdaRank adapter with deterministic settings."""

from __future__ import annotations

import inspect
from pathlib import Path

from rating_recsys.ranking.features import FEATURE_NAMES


class LightGBMLambdaRanker:
    """Stage 2 ranker trained on numeric matrices grouped by query."""

    def __init__(
        self,
        *,
        random_seed: int = 42,
        ranking_k: int = 10,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 10,
        reg_lambda: float = 1.0,
        n_jobs: int = 1,
        label_gain: tuple[float, ...] = (0.0, 1.0, 3.0),
    ) -> None:
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "LightGBM is required for R1; install the experiment extra with "
                "`pip install -e '.[experiment]'` and ensure the OpenMP runtime "
                "(libgomp on Linux) is available"
            ) from exc

        self._lgb = lgb
        self._ranking_k = ranking_k
        self._feature_names = feature_names
        self._fitted = False
        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            # Validation gain must not change with the training label_gain.
            metric="None",
            label_gain=list(label_gain),
            lambdarank_truncation_level=ranking_k + 3,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
            random_state=random_seed,
            bagging_seed=random_seed,
            feature_fraction_seed=random_seed,
            data_random_seed=random_seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=n_jobs,
            verbosity=-1,
        )

    def fit(
        self,
        features,
        labels,
        groups,
        *,
        eval_set: tuple[object, object, object] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        """Fit one row per candidate; ``groups`` holds the row count per query.

        ``eval_set`` is ``(features, labels, groups)``. Early stopping monitors
        NDCG at ``ranking_k`` only, so the chosen iteration optimizes the
        reported cutoff.
        """

        if not len(groups):
            raise ValueError("LambdaRank requires at least one query group")
        kwargs: dict[str, object] = {}
        if eval_set is not None:
            eval_features, eval_labels, eval_groups = eval_set
            import numpy as np

            width = max(eval_groups)
            gains = np.zeros((len(eval_groups), width), dtype=np.float64)
            valid = np.zeros_like(gains, dtype=bool)
            offset = 0
            for index, size in enumerate(eval_groups):
                group_labels = np.asarray(eval_labels[offset:offset + size], dtype=int)
                if np.any((group_labels < 0) | (group_labels > 2)):
                    raise ValueError("Validation labels must use the common relevance 0/1/2")
                gains[index, :size] = np.asarray([0.0, 1.0, 3.0])[group_labels]
                valid[index, :size] = True
                offset += size
            k = min(self._ranking_k, width)
            discount = 1 / np.log2(np.arange(k) + 2)
            ideal = (np.sort(gains, axis=1)[:, ::-1][:, :k] * discount).sum(axis=1)

            def common_ndcg(y_true, predictions):
                scores = np.full_like(gains, -np.inf)
                scores[valid] = predictions
                order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
                dcg = (np.take_along_axis(gains, order, axis=1) * discount).sum(axis=1)
                values = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
                return f"ndcg@{self._ranking_k}", float(values.mean()), True
            # LightGBM 4.7 deprecates eval_set in favour of eval_X/eval_y.
            if "eval_X" in inspect.signature(self._model.fit).parameters:
                kwargs = {"eval_X": (eval_features,), "eval_y": (eval_labels,)}
            else:
                kwargs = {"eval_set": [(eval_features, eval_labels)]}
            kwargs.update({"eval_group": [eval_groups], "eval_at": (self._ranking_k,)})
            kwargs["eval_metric"] = common_ndcg
            if early_stopping_rounds:
                kwargs["callbacks"] = [
                    self._lgb.early_stopping(
                        early_stopping_rounds, first_metric_only=True, verbose=False
                    )
                ]
        self._model.fit(features, labels, group=groups, **kwargs)
        self._fitted = True

    def predict(self, features):
        if not self._fitted:
            raise RuntimeError("Ranker must be fitted before predict()")
        return self._model.predict(features)

    @property
    def best_iteration(self) -> int | None:
        value = getattr(self._model, "best_iteration_", None)
        return int(value) if value else None

    @property
    def validation_curve(self) -> list[float]:
        results = getattr(self._model, "evals_result_", None) or {}
        for scores in results.values():
            for values in scores.values():
                return [float(value) for value in values]
        return []

    def feature_importance(self) -> dict[str, float]:
        if not self._fitted:
            return {}
        return {
            name: float(value)
            for name, value in zip(
                self._feature_names, self._model.feature_importances_, strict=True
            )
        }

    def save(self, destination: Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted ranker")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(destination))
