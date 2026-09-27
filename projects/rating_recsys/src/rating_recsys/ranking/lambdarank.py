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
            metric="ndcg",
            label_gain=[0, 1, 3],
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
            # LightGBM 4.7 deprecates eval_set in favour of eval_X/eval_y.
            if "eval_X" in inspect.signature(self._model.fit).parameters:
                kwargs = {"eval_X": (eval_features,), "eval_y": (eval_labels,)}
            else:
                kwargs = {"eval_set": [(eval_features, eval_labels)]}
            kwargs.update({"eval_group": [eval_groups], "eval_at": (self._ranking_k,)})
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
