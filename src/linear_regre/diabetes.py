"""Load and prepare scikit-learn's built-in Diabetes dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DiabetesDataset:
    """Learning and held-out evaluation examples from the Diabetes dataset."""

    X_learning: Any
    X_evaluation: Any
    y_learning: Any
    y_evaluation: Any
    source_examples: int

    @property
    def learning_examples(self) -> int:
        return len(self.y_learning)

    @property
    def evaluation_examples(self) -> int:
        return len(self.y_evaluation)


@dataclass(frozen=True)
class TrainedRegression:
    """A fitted model and its predictions for the held-out examples."""

    model: LinearRegression
    predictions: np.ndarray
    evaluation_targets: Any

    @property
    def evaluation_examples(self) -> int:
        """Return the number of held-out examples that were predicted."""

        return len(self.predictions)


@dataclass(frozen=True)
class PredictionQuality:
    """Numeric quality measures calculated on held-out predictions."""

    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r_squared: float

    @property
    def mae(self) -> float:
        """Return mean absolute error (MAE)."""

        return self.mean_absolute_error

    @property
    def mse(self) -> float:
        """Return mean squared error (MSE)."""

        return self.mean_squared_error

    @property
    def rmse(self) -> float:
        """Return root mean squared error (RMSE)."""

        return self.root_mean_squared_error

    @property
    def r2(self) -> float:
        """Return the coefficient of determination (R²)."""

        return self.r_squared


def measure_prediction_quality(trained: TrainedRegression) -> PredictionQuality:
    """Measure predictions against actual outcomes in the held-out set.

    The calculation intentionally reads only the evaluation predictions and
    targets.  Learning examples and the fitted model are not used, preventing
    training performance from being reported as evaluation performance.
    """

    predictions = np.asarray(trained.predictions, dtype=float)
    actual = np.asarray(trained.evaluation_targets, dtype=float)
    if predictions.ndim != 1 or actual.ndim != 1:
        raise ValueError("Evaluation predictions and targets must be one-dimensional")
    if len(predictions) == 0:
        raise ValueError("Evaluation predictions and targets must not be empty")
    if len(predictions) != len(actual):
        raise ValueError("Evaluation predictions and targets do not match")
    if not np.isfinite(predictions).all() or not np.isfinite(actual).all():
        raise ValueError("Evaluation predictions and targets must be finite")

    errors = predictions - actual
    mean_squared_error = float(np.mean(errors**2))
    total_sum_of_squares = float(np.sum((actual - np.mean(actual)) ** 2))
    # R² is conventionally zero when the evaluation targets are constant and
    # the predictions are not exact, matching a useful baseline interpretation.
    r_squared = (
        1.0 - float(np.sum(errors**2)) / total_sum_of_squares
        if total_sum_of_squares > 0
        else (1.0 if np.allclose(predictions, actual) else 0.0)
    )
    return PredictionQuality(
        mean_absolute_error=float(np.mean(np.abs(errors))),
        mean_squared_error=mean_squared_error,
        root_mean_squared_error=float(np.sqrt(mean_squared_error)),
        r_squared=float(r_squared),
    )


def train_regression_model(dataset: DiabetesDataset) -> TrainedRegression:
    """Fit a linear regression model and predict every evaluation example."""

    X_learning = np.asarray(dataset.X_learning)
    y_learning = np.asarray(dataset.y_learning)
    X_evaluation = np.asarray(dataset.X_evaluation)
    y_evaluation = np.asarray(dataset.y_evaluation)

    if X_learning.ndim != 2 or X_evaluation.ndim != 2:
        raise ValueError("Learning and evaluation features must be two-dimensional")
    if y_learning.ndim != 1 or y_evaluation.ndim != 1:
        raise ValueError("Learning and evaluation targets must be one-dimensional")
    if len(X_learning) != len(y_learning):
        raise ValueError("Learning features and targets do not match")
    if len(X_evaluation) != len(y_evaluation):
        raise ValueError("Evaluation features and targets do not match")
    if len(X_learning) == 0 or len(X_evaluation) == 0:
        raise ValueError("Learning and evaluation examples must not be empty")

    model = LinearRegression()
    model.fit(X_learning, y_learning)
    predictions = np.asarray(model.predict(X_evaluation))
    if predictions.shape != y_evaluation.shape:
        raise ValueError("Model did not produce one prediction per evaluation example")

    return TrainedRegression(
        model=model,
        predictions=predictions,
        evaluation_targets=y_evaluation,
    )


def prepare_diabetes_dataset(
    *, evaluation_size: float = 0.2, random_state: int = 42
) -> DiabetesDataset:
    """Load the built-in dataset and split it into learning/evaluation sets.

    No customer-provided path or data is needed.  The fixed default seed makes
    the preparation reproducible across runs.
    """

    dataset = load_diabetes()
    features = np.asarray(dataset.data)
    targets = np.asarray(dataset.target)

    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("Diabetes dataset has unexpected dimensions")
    if len(features) != len(targets) or len(features) == 0:
        raise ValueError("Diabetes dataset features and targets do not match")
    if not 0 < evaluation_size < 1:
        raise ValueError("evaluation_size must be between 0 and 1")

    X_learning, X_evaluation, y_learning, y_evaluation = train_test_split(
        features,
        targets,
        test_size=evaluation_size,
        random_state=random_state,
    )
    result = DiabetesDataset(
        X_learning=X_learning,
        X_evaluation=X_evaluation,
        y_learning=y_learning,
        y_evaluation=y_evaluation,
        source_examples=len(features),
    )
    if result.learning_examples + result.evaluation_examples != result.source_examples:
        raise ValueError("learning and evaluation groups do not cover the source data")
    return result


def main() -> None:
    """Run the input-free preparation and training workflow."""

    prepared = prepare_diabetes_dataset()
    trained = train_regression_model(prepared)
    quality = measure_prediction_quality(trained)
    print("Loaded Diabetes dataset successfully")
    print(
        f"Prepared {prepared.source_examples} examples: "
        f"{prepared.learning_examples} learning, "
        f"{prepared.evaluation_examples} held-out evaluation"
    )
    print(f"Trained linear regression and produced {trained.evaluation_examples} predictions")
    print(
        "Held-out prediction quality: "
        f"MAE={quality.mae:.3f}, "
        f"MSE={quality.mse:.3f}, "
        f"RMSE={quality.rmse:.3f}, "
        f"R²={quality.r2:.3f}"
    )


if __name__ == "__main__":
    main()
