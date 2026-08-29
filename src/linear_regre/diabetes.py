"""Load and prepare scikit-learn's built-in Diabetes dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_diabetes
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
    """Run the input-free preparation workflow and report its outcome."""

    prepared = prepare_diabetes_dataset()
    print("Loaded Diabetes dataset successfully")
    print(
        f"Prepared {prepared.source_examples} examples: "
        f"{prepared.learning_examples} learning, "
        f"{prepared.evaluation_examples} held-out evaluation"
    )


if __name__ == "__main__":
    main()
