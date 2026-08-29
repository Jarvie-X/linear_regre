"""Dataset workflows for linear regression."""

__all__ = [
    "DiabetesDataset",
    "PredictionQuality",
    "TrainedRegression",
    "measure_prediction_quality",
    "prepare_diabetes_dataset",
    "train_regression_model",
]


def __getattr__(name: str):
    """Keep the public convenience API without eagerly loading the CLI module."""

    if name in __all__:
        from .diabetes import (
            DiabetesDataset,
            PredictionQuality,
            TrainedRegression,
            measure_prediction_quality,
            prepare_diabetes_dataset,
            train_regression_model,
        )

        return {
            "DiabetesDataset": DiabetesDataset,
            "PredictionQuality": PredictionQuality,
            "TrainedRegression": TrainedRegression,
            "measure_prediction_quality": measure_prediction_quality,
            "prepare_diabetes_dataset": prepare_diabetes_dataset,
            "train_regression_model": train_regression_model,
        }[name]
    raise AttributeError(name)
