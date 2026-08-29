"""Dataset workflows for linear regression."""

__all__ = [
    "DiabetesDataset",
    "PredictionQuality",
    "PredictionExample",
    "ResultSummary",
    "TrainedRegression",
    "format_result_summary",
    "measure_prediction_quality",
    "prepare_diabetes_dataset",
    "run_workflow",
    "summarize_results",
    "train_regression_model",
]


def __getattr__(name: str):
    """Keep the public convenience API without eagerly loading the CLI module."""

    if name in __all__:
        from .diabetes import (
            DiabetesDataset,
            PredictionQuality,
            PredictionExample,
            ResultSummary,
            TrainedRegression,
            format_result_summary,
            measure_prediction_quality,
            prepare_diabetes_dataset,
            run_workflow,
            summarize_results,
            train_regression_model,
        )

        return {
            "DiabetesDataset": DiabetesDataset,
            "PredictionQuality": PredictionQuality,
            "PredictionExample": PredictionExample,
            "ResultSummary": ResultSummary,
            "TrainedRegression": TrainedRegression,
            "format_result_summary": format_result_summary,
            "measure_prediction_quality": measure_prediction_quality,
            "prepare_diabetes_dataset": prepare_diabetes_dataset,
            "run_workflow": run_workflow,
            "summarize_results": summarize_results,
            "train_regression_model": train_regression_model,
        }[name]
    raise AttributeError(name)
