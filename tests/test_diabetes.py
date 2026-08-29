import numpy as np

from linear_regre.diabetes import (
    PredictionQuality,
    TrainedRegression,
    format_result_summary,
    measure_prediction_quality,
    prepare_diabetes_dataset,
    run_workflow,
    summarize_results,
    train_regression_model,
)


def test_dataset_is_split_into_learning_and_evaluation_groups():
    prepared = prepare_diabetes_dataset()

    assert prepared.source_examples == 442
    assert prepared.learning_examples == 353
    assert prepared.evaluation_examples == 89
    assert prepared.learning_examples + prepared.evaluation_examples == prepared.source_examples
    assert not np.shares_memory(prepared.X_learning, prepared.X_evaluation)


def test_preparation_is_repeatable_without_customer_input():
    first = prepare_diabetes_dataset()
    second = prepare_diabetes_dataset()

    for first_values, second_values in zip(
        (first.X_learning, first.X_evaluation, first.y_learning, first.y_evaluation),
        (second.X_learning, second.X_evaluation, second.y_learning, second.y_evaluation),
    ):
        np.testing.assert_array_equal(first_values, second_values)


def test_training_predicts_every_held_out_example():
    prepared = prepare_diabetes_dataset()

    trained = train_regression_model(prepared)

    assert trained.evaluation_examples == prepared.evaluation_examples
    assert trained.predictions.shape == prepared.y_evaluation.shape
    assert np.isfinite(trained.predictions).all()
    assert trained.model.coef_.shape == (prepared.X_learning.shape[1],)


def test_training_is_repeatable_without_customer_input():
    prepared = prepare_diabetes_dataset()

    first = train_regression_model(prepared)
    second = train_regression_model(prepared)

    np.testing.assert_array_equal(first.predictions, second.predictions)


def test_prediction_quality_uses_held_out_predictions_and_actual_outcomes():
    trained = TrainedRegression(
        model=None,
        predictions=np.array([2.0, 4.0, 8.0]),
        evaluation_targets=np.array([1.0, 5.0, 7.0]),
    )

    quality = measure_prediction_quality(trained)

    assert isinstance(quality, PredictionQuality)
    assert quality.mae == 1.0
    assert quality.mse == 1.0
    assert quality.rmse == 1.0
    assert quality.r2 == 0.8392857142857143


def test_prediction_quality_rejects_invalid_evaluation_data():
    with np.testing.assert_raises(ValueError):
        measure_prediction_quality(
            TrainedRegression(model=None, predictions=np.array([]), evaluation_targets=np.array([]))
        )
    with np.testing.assert_raises(ValueError):
        measure_prediction_quality(
            TrainedRegression(model=None, predictions=np.array([1.0]), evaluation_targets=np.array([1.0, 2.0]))
        )


def test_result_summary_presents_samples_measures_interpretation_and_disclaimer():
    prepared = prepare_diabetes_dataset()
    trained = train_regression_model(prepared)
    summary = summarize_results(prepared, trained, measure_prediction_quality(trained))

    output = format_result_summary(summary)

    assert len(summary.samples) >= 5
    assert output.count("   ") >= 5
    assert "Loaded Diabetes dataset successfully" in output
    assert "MAE" in output and "MSE" in output and "RMSE" in output and "R²" in output
    assert "In plain language" in output
    assert "teaching and demonstration" in output
    assert "not for automated decision-making" in output


def test_result_summary_rejects_mismatched_evaluation_data():
    prepared = prepare_diabetes_dataset()
    trained = TrainedRegression(
        model=None,
        predictions=np.array([1.0]),
        evaluation_targets=np.array([1.0]),
    )

    with np.testing.assert_raises(ValueError):
        summarize_results(prepared, trained, PredictionQuality(0, 0, 0, 0))


def test_complete_workflow_returns_reviewable_summary_repeatedly():
    first = run_workflow()
    second = run_workflow()

    assert len(first.samples) >= 5
    assert len(second.samples) >= 5
    assert format_result_summary(first) == format_result_summary(second)
