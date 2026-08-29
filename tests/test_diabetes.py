import numpy as np

from linear_regre.diabetes import prepare_diabetes_dataset


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
