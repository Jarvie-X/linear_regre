"""Dataset workflows for linear regression."""

__all__ = ["DiabetesDataset", "prepare_diabetes_dataset"]


def __getattr__(name: str):
    """Keep the public convenience API without eagerly loading the CLI module."""

    if name in __all__:
        from .diabetes import DiabetesDataset, prepare_diabetes_dataset

        return {"DiabetesDataset": DiabetesDataset, "prepare_diabetes_dataset": prepare_diabetes_dataset}[name]
    raise AttributeError(name)
