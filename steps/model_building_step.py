import logging
from typing import Annotated
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

model = Model(
    name="Loan Default Predictor",
    version=None,
    license="Apache 2.0",
    description="Model for predicting loan defaults.",
)

@step(enable_cache=False, model=model)
def model_building_step(
    x_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a Random Forest model using scikit-learn wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Random Forest model.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(x_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    logging.info("Initializing Random Forest model.")

    # Create a pipeline with Random Forest model
    pipeline = Pipeline(
        [
            ('model', RandomForestClassifier(random_state=42))
        ]
    )

    try:
        logging.info("Building and training the Random Forest model.")
        pipeline.fit(x_train, y_train)
        logging.info("Model training completed.")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    return pipeline
