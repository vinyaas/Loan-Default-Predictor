import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model_evaluator import ModelEvaluator, ClassificationModelEvaluationStrategy
from zenml import step

class ClassificationModelEvaluationStrategy:
    def evaluate_model(self, model, x_test, y_test) -> dict:
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(x_test)

        logging.info("Calculating evaluation metrics.")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics

@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[dict, float]:
    if not isinstance(x_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Evaluating the model using the classification strategy.")
    evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())

    evaluation_metrics = evaluator.evaluate(
        trained_model, x_test, y_test
    )

    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    
    accuracy = evaluation_metrics.get("Accuracy", None)
    return evaluation_metrics, accuracy
