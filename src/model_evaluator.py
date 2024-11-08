import logging
from abc import ABC, abstractmethod
from joblib import load
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (ClassifierMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass


# Concrete Strategy for Classification Model Evaluation
class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates a classification model using accuracy, precision, recall, and F1 score.

        Parameters:
        model (ClassifierMixin): The trained classification model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
        """
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

        logging.info(f"Model Evaluation Metrics: \n {metrics}")
        return metrics


# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (ClassifierMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, x_test, y_test)


# Example usage
if __name__ == "__main__":
    file_path = r'C:\\Users\\10714194\\OneDrive - LTIMindtree\\Desktop\\py\\Machine Learning Tutorial\\Loan Defaultor Predictor\\model\\model.pkl'  
    model = load(file_path)
    logging.info('reading the test datasets')
    x_test =  pd.read_csv(r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\x_test.csv')
    y_test =  pd.read_csv(r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\y_test.csv').squeeze()
    
    # Initialize model evaluator with a specific strategy
    model_evaluator = ModelEvaluator(ClassificationModelEvaluationStrategy())
    evaluation_metrics = model_evaluator.evaluate(model, x_test, y_test)
    logging.info('Model Evaluation Completed')
