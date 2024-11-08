from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os 


# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
# Subclasses must implement the analyze method.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass


# Concrete Strategy for Numerical Features
# -----------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame):
        """
        Plots the distribution of a numerical features using a histogram , KDE and Subplots.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram with a KDE plot.
        """
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        fig , axes = plt.subplots(nrows = 4 , ncols = 2 , figsize = (15,20))
        plt.suptitle('Numerical Data Distribution')
        for i in range(len(numerical_columns)):
            plt.subplot(4 , 2 , i+1)
            sns.histplot(df[numerical_columns[i]] , color = 'r' , kde = True)
            title = 'Distribution :' + numerical_columns[i]
            plt.title(title)
            plt.tight_layout()
        plt.show()
            


# Concrete Strategy for Categorical Features
# -------------------------------------------
# This strategy analyzes categorical features by plotting their frequency distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        fig , axes = plt.subplots(nrows = 2, ncols = 2 , figsize = (15,10))
        plt.suptitle('Categorical Data Distribution')
        for i in range(len(categorical_columns)):
            plt.subplot(2, 2 , i+1)
            sns.countplot(x = df[categorical_columns[i]] , color = 'r')
            title = 'Distribution :' + categorical_columns[i]
            plt.title(title)
            plt.xticks(rotation = 90)
            plt.tight_layout()
        plt.show()


# Context Class that uses a UnivariateAnalysisStrategy
# ----------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df)


# Example usage
if __name__ == "__main__":
    # Example usage of the UnivariateAnalyzer with different strategies.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Analyzing a numerical feature
    # analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'Column1')

    # Analyzing a categorical feature
    # analyzer.set_strategy(CategoricalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'Column2')
    pass
