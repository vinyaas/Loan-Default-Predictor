from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Missing Values Analysis
# -----------------------------------------------
# This class defines a template for missing values analysis.
# Subclasses must implement the methods to identify and visualize missing values.
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        print('Starting Analysis...')
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
        self.find_duplicates(df)
        print('Analysis complete...')

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.
        """
        pass
    
    @abstractmethod
    def find_duplicates(self ,df:pd.DataFrame):
        """
        Finds duplicated values in dataframe
        """
        pass
        
# Concrete Class for Missing Values Identification
# -------------------------------------------------
# This class implements methods to identify and visualize missing values in the dataframe.
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each column in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints the missing values count to the console.
        """
        print("\nMissing Values Count by Column:")
        missing_values = df.isnull().sum().sort_values(ascending=False)
        percentage = (missing_values / len(df)) * 100
        print(pd.concat((missing_values, percentage) , axis = 1 , keys = ['Total' , 'Percentage']))

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays a heatmap of missing values.
        """
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull() , cbar = False , cmap = 'viridis')
        plt.title("Missing Values Heatmap")
        plt.show()

    def find_duplicates(self, df: pd.DataFrame):
        """
        Parameters:
        df(pd.DataFrame) : The dataframe to be visualized
        
        Returns:
        None: This method should return duplicated values if any .
        """
        print("\nFinding Duplicates...")
        duplicated_count = df.duplicated().sum()
        print(f'The number of duplicated values: {duplicated_count}')
        

if __name__ == "__main__":
    # Example usage of the SimpleMissingValuesAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Missing Values Analysis
    # missing_values_analyzer = SimpleMissingValuesAnalysis()
    # missing_values_analyzer.analyze(df)
    pass
