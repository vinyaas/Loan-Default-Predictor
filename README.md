<div align="center">

# Loan Default Predictor

## Predict if a customer will default on the loan .
___
https://github.com/user-attachments/assets/bcab6198-4132-4c1c-9524-086a06f4beca
___
</div>

## Approach:

- Conducted comprehensive exploratory data analysis in Jupyter Notebook.
- Utilized design patterns for univariate, bivariate, and multivariate analysis.
- Captured analysis visuals and stored them in the `visualizations` folder for presentation purposes.
- Employed a modular programming approach, incorporating Factory, Strategy, and Template design patterns, and implemented logging mechanisms where necessary.
- Performed feature engineering, including outlier detection and imputation.
- Applied a Random Forest model for predicting loan defaults.
- Developed step files for pipeline implementation using ZENML.
- Created a Flask API for end-to-end implementation of the model, enabling user data input.
- Designed the frontend using HTML and CSS for a seamless user experience.
___
<div align="center">
  
![Screenshot 2024-11-08 134546](https://github.com/user-attachments/assets/084fcc23-d92e-4899-9513-63cfecf8fed6)
___
</div>

## How to Run

To run the app, follow these steps:

1. **Download the Repository**:
   - Clone this repository to your local machine using:
     ```sh
     git clone https://github.com/your-username/your-repo-name.git
     ```
   - Navigate into the directory:
     ```sh
     cd your-repo-name
     ```

2. **Install the Required Libraries**:
   - Create and activate a virtual environment (optional but recommended):
     ```sh
     python -m venv env
     source env/bin/activate  # On Windows, use `env\Scripts\activate`
     ```
   - Install the required libraries:
     ```sh
     pip install -r requirements.txt
     ```

3. **Run the Application**:
   - Execute the `app.py` file:
     ```sh
     python app.py
     ```

4. **Open the Application in Your Browser**:
   - Once the server is running, open your web browser and go to:
     ```sh
     http://127.0.0.1:5000
     ```
___

## Directory Structure

<pre>
.vscode
|   settings.json
|
analysis
|   EDA.ipynb
|
|---- analyze_src
|    |-- basic_data_inspection.py
|    |-- bivariate_analysis.py
|    |-- missing_values.py
|    |-- multivariate_analysis.py
|    |-- univariate_analysis.py
|
data
|   credit_risk_dataset.zip
|
explanations
|   factory_design_pattern.py
|   strategy_pattern.py
|   template_design_pattern.py
|
extracted_data
|
models
|   label_encoder.pkl
|   log_transformer.pkl
|   model.pkl
|   standard_scaler.pkl
|
src
|   data_ingestion.py
|   data_splitter.py
|   feature_engineering.py
|   handle_missing_values.py
|   model_building.py
|   model_evaluator.py
|   outlier_detection.py
|   utils.py
|   __init__.py
|
steps
|   Custom_data.py
|   data_ingestion_step.py
|   data_splitter_step.py
|   feature_engineering_step.py
|   handle_missing_values_step.py
|   model_building_step.py
|   model_evaluator_step.py
|   outlier_detection_step.py
|
styles
|   layout.css
|
templates
|   index.html
|   layout.html
|   Prediction.html
|
tests
|   pickle_accuracy.py
|
visualizations
|
zenml_pipeline
|   predict_pipeline.py
|
.gitignore
README.md
application.py
requirements.txt
</pre>
___
## Technologies Used

- **Python Libraries**:
  - numpy
  - pandas
  - seaborn
  - matplotlib
  - flask
  - sklearn
  - pickle
  - random forest
- **Version Control**:
  - git
- **Backend**:
  - flask
- **Concept**:
  - Machine Learning
___

