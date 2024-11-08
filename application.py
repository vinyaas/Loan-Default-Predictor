import numpy as np
import pandas as pd
from rich.logging import RichHandler
import pickle, logging
from flask import Flask, render_template, request
from src.feature_engineering import FeatureEngineer, LogTransformation, LabelEncoding, StandardScaling

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("rich")

# Load the data
df = pd.read_csv('extracted_data/credit_risk_dataset.csv')

# Load the transformation objects
with open('log_transformer.pkl', 'rb') as f:
    log_transformer = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    standard_scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Get unique values for each column
    person_age = list(df.person_age.unique())
    person_income = list(df.person_income.unique())
    person_home_ownership = list(df.person_home_ownership.unique())
    person_emp_length = list(df.person_emp_length.unique())
    loan_intent = list(df.loan_intent.unique())
    loan_grade = list(df.loan_grade.unique())
    loan_amnt = list(df.loan_amnt.unique())
    loan_int_rate = list(df.loan_int_rate.unique())
    loan_percent_income = list(df.loan_percent_income.unique())
    cb_person_default_on_file = list(df.cb_person_default_on_file.unique())
    cb_person_cred_hist_length = list(df.cb_person_cred_hist_length.unique())

    # Sort the lists
    person_age.sort()
    person_income.sort()
    person_home_ownership.sort()
    person_emp_length.sort()
    loan_intent.sort()
    loan_grade.sort()
    loan_amnt.sort()
    loan_int_rate.sort()
    loan_percent_income.sort()
    cb_person_default_on_file.sort()
    cb_person_cred_hist_length.sort()

    return render_template(
        'index.html',
        person_age_list=person_age,
        person_income_list=person_income,
        person_home_ownership_list=person_home_ownership,
        person_emp_length_list=person_emp_length,
        loan_intent_list=loan_intent,
        loan_grade_list=loan_grade,
        loan_amnt_list=loan_amnt,
        loan_int_rate_list=loan_int_rate,
        loan_percent_income_list=loan_percent_income,
        cb_person_default_on_file_list=cb_person_default_on_file,
        cb_person_cred_hist_length_list=cb_person_cred_hist_length
    )

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        person_age = int(request.form['person_age'])
        person_income = int(request.form['person_income'])
        person_emp_length = float(request.form['person_emp_length'])
        loan_amnt = float(request.form['loan_amnt'])
        loan_int_rate = float(request.form['loan_int_rate'])
        loan_percent_income = float(request.form['loan_percent_income'])
        cb_person_cred_hist_length = int(request.form['cb_person_cred_hist_length'])
        person_home_ownership = request.form['person_home_ownership']
        loan_intent = request.form['loan_intent']
        loan_grade = request.form['loan_grade']
        cb_person_default_on_file = request.form['cb_person_default_on_file']
    

        # Create a DataFrame from the form inputs
        data = {
            'person_age': [person_age],
            'person_income': [person_income],
            'person_emp_length': [person_emp_length],
            'loan_amnt': [loan_amnt],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length],
            'person_home_ownership': [person_home_ownership],
            'loan_intent': [loan_intent],
            'loan_grade': [loan_grade],
            'cb_person_default_on_file': [cb_person_default_on_file]
        }
        df = pd.DataFrame(data)
        
        # Apply transformations without changing the column structure
        try:
            df[['person_income', 'person_age', 'person_emp_length']] = log_transformer.apply_feature_engineering(df[['person_income', 'person_age', 'person_emp_length']])
            df[['loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']] = standard_scaler.apply_feature_engineering(df[['loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']])
            df[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']] = label_encoder.apply_feature_engineering(df[['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']])
        except KeyError as e:
            logging.error(f"KeyError in transformation: {e}")
            return render_template('index.html', prediction_value="Error in data transformation")

        # Make sure all the required features are in the DataFrame for prediction
        try:
            prediction = model.predict(df)
            prediction_probabilities = model.predict_proba(df)[0] 
            if prediction_probabilities[1] > prediction_probabilities[0]: 
                prediction_label = 'Yes' 
                prediction_probability = prediction_probabilities[1] * 100
            else: 
                prediction_label = 'No' 
                prediction_probability = prediction_probabilities[0] * 100
                
        except ValueError as ve:
            logging.error(f"ValueError during prediction: {ve}")
            
            return render_template('index.html', prediction_value="Error during prediction")

        results = f"{prediction_label} with a probability of  {prediction_probability:.2f}%"

        return render_template('prediction.html', results=results)
    else:
        return render_template('index.html', prediction_value="Invalid response")

if __name__ == "__main__":
    app.run(debug=True)
