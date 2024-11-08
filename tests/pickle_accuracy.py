import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the model from the pickle file
with open(r'C:\\Users\\10714194\\OneDrive - LTIMindtree\\Desktop\\py\\Machine Learning Tutorial\\Loan Defaultor Predictor\\model\\model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load your test dataset
X_test = pd.read_csv(r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\x_test.csv')
y_test = pd.read_csv(r'C:\Users\10714194\OneDrive - LTIMindtree\Desktop\py\Machine Learning Tutorial\Loan Defaultor Predictor\extracted_data\y_test.csv').squeeze()

# Make predictions using the loaded model
predictions = model.predict(X_test)

# Measure accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy of the model: {accuracy}')
