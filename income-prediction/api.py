import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = Flask(__name__)

# Check if the model file exists
model_filename = "model.pkl"
dt_model = None
le = LabelEncoder()

# Load the model if it exists
if os.path.exists(model_filename):
    with open(model_filename, "rb") as model_file:
        dt_model = pickle.load(model_file)

# Function to train the model using `data.csv`
# Function to train the model using the CSV file
def train_model():
    global dt_model, le

    # Check if the CSV file exists before loading
    if not os.path.exists(csv_file_path): # type: ignore
        raise FileNotFoundError(f"Data file not found: {csv_file_path}") # type: ignore

    # Load dataset
    data = pd.read_csv(csv_file_path) # type: ignore


    # Encode the target variable and other categorical columns
    data['Income'] = le.fit_transform(data['Income'])
    columns_to_encode = ['Working Class', 'Marital Status', 'Occupation', 'Relationship', 'Gender']
    for column in columns_to_encode:
        data[column] = le.fit_transform(data[column])

    # Select features and target
    X = data[['Age', 'Working Class', 'Marital Status', 'Occupation', 'Relationship', 'Gender',
              'Capital Gain', 'Capital Loss', 'Hours per Week']]
    y = data['Income']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Decision Tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    # Save the model to a file
    with open(model_filename, "wb") as model_file:
        pickle.dump(dt_model, model_file)

# Train the model if it's not loaded
if dt_model is None:
    train_model()

# Home route to render the input form
@app.route('/')
def index():  # Changed function name from 'home' to 'index' to avoid conflict
    return render_template('index.html')

# Function for making predictions from the form
def ValuePredictor(to_predict_list):
    try:
        # Ensure the input is the correct shape
        to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
        if dt_model is None:
            return None
        result = dt_model.predict(to_predict)
        return result[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Route to handle form submissions for predictions
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        prediction = "Prediction could not be made."

        try:
            # Convert form input into the required format
            to_predict_list = request.form.to_dict()
            to_predict_list['age'] = float(to_predict_list['age'])
            to_predict_list['capital_gain'] = int(to_predict_list['capital_gain'])
            to_predict_list['capital_loss'] = int(to_predict_list['capital_loss'])
            to_predict_list['hours_per_week'] = int(to_predict_list['hours_per_week'])
            to_predict_list = list(map(int, to_predict_list.values()))

            # Predict using the form data
            result = ValuePredictor(to_predict_list)

            if result is None:
                prediction = "An error occurred during prediction."
            else:
                prediction = 'Income more than 50K' if int(result) == 1 else 'Income less than 50K'

            # Display results with the current data
            data_df = pd.read_csv('D:\Predict Income\income-prediction\dataJ.csv')
            data_html = data_df.to_html(classes='data', header="true", index=False)

            return render_template("result.html", prediction=prediction, data_html=data_html, user_input=request.form)

        except ValueError as e:
            return render_template("result.html", prediction="Invalid input, please try again.")

    return render_template("result.html", prediction=prediction)

# API route for predicting using JSON input
@app.route('/predict', methods=['POST'])
def api_predict():
    try:
        # Get data in JSON format
        data = request.get_json()

        # Create a DataFrame for prediction
        input_data = pd.DataFrame([{
            'Age': data['Age'],
            'Working Class': data['Working Class'],
            'Marital Status': data['Marital Status'],
            'Occupation': data['Occupation'],
            'Relationship': data['Relationship'],
            'Gender': data['Gender'],
            'Capital Gain': data['Capital Gain'],
            'Capital Loss': data['Capital Loss'],
            'Hours per Week': data['Hours per Week']
        }])

        # Encode categorical columns
        columns_to_encode = ['Working Class', 'Marital Status', 'Occupation', 'Relationship', 'Gender']
        for column in columns_to_encode:
            input_data[column] = le.transform(input_data[column])

        # Predict using the Decision Tree model
        dt_prediction = dt_model.predict(input_data)
        income_type = le.inverse_transform(dt_prediction)[0]

        return jsonify({'Decision Tree Prediction': income_type})

    except ValueError as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
