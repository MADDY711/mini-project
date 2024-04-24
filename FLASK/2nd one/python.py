# Importing necessary libraries
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('E:/clg/clg projects/mini project/execution/dataset/dummy/Testset.csv')

# Split the data into features (X) and target variable (y)
X = df.drop('sr', axis=1)
y = df['Total']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Define the route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Prepare the input features for prediction
    input_features = [data['feature1'], data['feature2']]  # Adjust as per your dataset features

    # Make prediction
    prediction = lr_model.predict([input_features])

    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
