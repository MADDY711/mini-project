from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Reading data from CSV file
df = pd.read_csv('C:\\Python program\\pandas\\Total.csv')

# Splitting the data into features (X) and target variable (y)
X = df.drop('sr', axis=1)
y = df['total']

# Creating and training the Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Prepare the new features for prediction
    new_data = pd.DataFrame(data, index=[0])

    # Use the trained model to make predictions
    predicted_total = lr_model.predict(new_data)

    # Return the prediction as JSON
    return jsonify({'predicted_total': predicted_total[0]})

if __name__ == '__main__':
    app.run(debug=True)
