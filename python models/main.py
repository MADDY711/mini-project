# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Reading data from CSV file
df = pd.read_csv('E:\clg\clg projects\mini project\execution\dataset\dummy\Testset.csv')

# Splitting the data into features (X) and target variable (y)
X = df.drop('sr', axis=1)
y = df['Total']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Creating and training the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Making predictions
predictions = lr_model.predict(X_test)

# Calculating Mean Squared Error to evaluate the model performance
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)
print(predictions)

#added changes
'''# Prepare the new features for prediction
new_data = {
    'Feature1': [10],  # Replace with the actual value
    'Feature2': [11],  # Replace with the actual value
    # Add other features similarly
}

# Create a DataFrame from the new data
new_df = pd.DataFrame(new_data)

# Use the trained model to make predictions
predicted_total = lr_model.predict(new_df)

print('Predicted Total:', predicted_total[0])
'''
