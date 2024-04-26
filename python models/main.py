# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pickle

#take input from user
n = int(input("Enter month no. : "))

# Reading data from CSV file
df = pd.read_csv('E:\clg\clg projects\mini project\execution\dataset\dummy\Testset2.csv' )

# Splitting the data into features (X) and target variable (y)
X = df.drop('sr.', axis=1)
y = df['total']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n/100, random_state=n)

# Creating and training the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Making predictions
predictions = lr_model.predict(X_test)

# Calculating Mean Squared Error to evaluate the model performance
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)
print(predictions)

print("note: the last value in array is the value of the target month ")

# #plt.figure(figsize=(8, 6))
# #plt.subplot(1, 1, 1)
# plt.plot()
# #plt.plot(X, y, marker='o', color='b', linestyle='-', linewidth=1, markersize=2)
# plt.title("crowd comparision")
# plt.xlabel("month")
# plt.ylabel("no. of tourist")
# #plt.ylim(0, 300000)
# #plt.xlim(0, 100)
# plt.grid(True)
# #plt.xticks(
# #plt.yticks(y)
# #plt.figure(figsize=(15,5))
# plt.show()

import matplotlib.pyplot as plt

# Sample data
months = [n] # Example months from January to December
tourists = [predictions] # Example number of tourists per month

# Plotting the line graph
plt.figure(figsize=(8, 6))
plt.plot(months, tourists, marker='o', color='b', linestyle='-', linewidth=1, markersize=6)

# Adding title and labels
plt.title("Crowd Comparison")
plt.xlabel("Month")
plt.ylabel("No. of Tourists")

# Setting grid and displaying the plot
plt.grid(True)
plt.show()


pickle.dump(lr_model,open('ml.pkl','wb'))



#json data fetching
'''
    # Get JSON data from the request
    data = request.get_json()

    # Prepare the new features for prediction
    new_data = pd.DataFrame(data, index=[0])

    # Use the trained model to make predictions
    predicted_total = lr_model.predict(new_data)

    # Return the prediction as JSON
    return jsonify({'predicted_total': predicted_total[0]})
    '''


#flask hosting
'''
mode='prod'
if __name__ == '__main__':
    if mode=='prod':
        app.run(host='0.0.0.0' , port=5500 , debug=True)
    else:
        serve(app, host='0.0.0.0', port=5500, threads=10 )  
'''

#added changes
'''
# Prepare the new features for prediction
new_data = {
    'foreign': [10],  # Replace with the actual value
    'native': [11],  # Replace with the actual value
    # Add other features similarly
}

# Create a DataFrame from the new data
new_df = pd.DataFrame(new_data)

# Use the trained model to make predictions
predicted_total = lr_model.predict(new_df)

print('Predicted Total:', predicted_total[0])
'''
