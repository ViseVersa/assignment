#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import pandas as pd

# Function to fetch data from API
def fetch_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

# API URLs
expiries_api = "https://live.markethound.in/api/history/expiries?index=FINNIFTY"
decay_api = "https://live.markethound.in/api/history/decay?name=FINNIFTY&expiry=2024-03-05T00:00:00.000Z&dte=0"

# Fetching data
expiries_data = fetch_data(expiries_api)
decay_data = fetch_data(decay_api)

# Convert to DataFrame if needed
# expiries_df = pd.DataFrame(expiries_data)
# decay_df = pd.DataFrame(decay_data)


# In[3]:


# Example of preprocessing - to be adjusted based on actual data structure
def preprocess_data(data):
    df = pd.DataFrame(data)
    # Handle missing values, convert data types, etc.
    df = df.dropna()  # Example: removing missing values
    return df

# Preprocess the fetched data
# processed_expiries = preprocess_data(expiries_data)
# processed_decay = preprocess_data(decay_data)


# In[28]:


from sklearn.ensemble import GradientBoostingRegressor

# Initialize the predictions dictionary
predictions = {}

# Loop through each feature to create a model and predict the next day's value
for feature in ['open', 'high', 'low', 'close']:
    # Separate features and target for modeling
    X = df.drop(feature, axis=1)
    y = df[feature]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    test_predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, test_predictions)
    print(f"Mean Squared Error for {feature}: {mse}")

    # Predict the next day's value using the trained model
    next_day_data = df.iloc[-1].drop(feature)
    next_day_df = pd.DataFrame([next_day_data], columns=X_train.columns)

    next_day_prediction = model.predict(next_day_df)
    predictions[feature] = next_day_prediction[0]

print(f"Predicted values for the next day: {predictions}")

