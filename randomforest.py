import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#reading the csv file
df = pd.read_csv('data/advertising.csv')


# Separate Features and Target
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

# Split into Training and Testing Sets
#X-input features, y-target variable
#test_size=0.2 means 20% of the data will be used for testing
#random_state=42 sets the seed for random number generator
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and Train the Random Forest Regressor
#n_estimators=100 means 100 trees in the forest
#random_state=42 sets the seed for random number generator
#fit() method is used to train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on Test Data
y_pred = rf_model.predict(X_test)
print("Predicted Sales:", y_pred[:5])
print("Actual Sales:", y_test[:5].values)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

#mse and r2 are the metrics used to evaluate the model
#Mean Squared Error (MSE): 0.59
#R-squared (R2 Score): 0.98

# Output:
# Predicted Sales: [16.84 20.57 21.18 10.64 22.47] Actual Sales: [17.6  20.8  21.2  8.7 22.6]





