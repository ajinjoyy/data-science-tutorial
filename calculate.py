# Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the dataset
# Replace 'data/Advertising.csv' with the correct path to your CSV file
data = pd.read_csv('data/Advertising.csv')

# Define predictors (TV, radio, newspaper) and target (sales)
X = data[['TV', 'radio', 'newspaper']]  # Predictors
y = data['sales']  # Target

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Calculate Residual Standard Error (RSE)
residuals = model.resid  # Residuals
n = len(y)  # Number of observations
p = X.shape[1] - 1  # Number of predictors
RSE = np.sqrt(np.sum(residuals**2) / (n - p - 1))
print(f"Residual Standard Error (RSE): {RSE}")

# Extract F-statistics
f_stat = model.fvalue
print(f"F-statistics: {f_stat}")

