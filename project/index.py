import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the CSV file
data = pd.read_csv('teams_data.csv')

# Convert columns to numeric for analysis
data['Wins'] = pd.to_numeric(data['Wins'], errors='coerce')
data['Losses'] = pd.to_numeric(data['Losses'], errors='coerce')
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# Calculate Winning Percentage
data['Winning_Percentage'] = data['Wins'] / (data['Wins'] + data['Losses']) * 100

# Drop rows with missing or infinite values (if any)
data = data.dropna()

# Features (X) and Target (y)
X = data[['Losses', 'Year']]  # Use 'Losses' and 'Year' as features
y = data['Winning_Percentage']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Scatter Plot of Actual vs Predicted Winning Percentages
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Winning Percentages', fontsize=16)
plt.xlabel('Actual Winning Percentage', fontsize=12)
plt.ylabel('Predicted Winning Percentage', fontsize=12)
plt.grid(True)
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
plt.bar(feature_names, importances, color='skyblue')
plt.title('Feature Importance', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.grid(True)
plt.show()
