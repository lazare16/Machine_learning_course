import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import re
import joblib

# Load the dataset
df = pd.read_csv('crypto_data.csv')

# Data Preprocessing
def clean_and_convert(value):
    """Clean and convert string values to numeric."""
    if isinstance(value, str):
        value = re.sub(r'[^\d.-]', '', value)  # Remove non-numeric characters
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

# Apply cleaning to numeric columns
for col in ['Price', '1h Change', '24h Change', '7d Change', 'Market Cap', '24h Volume', 'Circulating Supply']:
    df[col] = df[col].apply(clean_and_convert)

# Drop rows with missing or invalid values
df = df.dropna()

# Log transformation for skewed data
for col in ['Price', 'Market Cap', '24h Volume']:
    df[col] = np.log1p(df[col])

# Feature Engineering: Select relevant features and target
features = ['1h Change', '24h Change', '7d Change', 'Market Cap', '24h Volume', 'Circulating Supply']
target = 'Price'

X = df[features]
y = df[target]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the trained model
joblib.dump(model, 'crypto_price_predictor.pkl')
print("Model saved as 'crypto_price_predictor.pkl'")

# ===================== Visualization ===================== #

# 1. Correlation Heatmap
# Ensure only numeric columns are used for the correlation matrix
numeric_cols = df.select_dtypes(include=[np.number])  # Select only numeric columns
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 2. Price Distribution (Histogram)
plt.figure(figsize=(8, 5))
plt.hist(df['Price'], bins=30, edgecolor='k', alpha=0.7)
plt.title("Price Distribution")
plt.xlabel("Price (Log-Transformed)")
plt.ylabel("Frequency")
plt.show()

# 3. Market Cap vs. Volume (Scatter Plot)
plt.figure(figsize=(8, 5))
plt.scatter(df['Market Cap'], df['24h Volume'], alpha=0.5)
plt.title("Market Cap vs. 24h Volume")
plt.xlabel("Market Cap (Log-Transformed)")
plt.ylabel("24h Volume (Log-Transformed)")
plt.show()

# 4. Feature Importance (Bar Plot)
importances = model.feature_importances_
feature_names = features
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# 5. 1-Hour Change vs. 24-Hour Change (Joint Plot)
sns.jointplot(data=df, x='1h Change', y='24h Change', kind='scatter', height=6, color='blue')
plt.suptitle("1-Hour Change vs. 24-Hour Change", y=1.02)
plt.show()

# 6. Price Over Rank (Line Plot)
# Ensure the 'Rank' column is numeric for plotting
if 'Rank' in df.columns:
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df_sorted = df.dropna(subset=['Rank']).sort_values(by='Rank')  # Drop NaN ranks
    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted['Rank'], df_sorted['Price'], marker='o')
    plt.title("Price Over Rank")
    plt.xlabel("Rank")
    plt.ylabel("Price (Log-Transformed)")
    plt.show()

# 7. Pairplot for All Features
sns.pairplot(df[['1h Change', '24h Change', '7d Change', 'Market Cap', '24h Volume', 'Price']], diag_kind='kde')
plt.suptitle("Pairplot for Features", y=1.02)
plt.show()
