import requests
from bs4 import BeautifulSoup
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

# ==================== Web Scraping ==================== #

# URL of the page to scrape
url = "https://coinmarketcap.com/"  # Replace with the correct URL if necessary

# Fetch the page content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table (update the class name if necessary)
table = soup.find('table')
if table is None:
    print("Table not found on the page!")
    exit()

# Extract rows from the table
rows = table.find_all('tr')[1:]  # Skip header row

# Initialize a list to store the data
crypto_data = []

# Iterate over each row and extract the required information
for row in rows:
    columns = row.find_all('td')
    if len(columns) < 10:  # Skip rows that don't have enough columns
        continue
    try:
        rank = columns[1].get_text(strip=True) if len(columns) > 1 else "N/A"
        name = columns[2].find('p', class_='coin-item-name').get_text(strip=True) if columns[2].find('p', class_='coin-item-name') else "N/A"
        symbol = columns[2].find('p', class_='coin-item-symbol').get_text(strip=True) if columns[2].find('p', class_='coin-item-symbol') else "N/A"
        price = columns[3].get_text(strip=True) if len(columns) > 3 else "N/A"
        change_1h = columns[4].get_text(strip=True) if len(columns) > 4 else "N/A"
        change_24h = columns[5].get_text(strip=True) if len(columns) > 5 else "N/A"
        change_7d = columns[6].get_text(strip=True) if len(columns) > 6 else "N/A"
        market_cap = columns[7].get_text(strip=True) if len(columns) > 7 else "N/A"
        volume_24h = columns[8].find('p', class_='font_weight_500').get_text(strip=True) if columns[8].find('p', class_='font_weight_500') else "N/A"
        supply = columns[9].get_text(strip=True) if len(columns) > 9 else "N/A"

        # Append to the data list
        crypto_data.append({
            'Rank': rank,
            'Name': name,
            'Symbol': symbol,
            'Price': price,
            '1h Change': change_1h,
            '24h Change': change_24h,
            '7d Change': change_7d,
            'Market Cap': market_cap,
            '24h Volume': volume_24h,
            'Circulating Supply': supply
        })
    except Exception as e:
        print(f"Error processing row: {e}")
        continue

# Convert to a pandas DataFrame
df = pd.DataFrame(crypto_data)

# Save the data to a CSV file for further processing
df.to_csv('crypto_data.csv', index=False)

# ==================== Data Preprocessing ==================== #

# Load the dataset
df = pd.read_csv('crypto_data.csv')

# Clean and convert string values to numeric
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

# ==================== Model Training ==================== #

# Train a Random Forest Regressor
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

# ==================== Visualization ==================== #

# 1. Correlation Heatmap
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

# 3. Market Cap vs. Volume (Scatter Plot with Names)
plt.figure(figsize=(10, 6))
plt.scatter(df['Market Cap'], df['24h Volume'], alpha=0.5)
for i in range(len(df)):
    plt.text(df['Market Cap'].iloc[i], df['24h Volume'].iloc[i], df['Name'].iloc[i], fontsize=8, alpha=0.7)
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

# 6. Price Over Rank (Line Plot with Names)
if 'Rank' in df.columns:
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df_sorted = df.dropna(subset=['Rank']).sort_values(by='Rank')  # Drop NaN ranks
    plt.figure(figsize=(12, 6))
    plt.plot(df_sorted['Rank'], df_sorted['Price'], marker='o')
    for i in range(len(df_sorted)):
        plt.text(df_sorted['Rank'].iloc[i], df_sorted['Price'].iloc[i], df_sorted['Name'].iloc[i], fontsize=8, alpha=0.7)
    plt.title("Price Over Rank")
    plt.xlabel("Rank")
    plt.ylabel("Price (Log-Transformed)")
    plt.show()

# 7. Pairplot for All Features
sns.pairplot(df[['1h Change', '24h Change', '7d Change', 'Market Cap', '24h Volume', 'Price']], diag_kind='kde')
plt.suptitle("Pairplot for Features", y=1.02)
plt.show()
