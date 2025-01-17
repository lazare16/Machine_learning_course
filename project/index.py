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

# ==================== Web Scraping ==================== #
url = "https://coinmarketcap.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract the table from the page
table = soup.find('table')
if table is None:
    print("Table not found on the page!")
    exit()

# Extract rows from the table
rows = table.find_all('tr')[1:]  # Skip the header row
crypto_data = []

for row in rows:
    columns = row.find_all('td')
    if len(columns) < 10:  
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

        # Append data to list
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

# Convert the data to a DataFrame
df = pd.DataFrame(crypto_data)

# Save the raw data to CSV (optional)
df.to_csv('crypto_data.csv', index=False)

# ==================== Data Preprocessing ==================== #
df = pd.read_csv('crypto_data.csv')

# Clean and convert string values to numeric
def clean_and_convert(value):
    if isinstance(value, str):
        value = re.sub(r'[^\d.-]', '', value)  # Remove non-numeric characters
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

# Apply cleaning to relevant columns
for col in ['Price', '1h Change', '24h Change', '7d Change', 'Market Cap', '24h Volume', 'Circulating Supply']:
    df[col] = df[col].apply(clean_and_convert)

# Drop rows with missing values
df = df.dropna()

# Log transformation for skewed data
for col in ['Price', 'Market Cap', '24h Volume']:
    df[col] = np.log1p(df[col])

# Prepare features and target
features = ['1h Change', '24h Change', '7d Change', 'Market Cap', '24h Volume', 'Circulating Supply']
target = 'Price'

X = df[features]
y = df[target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==================== Model Training ==================== #
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict prices for the entire dataset
df['Predicted Price (Log-Transformed)'] = model.predict(X_scaled)

# Reverse log transformation to get actual prices
df['Predicted Price'] = np.expm1(df['Predicted Price (Log-Transformed)'])

# ==================== Print Predictions ==================== #
print("\nPredicted Prices for Cryptocurrencies:")
print(df[['Name', 'Symbol', 'Predicted Price']])

# ==================== Evaluation Metrics ==================== #
y_test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\nMean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the predictions to a CSV file
df[['Name', 'Symbol', 'Predicted Price']].to_csv('predicted_crypto_prices.csv', index=False)
print("\nPredicted prices saved to 'predicted_crypto_prices.csv'.")
