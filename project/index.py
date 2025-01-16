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

url = "https://coinmarketcap.com/"  
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')


table = soup.find('table')
if table is None:
    print("Table not found on the page!")
    exit()

rows = table.find_all('tr')[1:] 
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

df = pd.DataFrame(crypto_data)

df.to_csv('crypto_data.csv', index=False)

df = pd.read_csv('crypto_data.csv')

def clean_and_convert(value):
    if isinstance(value, str):
        value = re.sub(r'[^\d.-]', '', value)  
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

for col in ['Price', '1h Change', '24h Change', '7d Change', 'Market Cap', '24h Volume', 'Circulating Supply']:
    df[col] = df[col].apply(clean_and_convert)

df = df.dropna()

for col in ['Price', 'Market Cap', '24h Volume']:
    df[col] = np.log1p(df[col])

features = ['1h Change', '24h Change', '7d Change', 'Market Cap', '24h Volume', 'Circulating Supply']
target = 'Price'

X = df[features]
y = df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Prices (Log-Transformed):")
print(y_pred)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

numeric_cols = df.select_dtypes(include=[np.number])  
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ფასების განაწილება
plt.figure(figsize=(8, 5))
plt.hist(df['Price'], bins=30, edgecolor='k', alpha=0.7)
plt.title("Price Distribution")
plt.xlabel("Price (Log-Transformed)")
plt.ylabel("Frequency")
plt.show()

# ბაზრის კაპიტალიზაციის და მოცულობის გაბნევის დიაგრამა.
plt.figure(figsize=(10, 6))
plt.scatter(df['Market Cap'], df['24h Volume'], alpha=0.5)
for i in range(len(df)):
    plt.text(df['Market Cap'].iloc[i], df['24h Volume'].iloc[i], df['Name'].iloc[i], fontsize=8, alpha=0.7)
plt.title("Market Cap vs. 24h Volume")
plt.xlabel("Market Cap (Log-Transformed)")
plt.ylabel("24h Volume (Log-Transformed)")
plt.show()

# ფუნქციის მნიშვნელოვანობა
importances = model.feature_importances_
feature_names = features
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

#1 საათის ცვლილება vs 24 საათი
sns.jointplot(data=df, x='1h Change', y='24h Change', kind='scatter', height=6, color='blue')
plt.suptitle("1-Hour Change vs. 24-Hour Change", y=1.02)
plt.show()

# ფასი რეიტინგითან მიმართებაში
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


