import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page to scrape
url = "https://coinmarketcap.com/"  # Replace with the correct URL if different

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

# Display the data
print(df)

# Optionally, save the data to a CSV file
df.to_csv('crypto_data.csv', index=False)
