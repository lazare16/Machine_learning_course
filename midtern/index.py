# 1
# def printRoman(number):
# 	num = [1, 4, 5, 9, 10, 40, 50, 90,
# 		100, 400, 500, 900, 1000]
# 	sym = ["I", "IV", "V", "IX", "X", "XL",
# 		"L", "XC", "C", "CD", "D", "CM", "M"]
# 	i = 12
	
# 	while number:
# 		div = number // num[i]
# 		number %= num[i]

# 		while div:
# 			print(sym[i], end = "")
# 			div -= 1
# 		i -= 1

# # Driver code
# if __name__ == "__main__":
# 	number = int(input('enter Arabic number:'))
# 	print("Roman value is:", end = " ")
# 	printRoman(number)

# 2 
# import pandas as pd
# import random
# import string

# categories = ["CS", "FL", "DZ", "AS", "OK"]

# df = pd.DataFrame(
#     {
#         "Product": [
#             ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 100)))
#             for _ in range(100)
#         ],
#         "Category": [
#             categories[random.randint(0, 4)]
#             for _ in range(100)
#         ],
#         "Price": [
#             random.uniform(5, 100)  # Generate a random price between 5 and 100
#             for _ in range(100)  # Loop 100 times to generate 100 prices
#         ],
#         "Quantity": [
#             random.randint(1, 100)
#             for _ in range(100)
#         ]
#     }
# )

# # Calculate Revenue as Price * Quantity
# df['Revenue'] = df['Price'] * df['Quantity']
# print('print df:', df)

# 3
# import pandas as pd
# import matplotlib.pyplot as plt  # Make sure to import matplotlib

# data = pd.read_csv('timeseries.csv')


# if not data.empty:  # Ensure there are at least 5 rows of data
#     # Use the 'income' column directly
#     df = pd.DataFrame({
#         'income': data['income'],  # Use the 'income' column from the loaded data
      
#     }, index=[data['date']])

#     # Plotting the line chart for 'income' and 'horse'
#     df.plot.line()

#     # Show the plot
#     plt.show()
# else:
#     print("Error: No data or insufficient data in the CSV file.")

# 4
import requests
from bs4 import BeautifulSoup

url = 'https://roamingo.com/'

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    sim_wrapper = soup.find('div', class_='popularCountries_countries_wrapper__iBjva')

    if sim_wrapper:
        # Find all country items inside the wrapper div
        esim = sim_wrapper.find_all('div', class_='countryItem_container__WhnsR')

        # Iterate over each country item in 'esim'
        for e in esim:
            # Find the <p> tag inside the country item (which contains the price)
            price = e.find('p')
            
            if price:
                # If price is found, print it (or any other action you need)
                print(price.get_text())
            else:
                # If no price is found
                print('Price not found')
    else:
        print('Wrapper div not found')
else:
    print('Failed to retrieve the webpage')





