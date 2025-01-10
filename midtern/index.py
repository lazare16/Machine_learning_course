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
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import pandas as pd
# import time

# Initialize Selenium WebDriver
# service = Service(r"C:\Users\Lazare\Documents\Chrome-driver\chromedriver.exe")  # Add the actual WebDriver file
# driver = webdriver.Chrome(service=service)

# Define the URL of the website
# base_url = "https://zoommer.ge/"

# Open the website
# driver.get(base_url)

# Wait for the page to load (adjust timeout as needed)
# wait = WebDriverWait(driver, 10)
# wait.until(EC.presence_of_element_located((By.CLASS_NAME, "sc-38bab3e0-0")))

# Scroll down to load more products (if applicable)
# Adjust the scrolling logic based on the website's behavior
# for _ in range(5):  # Adjust range for how much you want to scroll
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     time.sleep(2)  # Wait for new products to load

# Extract product information
# products = []
# try:
#     product_elements = driver.find_elements(By.CLASS_NAME, "sc-38bab3e0-0")  # Update class based on site structure
#     for product in product_elements:
#         try:
#             name_element = product.find_element(By.CSS_SELECTOR, ".sc-38bab3e0-5 h2")  # Update selector
#             name = name_element.text.strip() if name_element else "No Name"

#             price_element = product.find_element(By.CSS_SELECTOR, ".sc-38bab3e0-6 h4")  # Update selector
#             price = price_element.text.strip() if price_element else "No Price"

#             # Add product data to the list
#             products.append({"Name": name, "Price": price})
#         except Exception as e:
#             print(f"Error extracting product details: {e}")
# except Exception as e:
#     print(f"Error locating products: {e}")

# Save data to CSV
# if products:
#     df = pd.DataFrame(products)
#     df.to_csv("products_selenium.csv", index=False, encoding="utf-8")
#     print("Data saved to products_selenium.csv")
# else:
#     print("No products found")

# Close the WebDriver
# driver.quit()





