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
import pandas as pd
import random
import string

categories = ["CS", "FL", "DZ", "AS", "OK"]

df = pd.DataFrame(
    {
        "Product": [
            ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 100)))
            for _ in range(100)
        ],
        "Category": [
            categories[random.randint(0, 4)]
            for _ in range(100)
        ],
        "Price": [
            random.uniform(5, 100)  # Generate a random price between 5 and 100
            for _ in range(100)  # Loop 100 times to generate 100 prices
        ],
        "Quantity": [
            random.randint(1, 100)
            for _ in range(100)
        ]
    }
)

# Calculate Revenue as Price * Quantity
df['Revenue'] = df['Price'] * df['Quantity']
print('print df:', df)

