# 1
# exchange = {"USD": 2.72, "EUR": 2.96, "GBP": 3.54}

# def task_1(exchange):
#     convertable = input('Please enter the money to convert (ex: 10 USD): ')
    
#     try:
#         amount, currency = convertable.split()
#         amount = float(amount)
        
#         if currency in exchange:
#             converted_amount = amount * exchange[currency]
#             print(f"{amount} {currency} is {converted_amount:.2f} in the local currency.")
#         else:
#             print("Unknown currency. Please use USD, EUR, or GBP.")
#     except ValueError:
#         print("Invalid input. Please use the format: '<amount> <currency>'.")

# task_1(exchange)

# 2
# import random

# keys = random.sample(range(10, 101), 10)

# result_dict = {key: sum(map(int, str(key))) for key in keys}

# print(result_dict)

# 3
# import numpy as np
# import random

# while True:
#     try:
#         m, n = map(int, input('Please enter dimensions (separate digits using space): ').split())
#         if m <= 0 or n <= 0:  
#             raise ValueError("Dimensions must be positive integers.")
#         break
#     except ValueError as e:
#         print(f"Invalid input: {e}. Please enter two positive integers separated by a space.")

# matrix = []

# for row in range(m):
#     a = []
#     for column in range(n):   
#         a.append(random.randint(0, 100)) 
#     matrix.append(a)

# for row in range(m):
#     for column in range(n):
#         print(matrix[row][column], end=" ")
#     print()

# 4
import pandas as pd

ford_cars = pd.read_excel("ford_escort.xlsx")

ford_cars.columns = ford_cars.columns.str.strip()

ford_cars['Distance km'] = ford_cars['Distance mile'] * 1.60934

print(ford_cars)



