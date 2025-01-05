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
import numpy as np
import random

m, n = map(int, input().split())

matrix = []

for row in range(m):
    a = []  # Corrected indentation here

    for column in range(n):   
        a.append(random.randint(0, 100))
    matrix.append(a)

for row in range(m):
    for column in range(n):
        print(matrix[row][column], end=" ")
    print()

