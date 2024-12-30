# 1
# input_1 = input('enter first number')
# input_2 = input('enter second number')
# input_3 = input('enter third number')
# print(max(input_1, input_2, input_3))

# 2

# 3
# def arithmetic_mean(x, y):
#  ar_mean = (x + y)/2
#  print(ar_mean)

# arithmetic_mean(10, 20)
# arithmetic_mean(5, 3)
# arithmetic_mean(7, 5)

# 4
# def is_number_odd(x):
#     if x % 2 == 0:
#         print('number is even')
#     else: print('number is odd')

# is_number_odd(3)

# 5
# a, b, c, d, e, f, g, h, i, j = map(int, input('Insert 10 digits: ').split())
# numbers = [a, b, c, d, e, f, g, h, i, j]


# def arithmetic_mean(a, b, c, d, e, f, g, h, i, j):
#     ar_mean = (a + b + c + d + e + f + g + h + i + j) / 10
#     print(f'Arithmetic Mean: {ar_mean}')
#     return ar_mean

# def median(numbers):
#     sorted_numbers = sorted(numbers)
#     n = len(sorted_numbers)
#     if n % 2 == 1:
#         median_value = sorted_numbers[n // 2]
#     else:
#         median_value = (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
#     print(f'Median: {median_value}')
#     return median_value

# def mode(numbers):
#     obj = {}
#     for num in numbers:
#         if not num in obj:
#             obj[num]=1
#         else:
#             obj[num]+=1
#     print(f'Median: {[g for g, l in obj.items() if  l == max(obj.values())]}')
#     return [g for g, l in obj.items() if  l == max(obj.values())]


# arithmetic_mean(*numbers)
# median(numbers)
# mode(numbers)

#6
# def isPalindrome(s):
#     rev = ''.join(reversed(s))

#     if (s == rev):
#         return True
#     return False

# s = "123456"
# ans = isPalindrome(s)

# if (ans):
#     print("Yes")
# else:
#     print("No")

# 7
# extensions = ['txt', 'jpg', 'gif', 'html']
# user_input = input('Enter file name: ')

# if any(user_input.endswith(f".{ext}") for ext in extensions):
#     print('Yes')
# else:
#     print('No')

# 8
# import random

# numbers = [1, 4, 6, 3, 9, 2, 6]

# random.shuffle(numbers)

# print(numbers)

# 9
# list_1 = [1, 4, 5, 6, 4, 3, 7]
# list_2 = [4, 2, 5, 6, 2, 8, 9]

# def match(list_1, list_2):
#     # Check if there is any common element in the two lists
#     if any(digit in list_2 for digit in list_1):
#         print('Some of the numbers do match!')
#     else:
#         print('Numbers do not match')

# match(list_1, list_2)

# 10
for number in range(20, 126):
    if number % 5 == 0:
        print(number)






