
###########################
# Sum of List

def sum_list(values):
    sum = 0
    for val in values:
        sum = sum + val
    return sum

###########################
# Triangular Number

def triangular_number(n):
    total = 0
    for i in range(1,n+1):
        total = total + i
    return total

###########################
# List of Squares

def square_list(values):
    squares = []
    for val in values:
        squares.append(val ** 2)
    return squares

# An alternative definition, which refers to positions
# by index. It first makes a copy of the input list, then 
# replaces the value at each position with its square. 

def square_list(values):
    squares = list(values) # makes a copy 
    for i in range(len(squares)):
        squares[i] = squares[i] ** 2
    return squares

###########################
# list of triangular numbers

def triangular_list(values):
    trinums = []
    for val in values:
        trinums.append(triangular_number(val))
    return trinums

###########################

