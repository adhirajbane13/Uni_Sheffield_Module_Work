def sum_list(x):
    s = 0
    for i in x:
       s+= i
    return s

print(sum_list([3,4,5]))

def square_list(x):
    newlist = [i**2 for i in x]
    return newlist

print(square_list([3,4,5]))