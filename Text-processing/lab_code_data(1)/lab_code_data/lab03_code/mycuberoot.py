import math

def MyCubeRoot(A):
    e = 0.001
    x = A
    while (math.pow(x, 3) - A >= e):
        xnew = x - (math.pow(x, 3) - A)/(3*(math.pow(x, 2)))
        x = xnew
    return x
A = float(input("Enter the number: "))
print(MyCubeRoot(A))