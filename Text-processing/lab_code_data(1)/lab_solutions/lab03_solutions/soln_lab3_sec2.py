
###########################
# Newton's method for Square Roots

def mySqrt(A):
    x = 1.0
    errorTolerance = 0.001
    while abs(x ** 2 - A) > errorTolerance:
        x = x - (x ** 2 - A)/(2 * x)
    return x

###########################
# Newton's method for Square Roots

def myCubeRoot(A):
    x = 1.0
    errorTolerance = 0.001
    while abs(x ** 3 - A) > errorTolerance:
        x = x - (x ** 3 - A)/(3 * x ** 2)
    return x

###########################



