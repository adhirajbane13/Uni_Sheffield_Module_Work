
###########################
# Factorial

def factorial(n):
    fact = 1
    while n > 1:
        fact = fact * n
        n = n - 1
    return fact

###########################
# Guessing Game

from random import randint # import statements should really be at top of file

def guess(attempts,numrange):
    number = randint(1,numrange)
    print("Welcome! Can you guess my secret number?")
    while attempts > 0:
        print('You have', attempts, 'guesses remaining')
        guess = input('Make a guess: ')
        guess = int(guess)
        if number == guess:
            print("Well done! You got it right.")
            break
        elif guess < number:
            print("No - too low!")
        elif guess > number:
            print("No - too high!")
        attempts = attempts - 1
    if attempts == 0:
        print("No more guesses - bad luck!")
    print("GAME OVER: thanks for playing. Bye.")

###########################


