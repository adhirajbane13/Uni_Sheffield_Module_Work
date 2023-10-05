
from random import randint

def guess(attempts,numrange):
    number = randint(1,numrange)
    print("Welcome! Can you guess my secret number?")
    for i in range(attempts,0,-1):
        print(f"You have {i} guesses remaining")
        guess_no = int(input("Make a guess: "))
        if guess_no == number:
            print("Well done! You got it right.")
            print("GAME OVER: thanks for playing. Bye.")
            return
        elif guess_no < number:
            print("No - too low!")
        else:
            print("No - too high!")
    
    print("No more guesses - bad luck!")
    print("GAME OVER: thanks for playing. Bye.")

guess(5,10)

