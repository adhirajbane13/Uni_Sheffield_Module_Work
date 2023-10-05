def check_PIN(truepin):
    for i in range(1,4):
        user_pin = int(input("Kindly, provide your PIN: "))
        if user_pin == int(truepin):
            return True
        else:
            print("Please, Try Again!!")
            if i == 3:
                return False

def withdrawl(amount,limit):
    if amount <= limit and amount%10 == 0:
        return True
    else:
        return False

def cashpoint(truepin,balance):
    if check_PIN(truepin):
        print("Welcome to Cashpoint!!")
        limit = 100
        amt = int(input("Please tell how much amount you want to withdraw: "))
        if withdrawl(amt, limit):
            return "Here is your cash of amount: {}".format(amt)
        else:
            return "Sorry, withdrawl not possible"
    
    else:
        return "Sorry, PIN failed!!"

