
def cashpoint(truepin,balance):
    pinAttempt = input('Please enter your PIN: ')
    if pinAttempt == truepin:
        print("\nPlease choose your transaction type")
        print("   - to request a balance  - enter 1")
        print("   - to make a withdrawal  - enter 2")
        print("   - to top-up a telephone - enter 3")
        transactionType = input('\nEnter your choice: ')
        if (transactionType == '1'):
            print("Your balance is", balance, 'GBP')
            print("Thank you for using this service.")
            return "balance-request"
        elif transactionType == '2':
            amount = float(input('Please enter your withdrawal amount: '))
            print("Please take your cash.")
            print("Your new balance is", balance - amount, 'GBP')
            return ("withdrawal-request",amount)
        elif transactionType == '3':
            print("\nSorry: the top-up service is not currently available")
            return ("mobile-topup",0)
        else:
            print("Error: transaction type not recognised.")
            return "transaction-error"
    else:
        print("\nError: PIN authorisation failed - exiting.")
        return "PIN-error"




