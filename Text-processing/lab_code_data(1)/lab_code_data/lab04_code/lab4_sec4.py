infile = open('noisy_signal.txt')

sig = list(infile)
signal = list(0*range(0,len(sig)))

for i in range(0,len(sig)):
    
w_size = 20
