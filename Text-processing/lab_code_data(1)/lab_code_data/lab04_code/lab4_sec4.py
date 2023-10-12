from pylab import *
infile = open('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab04_code\\noisy_signal.txt')

w_size = 20
sig0 = list(infile)
sig1 = []
time = []
for v in range(0,len(sig0)):
    val = sig0[v].split()
    sig1.append(float(val[1]))
    time.append(float(val[0]))

signal = [0]*len(sig1)

for i in range(0,len(sig1)-w_size):
    signal[i] = sum(sig1[i:i+w_size])/len(sig1[i:i+w_size])

for i in range(len(sig1)-w_size+1,len(sig1)):
    signal[i] = sum(sig1[i-w_size:i])/len(sig1[i-w_size:i])

plot(time,signal)
figure()
plot(time,sig1)
show()

