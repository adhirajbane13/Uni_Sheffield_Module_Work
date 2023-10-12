
from pylab import plot, subplot, show, figure

###########################
# Load noisy signal data

time = []
noisy = []
f = open('noisy_signal.txt','r')
for line in f:
    values = line.split()
    time.append(float(values[0]))
    noisy.append(float(values[1]))
f.close()

###########################
# Compute filtered signal

# Window size:
w = 20

# initialise filtered signal list 
filtered = []
for i in range(len(noisy)):
    filtered.append(0)

# compute moving average - this version treats the 
# situation where 'window' runs over start/end of 
# date as a special case, that is not computed. 

#for i in range(len(noisy)):
#    start = int(i-(w/2))
#    end = start+w
#    if start >= 0 and end < len(noisy):
#        filtered[i] = sum(noisy[start:end])/w

# Following is an alternative for computing the moving 
# average, exploits fact that python slicing is 'permissive'
# w.r.t. specifying positions outside of list

#for i in range(len(noisy)):
#    start = int(i-(w/2))
#    end = start+w
#    filtered[i] = sum(noisy[start:end])/w
    
# BUT if start value is negative, slicing returns an empty list,
# which can present problems if you decide to divide sum by 
# length of window (i.e. gives a divide-by-zero error). 
# This might be addressed as follows: 

for i in range(len(noisy)):
    start = int(i-(w/2))
    end = start+w
    if start < 0:
        start = 0
    window = noisy[start:end]
    filtered[i] = sum(window)/len(window)

###########################
# Plotting

plot(time, noisy,'b')
figure()
plot(time, filtered,'r')

show()

