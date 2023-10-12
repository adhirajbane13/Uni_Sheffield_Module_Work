from pylab import *

infile = open('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab05_code\\pulse_data.txt')
pulse = list(infile)
pul = []
for i in range(0,len(pulse)):
    pul.append(float(pulse[i].split()[0]))

#data = [4, 3, 5, 7.5, 3.8, 1.5]

minval = min(pul)
maxval = max(pul)

vspan = maxval - minval
bins = 50

d = [0]*bins

for dt in pul:
    k = int(bins*(dt-minval)/vspan) - 1
    d[k] = d[k] + 1

plot(d)

#Bar plot

bid = range(1,51)

figure()
bar(bid,d)

#histogram
figure()
hist(pul,bins)
show()