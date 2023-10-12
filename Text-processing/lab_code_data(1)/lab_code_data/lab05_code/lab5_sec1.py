from pylab import *
infile = open('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab05_code\\pulse_data.txt')
pulse = list(infile)
pul = []
for i in range(0,len(pulse)):
    pul.append(float(pulse[i].split()[0]))

x = list(range(0,len(pul)))

pul.sort()
plot(x,pul)
show()