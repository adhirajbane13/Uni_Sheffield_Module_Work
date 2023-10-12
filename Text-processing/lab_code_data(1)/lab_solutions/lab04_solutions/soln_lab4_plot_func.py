
from pylab import *

Xs = []
Fs = []
Gs = []

for x in range(1,21):
    Xs.append(x)
    Fs.append(x ** 2 + 20)
    Gs.append((x/2.0) ** 3 - 100)

plot(Xs,Fs,'b-o')
plot(Xs,Gs,'r-*')
show()



