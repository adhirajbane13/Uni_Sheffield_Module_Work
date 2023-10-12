from pylab import *

x = list(range(1,21))
Ys = []
for i in x:
    Ys.append(i**2 + 20)
plot(x,Ys)
figure()
show()