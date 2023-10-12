
from pylab import *

####################################
# Read data

data = []
inf = open('pulse_data.txt')
for line in inf:
    data.append(float(line))
inf.close()

####################################
# Initial plots

figure() # start a new figure
plot(data) # plot the unsorted data

data.sort() # sorted the values into ascending order

figure() # start a new figure
plot(data) # plot the sorted data

####################################
# Binning - compute bin counts

BINS = 50
minval = min(data)
maxval = max(data)

# initialise counting list:
bincounts = []
for i in range(BINS):
    bincounts.append(0)

# Count data into bins:
for value in data:
    binid = int((value - minval) / (maxval - minval) * BINS)
    if binid == BINS: 
        binid = BINS - 1
    bincounts[binid] += 1

####################################
# Plot bin counts onto a standard graph

figure()
plot(bincounts,'o')

####################################
# Plot bin counts as a bar chart proper

figure()
bar(range(BINS),bincounts)

####################################
# Get hist function to compute histogram from data

figure()
hist(data,BINS)

####################################
# Final call to show plotted figures

show()

