
from soln_lab7_countwords import *

################################
# Testing tasks 1-3 

infile = 'mobypara.txt'
infile = 'mobydick.txt'

stops = readStopWords('stopwords.txt')
#stops = []

moby_counts = countWords(infile, stops)
printTop20(moby_counts)

################################
# Testing task 4

infiles = ['george01.txt', 
           'george02.txt', 
           'george03.txt', 
           'george04.txt']

allcounts = []

for infile in infiles:
    counts = countWords(infile, stops)
    allcounts.append(counts)

for i in range(len(allcounts)):
    for j in range(len(allcounts)):
        if i < j:
            name1 = infiles[i]
            name2 = infiles[j]
            sim = similarity(allcounts[i], allcounts[j])
            print('compare: (%s <> %s) = %.3f' % (name1, name2, sim))

