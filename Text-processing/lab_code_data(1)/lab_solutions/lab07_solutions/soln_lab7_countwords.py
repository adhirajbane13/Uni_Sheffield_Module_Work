
##############################
# Version of word counting function that does NOT
# use a stoplist to filter words that are counted

def countWords(filename):
    counts = {}
    inf = open(filename, 'r')
    for line in inf:
        words = line.split()
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    return counts


# Version of word counting function that DOES use
# a stoplist, to filter words that are counted

def countWords(filename, stops):
    counts = {}
    inf = open(filename, 'r')
    for line in inf:
        words = line.split()
        for word in words:
            if word not in stops:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
    return counts


##############################
# Read in stop words, and return as a list

def readStopWords(stopfile):
    stops = []
    inf = open(stopfile, 'r')
    for line in inf:
        word = line.strip()
        stops.append(word)
    return stops

##############################
# Takes a dictionary of counts (e.g. for words)
# and prints out the 20 most frequent keys

def printTop20(counts):
    words = list(counts.keys())
    words.sort(reverse=True, key=lambda v:counts[v])

    for i in range(20):
        word = words[i]
        print(i+1, ':', word, '=', counts[word])

##############################
# Compute similarity score between two dictionaries
# of counts

def similarity(counts1, counts2):
    size1 = len(counts1)
    size2 = len(counts2)
    if size1 + size2 > 0: # avoid a divide-by-zero error
        overlap = 0
        for w in counts1:
            if w in counts2:
                overlap += 1
        return overlap / (size1 + size2 - overlap)
    else:
        return 0.0 

