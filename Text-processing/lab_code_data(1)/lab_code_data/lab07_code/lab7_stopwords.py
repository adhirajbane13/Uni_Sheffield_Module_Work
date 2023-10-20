def readStopWords(filename):
    stops = []
    infile = open(filename)
    a = list(infile)
    for i in range(len(a)):
        z = a[i].split()
        for j in z:
            stops.append(j)
    return stops