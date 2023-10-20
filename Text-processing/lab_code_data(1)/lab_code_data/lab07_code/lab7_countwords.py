def countWords(filename,stopwords):
    d = {}
    infile = open(filename)
    file_list = list(infile)
    for i in range(len(file_list)):
        k = file_list[i].split()
        for j in range(len(k)):
            if k[j] != '\n'and k[j] not in d.keys() and k[j] not in stopwords:
                d[k[j]] = 1
            elif k[j] in d.keys():
                d[k[j]] = d[k[j]] + 1
    return d
