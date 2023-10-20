def printTop20(counts):
    d = {}
    d = sorted(counts.items(),key= lambda a: a[1],reverse=True)

    for j in range(20):
        print('{} = {}'.format(d[j][0],str(d[j][1])))
    