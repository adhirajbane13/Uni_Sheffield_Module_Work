def similarity(d1,d2):
    list1 = list(d1.keys())
    list2 = list(d2.keys())

    N = 0

    for i in list1:
        for j in list2:
            if i == j:
                N = N + 1
    
    score = N/(len(list1) + len(list2) - N)
    return score