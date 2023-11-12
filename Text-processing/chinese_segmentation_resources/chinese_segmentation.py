"""
USE: python <PROGNAME> (options) WORDLIST-FILE INPUT-FILE OUTPUT-FILE
OPTIONS:
    -h : print this help message and exit
"""
################################################################

import sys

################################################################

MAXWORDLEN = 5

################################################################
# Command line options handling, and help

def print_help():
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit(0)
    
if '-h' in sys.argv or len(sys.argv) != 4:
    print_help()

word_list_file = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]

################################################################

# PARTS TO COMPLETE: 

################################################################
# READ CHINESE WORD LIST
# Read words from Chinese word list file, and store in 
# a suitable data structure (e.g. a set)
wordset = set()
with open(word_list_file, "r", encoding = "utf-8") as myfile:
    list1 = list(myfile)
    for word in list1:
        k = word.split()
        wordset.add(k[0])

################################################################
# FUNCTION TO PROCESS ONE SENTENCE
# Sentence provided as a string. Result returned as a list of strings 

def segment(sent, wordset):
    sent1 = []
    while len(sent) > 0:
        word = sent[0:1]
        for i in range(1,MAXWORDLEN+1):
            if sent[0:i] in wordset:
                word = sent[0:i]
        sent1.append(word)
        sent = sent[len(word):len(sent)]
    return sent1         # write code for main loop first. 



################################################################
# MAIN LOOP
# Read each line from input file, segment, and print to output file
with open(input_file, "r", encoding = "utf-8") as infile:
    f = open(output_file,'w',encoding = "utf-8")
    list2 = list(infile)
    for word1 in list2:
        j = word1.split()[0]
        sent1 = segment(j,wordset)
        sent2 = ''
        for i in range(len(sent1)):
            if i == len(sent1)-1:
                sent2 = sent2 + sent1[i]
            else:
                sent2 = sent2 + sent1[i] + ' '
        sent2 = sent2 + '\n'
        f.write(sent2)

f.close()

################################################################

