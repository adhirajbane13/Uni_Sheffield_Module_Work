"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) file1...fileN
OPTIONS:
    -h : print this help message
    -s FILE : use stoplist file FILE
    -p : use Porter stemming (default: no stemming)
    -b : use BINARY weights (default: count weighting)
------------------------------------------------------------\
"""

import sys, re, getopt
import glob
from nltk.stem import PorterStemmer

opts, args = getopt.getopt(sys.argv[1:], 'hs:pbI:')
opts = dict(opts)

##############################
# HELP option

if '-h' in opts:
    progname = sys.argv[0]
    progname = progname.split('/')[-1] # strip out extended path
    help = __doc__.replace('<PROGNAME>', progname, 1)
    print(help, file=sys.stderr)
    sys.exit()

##############################
# Identify input files, when "-I" option used

if '-I' in opts:
    filenames = glob.glob(opts['-I'])
else:
    filenames = args

# Check if filenames are being found 
# (comment out after checking)
print('INPUT-FILES:', filenames, file=sys.stderr)

##############################
# STOPLIST option

stops = set()
if '-s' in opts:
    with open(opts['-s'], 'r') as stop_fs:
        for line in stop_fs :
            stops.add(line.strip())
            
##############################
# Stemming function

stemmer = PorterStemmer().stem

def stem_word(word):
    return stemmer(word)


##############################
# COUNT-WORDS function. 
# Takes 2 inputs: 1= FILE-NAME, 2= stoplist
# Returns a dictionary of word counts

def count_words(filename, stops):
    counts = {}
    infile = open(filename)
    file_list = list(infile)
    for i in range(len(file_list)):
        k = file_list[i].split()
        for j in range(len(k)):
            if k[j] != '\n'and k[j] not in counts.keys() and k[j] not in stops:
                counts[k[j]] = 1
            elif k[j] in counts.keys():
                counts[k[j]] = counts[k[j]] + 1
    return counts
        

##############################
# Compute counts for individual documents

docs = []

for infile in filenames:
    docs.append(count_words(infile, stops))

##############################
# Compute similarity score for document pair
# Inputs are dictionaries of counts for each doc
# Returns similarity score

def jaccard(doc1, doc2):
    list1 = list(doc1.keys())
    list2 = list(doc2.keys())

    score = min(len(list1),len(list2))/max(len(list1),len(list2))
    return score
    

##############################
# Compute scores for all document pairs

results = {}
for i in range(len(docs)-1):
    for j in range(i+1, len(docs)):        
        pair_name = '%s <> %s' % (filenames[i], filenames[j])
        results[pair_name] = jaccard(docs[i], docs[j])

##############################
# Sort, and print top N results

top_N = 20

d = sorted(results.items(),key= lambda a: a[1],reverse=True)

pairs = []
for i in range(top_N):
    pairs.append(d[i][0]) 
# Replace with code to sort results based on scores.
# Have only results for highest "top_N" scores printed.

# Printing
c = 0
for pair in pairs:
    c += 1
    print('[%d] %s = %.3f' % (c, pair, results[pair]), file=sys.stdout)

##############################

