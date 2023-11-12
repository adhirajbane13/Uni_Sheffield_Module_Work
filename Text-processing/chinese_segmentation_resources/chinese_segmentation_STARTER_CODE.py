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



################################################################
# FUNCTION TO PROCESS ONE SENTENCE
# Sentence provided as a string. Result returned as a list of strings 

def segment(sent, wordset):   # dummy definition provided, so can 
    return list(sent)         # write code for main loop first. 



################################################################
# MAIN LOOP
# Read each line from input file, segment, and print to output file



################################################################

