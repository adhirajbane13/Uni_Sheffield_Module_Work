from lab7_countwords import countWords
from lab7_top20 import printTop20
from lab7_stopwords import readStopWords
from lab7_similarity import similarity

#Task1
stop_words = readStopWords('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab07_code\\stopwords.txt')
#print(stop_words)

dict_count = countWords('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab07_code\\mobydick.txt',stop_words)
print(dict_count)

printTop20(dict_count)

dict_count1 = countWords('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab07_code\\george01.txt',stop_words)
dict_count2 = countWords('C:\\UoS_DA_Lab\\My_Work\\Text-processing\\lab_code_data(1)\\lab_code_data\\lab07_code\\george03.txt',stop_words)

sim_score = similarity(dict_count1,dict_count2)
print(sim_score)