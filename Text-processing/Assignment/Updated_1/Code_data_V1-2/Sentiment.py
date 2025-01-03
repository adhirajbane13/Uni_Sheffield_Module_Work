#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=0

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

    with open('positive-words.txt', 'r', encoding="ISO-8859-1") as posDictionary:
        posWordList = []
        for line in posDictionary:
            if not line.startswith(';'):
                posWordList.extend(re.findall(r"[a-z\-]+", line))
    posWordList.remove('a')

    with open('negative-words.txt', 'r', encoding="ISO-8859-1") as negDictionary:
        negWordList = []
        for line in negDictionary:
            if not line.startswith(';'):
                negWordList.extend(re.findall(r"[a-z\-]+", line))

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets:
    #We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

  #create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    pNeg=1-pPos

    #These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: #calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
    confusion_matrix(correctpos,totalpospred,correctneg,totalnegpred,dataName)
 
 
# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
def confusion_matrix(cp,tppred,cn,tnpred,dataname):
    truepos = cp
    trueneg = cn
    falsepos = tppred - cp
    falseneg = tnpred - cn
    accuracy = (truepos +trueneg)/(truepos+falsepos+trueneg+falseneg)
    print('Performance Metrics for',dataname)
    print(f'Accuracy = {accuracy*100:0.2f}%')
    precision_pos = truepos/(truepos+falsepos)
    precision_neg = trueneg/(trueneg+falseneg)
    print(f'Precision for positive = {precision_pos*100:0.2f}%',f'\nPrecision for negative = {precision_neg*100:0.2f}%')
    recall_pos = truepos/(truepos+falseneg)
    recall_neg = trueneg/(trueneg+falsepos)
    print(f'Recall for positive = {recall_pos*100:0.2f}%',f'\nRecall for negative = {recall_neg*100:0.2f}&')
    f1_score_pos = (2*precision_pos*recall_pos)/(precision_pos+recall_pos)
    f1_score_neg = (2*precision_neg*recall_neg)/(precision_neg+recall_neg)
    print(f'F1-score for positive = {f1_score_pos*100:0.2f}%',f'\nF1-score for negative = {f1_score_neg*100:0.2f}%')
    print('\n') 




# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %score + sentence)
    confusion_matrix(correctpos,totalpospred,correctneg,totalnegpred,dataName)
 
    
# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
 



#Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)
    headcount = 0
    tailcount = 0
    for i in head:
        if i in sentimentDictionary:
            headcount+=1
    for i in tail:
        if i in sentimentDictionary:
            tailcount+=1
    
    print('Number of words in the Sentiment Dictionary:',(headcount+tailcount))

#Improved Rule-Based System
def rulebs(sentencesTest, dataName, sentimentDictionary, threshold):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    neg_words = ['NOT','not','Not','never','no']
    intensifier_dict = {'very':1,'extremely':2,'definitely':2,'utterly':2,'absolutely':2,
                        'incredibly':1}
    diminisher_list = ["somewhat", "barely", "rarely","marginally","fairly","partially"]
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
                score += sentimentDictionary[word]
                left_nh = Words[0:Words.index(word)]
                right_nh = Words[Words.index(word)+1:len(Words)]
                for neg_word in neg_words:
                    if neg_word in left_nh:
                        score += -1*(score-1)
                if (word.isupper()):
                    if sentimentDictionary[word] == 1:
                        score += 1
                    else:
                        score -= 1
                if '!!!' in right_nh or '!!' in right_nh or '!' in right_nh:
                    if sentimentDictionary[word] == 1:
                        score += 2
                    else:
                        score -= 2
                for ins_word in intensifier_dict:
                    if ins_word in left_nh or ins_word in right_nh:
                        if sentimentDictionary[word] == 1:
                            score = score + intensifier_dict[ins_word]
                        else:
                            score = score - intensifier_dict[ins_word]
                for dim_word in diminisher_list:
                    if dim_word in left_nh or dim_word in right_nh:
                        if sentimentDictionary[word] == 1:
                            score = score - 1
                        else:
                            score = score + 1
                
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
    confusion_matrix(correctpos,totalpospred,correctneg,totalnegpred,dataName)

#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

#run naive bayes classifier on datasets
print ("\x1B[4m" + "Naive Bayes" + "\x1B[0m" +":")
testBayes(sentencesTrain,  "Film(Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films(Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesNokia, "Nokia(All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)



#run sentiment dictionary based classifier on datasets
print("\x1B[4m" + 'Dictionary based Classifier' + "\x1B[0m" +":")
testDictionary(sentencesTrain,  "Film(Train Data, Dictionary-Based)\t", sentimentDictionary, 1)
testDictionary(sentencesTest,  "Films(Test Data, Dictionary-Based)\t",  sentimentDictionary, 1)
testDictionary(sentencesNokia, "Nokia(All Data, Dictionary-Based)\t",  sentimentDictionary, 1)

#run rule based classifier
print("\x1B[4m" + 'Improved Rule-based system' + "\x1B[0m" +":")
rulebs(sentencesTrain,  "Films(Train Data, Rule-Based)\t", sentimentDictionary, 1)
rulebs(sentencesTest,  "Films(Test Data, Rule-Based)\t",  sentimentDictionary, 1)
rulebs(sentencesNokia, "Nokia(All Data, Rule-Based)\t",  sentimentDictionary, 1)

# print most useful words
mostUseful(pWordPos, pWordNeg, pWord, 100)



