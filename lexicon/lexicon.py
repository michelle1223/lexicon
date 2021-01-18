import math
import pandas as pd
from collections import Counter
from tqdm import tqdm

# import data
train_name = input("Enter the data file name for train data: ")
test_name = input("Enter the data file name for test data: ")
encType = input("Enter the encoding type of train & test files: ")
train = pd.read_csv(train_name, encoding=encType)
test = pd.read_csv(test_name, encoding=encType)

# WARNING: train data must have the following columns; ['id', 'date', 'media', 'title', 'comment_cnt', 'like_cnt', 'similarity_score',
# 'body', 'sentence', 'tagging', 'body_preproc', 'sent_preproc', 'posneg_traintest', 'prospect_traintest']
# test data columns: 

# make positive word dictionary
tag1 = train[train['tagging']==1]['body_preproc']
posList = [t.split() for t in tag1]
posList = [word for sublist in posList for word in sublist]
posDict = Counter(posList)

# make negative word dictionary
tag2 = train[train['tagging']==2]['body_preproc']
negList = [t.split() for t in tag2]
negList = [word for sublist in negList for word in sublist]
negDict = Counter(negList)

# make full dictionary
fullDict = {}
logN = {}
allKeys = posDict.keys() | negDict.keys()
for k in allKeys:
    p = posDict[k]
    n = negDict[k]
    logN[k] = math.log(p+n)  #문장 가중치 계산 시 쓰일 각 단어 전체 횟수 로그 값
    fullDict[k] = (p-n)/(p+n)  #전체 단어 사전 (단어의 긍부정 값) 계산

# sentence weight function
def sentWeight(Sentence):
    sentence = Sentence.split()
    weight = 0
    for word in sentence:
        try:
            logW = math.log(dict(posDict)[word]+dict(negDict)[word])  #posDict, negDict는 딕셔너리 중에서도 counter여서 없는 단어를 []하면 KeyError가 안 뜨고 그냥 0이 출력됨.
            #이때 word가 전체 단어 사전에 없던 단어여서 []했을 때 0+0이 나오면 log값이 계산 안됨. 그러므로 일반 dict로 바꿔서 없는 단어면 KeyError가 뜰 수 있도록 함.
            weight += fullDict[word]*(logW/sum(logN.values()))
        except KeyError:
            continue
    return weight

# make result dataframe with weekly lexicon score column
test['score'] = test['sent_preproc'].apply(sentWeight)
grouped = test.groupby(['week', 'news_id']).mean()
weekly = grouped.groupby('week').mean()
weekly.to_csv('lexicon_score.csv')
