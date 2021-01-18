import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 
n = WordNetLemmatizer()
s = PorterStemmer()  # 초기 설정: stopwords, lemmatizer, stemmer 설정 필요!

def cleanText(readData):
    #텍스트에 포함되어 있는 특수 문자 제거 (단, $, % 제외)
    text = re.sub('[-=+,#/\?:^.@*\"※~&ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', str(readData))
    #길이 1~2인 단어 제거 (숫자/특수문자 제외)
    shortword = re.compile(r'\W*\b[a-zA-Z]{1,2}\b')
    cleaned_text = shortword.sub('', text)
    return cleaned_text

def listToString(l):  
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(l))

def textPreproc(text):
    temp = cleanText(text)
    words = word_tokenize(temp)
    result = []
    for w in words: 
        if w not in stop_words: 
            result.append(w) 
    lresult = [n.lemmatize(w) for w in result]
    sresult = [s.stem(w) for w in lresult]
    final = listToString(sresult)
    return final
