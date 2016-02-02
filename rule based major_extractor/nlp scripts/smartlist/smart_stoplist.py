import os
import sys
import nltk
from utilities import *

keyword_path  = os.path.join(BASE_DIR,'engineering/keywords')
doctotal_path = os.path.join(BASE_DIR,'engineering/engineering')

V = {}
S = []
adjacency_frequency = {}
keyword_frecuency = {}


def getAdjFreq(keywords_list,Sentences,Vocabulary):
    """ input: lista de keywords | lista de oraciones procesadas | Vocabulario
        ouput: dic de adjacent keys
        desc: arma adj_freq dictionary usando las oraciones. FTW!
    """
    AF = {}
    for sentence in Sentences:
        L = len(sentence)
        founded = []
        for keyword in keywords_list:
            Lk = len(keyword)
            flag = True
            idx = -1
            for i in range(2,L-Lk+1):
                idx = i
                flag = True
                for k in keyword:
                    if sentence[i] != k:
                        flag = False
                        break
            if flag and idx != -1:
                founded.append((keyword,idx-Lk+1))
       
        for key in founded:
            Lk = len(key[0])
            idx = key[1]
            insert(AF,sentence[idx-1])
            if idx + Lk < len(sentence):
                insert(AF,sentence[idx+Lk])
    
    for k,v in Vocabulary.iteritems():
        if k not in AF:
            AF[k] = 0
    return AF
            

def getKeyFreq(keywords,Vocabulary):
    """ input: keywords list with format: sentence list | corpus vocabulary 
        output: keyword freq dictionary
        descp: armar un dict de frecuencias de solo las palabras en keywords
    """
    KF = {}
    for key in keywords:
        for word in key:
            insert(KF,word)
    
    for k,v in Vocabulary.iteritems():
        if k not in keyword_frecuency:
            KF[k] = 0
    return KF


def getSmartStopList(AF,KF,V):
    """ input: adjacency freq {}| keyword freq{} | vocabulary{}
        ouput: smartStopwords list[]
        desc: descarta stopwords generales. Devuelve sw + frecuentes mayores que un umbral
    """
    smL = []
    stopwords = getStopWords(ignore_smart=True)

    # Condicion de SmartList | no considera stopwords ya existentes
    cleanList = [(v,key) for (key,v) in V.iteritems() if (all([key not in stopwords,
                                                              AF[key] >= KF[key],
                                                              key not in TAGS,
                                                              ])) ]
    cleanList.sort(reverse=True)

    # porcentaje inferior de rango de frecuencia a ignorar!
    freq_percentage = 0.1
    
    # umbral para frecuencia
    frequency_threshold = round((cleanList[0][0]-cleanList[-1][0])*freq_percentage)
    for item in cleanList:
        v = item[0]
        k = item[1]
        if v >= frequency_threshold:
            smL.append(k)
    return smL


if __name__== "__main__":
    keywords = [strip_encode(str(key)).split(' ') for key in open(keyword_path).read().split('\n') if len(key)>0]
    corpus = open(doctotal_path).read()
    
    (V,S) = getVocabSentences(corpus, option="all",stem_flag=False, filters=[NUM_TAG])
    
    adjacency_frequency = getAdjFreq(keywords,S,V)
    keyword_frecuency   = getKeyFreq(keywords,V)
    
    SML = getSmartStopList(adjacency_frequency,keyword_frecuency,V)
    
    open(smart_stoplist,'w').write('\n'.join(SML))
    