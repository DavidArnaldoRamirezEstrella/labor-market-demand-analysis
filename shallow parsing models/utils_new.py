import nltk
import os,sys
import pdb, ipdb
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from nltk.corpus.reader.util import concat

from sequences.label_dictionary import *
from sequences.sequence import *
from sequences.sequence_list import *
from os.path import dirname
import numpy as np

from metrics import *
import re, string, unicodedata

from multiprocessing import Pool
from functools import partial
from datetime import datetime


import sequences.confusion_matrix as cm
import matplotlib.pyplot as plt
from nltk.stem.snowball import SpanishStemmer
from nltk.stem import SnowballStemmer
stemmer = SpanishStemmer()
eng_stemmer = SnowballStemmer("english")

## Directorie where the data files are located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
careers_dir = os.path.join(BASE_DIR,"careers tagged/")
random_dir = os.path.join(BASE_DIR,"random/")

START = '_START_'
END = '_END_'
START_TAG = '<START>'
END_TAG = '<STOP>'
RARE = "<RARE>"
BR = "**"

RARE_THR = 2
RARE_POS_THR = 5

############################################################
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
PARALLELIZE = False
NPROCESSORS = 3
############################################################

TAGS = ['REQ','AREA','JOB','CARR']
NO_LABELS = [
    START_TAG,
    END_TAG,
    BR,
    RARE,
]

NOUN = 'nc'
############################################################
#  WORD FILTERING

LONG_WORD_THR = 20
PUNCTUATION = string.punctuation + 'º¡¿ª°'

filters = [
    re.compile(r'^[a-zA-Z]$', re.UNICODE),                      # SINGLE CHAR
    re.compile(r'^[0-9]$', re.UNICODE),                         # SINGLE DIGIT
    re.compile(r'^9[0-9]{8}([%s]9[0-9]{8})*$' % PUNCTUATION, re.UNICODE),  # MOBILE PHONE NUMBER
    re.compile(r'^[0-9]{7}([%s][0-9]{7})*$' % PUNCTUATION, re.UNICODE),  # OFFICE PHONE NUMBER
    re.compile(r'^\d+$'),                           # ALL NUMBERS

    re.compile(r'^(([sS$]/\.?)|\$)[0-9a-z]+([.,]\d+)?([.,]\d+)?([.,]\d+)?([.,]\d+)?$',re.UNICODE),    # IS MONEY
    re.compile(r'\d{1,2}:\d{1,2}([-/]\d{1,2}:\d{1,2})*', re.UNICODE),      # RANGO HORARIO U HORA SIMPLE
    re.compile(r'^\d{1,2}(:\d{1,2})?[ªaApP]\.?[mM]\.?$', re.UNICODE),      # HOUR 9am, 9:00am, 9no, ...
    re.compile(r'^[ªaApP]\.?[mM]\.?$', re.UNICODE),      # HOUR am pm a.m p.m

    re.compile(r'^\d+([%s]+\d*)+$' % PUNCTUATION, re.UNICODE),  # NUMBER PUNCT NUMBER PUNCT?
    re.compile(r'\d', re.UNICODE),  # HAS DIGIT

    re.compile(r'^[%s]+$' % PUNCTUATION, re.UNICODE),  # IS PUNCTUATION
    re.compile(r'[%s]' % PUNCTUATION, re.UNICODE),  # HAS PUNCTUATION
    
    re.compile(r'^[a-zA-Z]+[%s]+[a-zA-Z]+([%s]+[a-zA-Z]+)?$' % (PUNCTUATION, PUNCTUATION), re.UNICODE),  # alpha + PUNCT + alpha
    re.compile(r'^[xX]+([%s][xX]+)?$' % PUNCTUATION, re.UNICODE),  # ONLY X's - for emails, websites
    re.compile(r'(www)|(https?)|(gmail)|(WWW)|(HTTPS?)|(GMAIL)|(^com$)|(^COM$)|(php)|(PHP)', re.UNICODE),  # urls

    re.compile(r'^.{%i,}$' % LONG_WORD_THR, re.UNICODE),        # LONG WORD >= THRESHOLD
    re.compile(r'^[a-z]+$', re.UNICODE),                        # ALL LOWERCASE
]







filter_tags = [
    'SINGLE_CHAR',
    'SINGLE_DIGIT',
    'MOB_NUMBER',
    'OFFICE_NUMBER',
    'NUMBER',
    'MONEY',
    'HOUR',
    'HOUR',
    'HOUR',    
    'NUMBER_PUNCT',
    'HAS_DIGIT',
    'PUNCTUATION',
    'HAS_PUNCTUATION',
    'ALPHA_PUNCT',
    'URL',
    'URL',
    'LONG_WORD',
    'RARE',
]
permanent_filters = list(range(2,10)) + [14,15,16]
filter_names = list(set(['<'+w+'>' for w in filter_tags] + [RARE]))

##################################################
## Leer lista no_stemming
EXTERNAL_GAZZETER_DIR = os.path.join(BASE_DIR,'external_gazetters')
no_stem_words = [word.strip('\n') for word in open(os.path.join(EXTERNAL_GAZZETER_DIR,'no_stemming'))]

##################################################

def insert(_dict, key, val = 1):
    if key not in _dict:
        _dict[key] = val
    else:
        _dict[key] += val

def assignFilterTag(word, indexes = range(len(filters))):
    wordL = ''
    if word=='¡' or word=='¿':
        wordL=word
    else:
        wordL = unicodedata.normalize('NFKD',word.lower()).encode('ascii','ignore').decode('utf8')
    for idx in indexes:
        pat = filters[idx]
        if pat.search(word):
            return '<' + filter_tags[idx] + '>'
    return RARE

def permanentFilter(word, filter_idx=permanent_filters):
    wordL = ''
    if word=='¡' or word=='¿':
        wordL=word
    else:
        wordL = unicodedata.normalize('NFKD',word.lower()).encode('ascii','ignore').decode('utf8')
    for idx in filter_idx:
        pat = filters[idx]
        if pat.search(wordL):
            return '<' + filter_tags[idx] + '>'
    return word


def stemAugmented(word):
    if word in (filter_names+[START,END]):
        return word
    
    word = word.lower()
    for pref in no_stem_words:
        if word.find(pref) == 0:
            # ascii version of word
            if pref == 'ingenier':
                # arreglar ing para q coincida con ident
                pref = 'ing'
            return unicodedata.normalize('NFKD',pref).encode('ascii','ignore').decode('utf8')
    
    stem = ''
    try:
        stem = stemmer.stem(word)
    except:
        stem = eng_stemmer.stem(word)
    return stem

##################################################

def saveObject(obj, name='model'):
    with open(name + '.pickle', 'wb') as fd:
        pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name):
    # Load tagger
    ipdb.set_trace()
    with open(obj_name + '.pickle', 'rb') as fd:
        obj = pickle.load(fd)
    return obj

#####################################################################################

class Corpus(object):
    def __init__(self):
        # Word dictionary.
        self.word_dict = LabelDictionary()
        self.pos_dict = LabelDictionary([NOUN])
        self.ne_dict = LabelDictionary()
        self.stem_vocabulary = []
        #self.ne_dict.add(BR)

        self.word_counts = {}

        # Initialize word & tag dicts
        self.word_dict.add(RARE)
        self.pos_dict.add(RARE)

        #self.stem_reference = uploadObject(os.path.join(BASE_DIR,'stem_dict'))


def simplify2ep(pos):
    if pos[0] == 'a':   # ADJETIVO
        return 'adj'
    elif pos[0] == 'd': # DETERMINANTE
        return 'det'
    elif pos[0] == 'n': # SUSTANTIVO
        return pos[:2]
    elif pos[0] == 'p': # PRONOMBRE
        return 'pron'
#    elif pos[0] == 'c':  # CONJUNCION
#        return 'conj'
    return pos

def reader(tags=TAGS,target='BIO', mode='by_doc', simplify_POS=True, path=''):
    '''
       param: tags: list of NE to consider
       param: target:
               - BIO: only I,B,O tags considered
               - TODO: complete NE tag considered
       param: mode: - by_doc: whole doc considered as one sentence
                    - by_sent: doc is split in sentences
       param: simplify_POS: apply simplifying rules to POS
       param: path: folder where to find the data
       return: list of tuples (word, pos, ne)
    '''
    data = []
    for i in range(1, 401):
        doc = []
        sent = []
        for line in open(path + str(i) + '.tsv'):
            line = line.strip('\n').strip(' ')
            x = ''
            y = ''
            pos = ''
            if len(line)>0:
                temp = line.split('\t')
                pos = simplify2ep(temp[1])
                x = temp[0]
                if len(temp) != 3:
                    ipdb.set_trace()

                if temp[-1][2:] in tags:
                    if target == 'BIO':
                        y = temp[-1][0]
                    else:
                        y = temp[-1]
                else:
                    y = 'O'
                sent.append(tuple([x,pos,y]))
            else:
                if len(sent)>0:
                    doc.append(sent)
                sent = []
        if mode == 'by_doc':
            temp = []
            for sent in doc:
                temp.extend(sent)
                temp.append((BR,BR,BR))
            temp.pop()
            doc = list(temp)
        data.append(doc)
    return data
    

def makeCorpus_wrapper(params=[]):
    raw_data,train_idx,mode,START_END_TAGS,stem_vocab,pos_vocab,filter_empty,extended_filter = params# [raw_data,train_idx,mode,START_END_TAGS,stem_vocab,pos_vocab,filter_empty,extended_filter]

    def makeCorpus(data,idx,mode='by_sent',START_END_TAGS=True, stem_vocab=[], pos_vocab=[], filter_empty=False, extended_filter = True):
        # El output sale ya filtrado para cualquier corpus
        corpus = Corpus()
        if mode=='by_doc':
            corpus.ne_dict.add(BR)
        if START_END_TAGS:
            corpus.word_dict.add(START)
            corpus.word_dict.add(END)
            corpus.ne_dict.add(START_TAG)
            corpus.ne_dict.add(END_TAG)
            corpus.pos_dict.add(START_TAG)
            corpus.pos_dict.add(END_TAG)
            
        seq_list = []
        file_ids = []
        br_pos_list = []
        name_folder = ''
        corpus.stem_vocabulary = stem_vocab

        for id in idx:
            doc = data[id]
            if id < 400:
                name_folder='car_tag_' + str(id+1)
            else:
                name_folder='random_' + str(id-400+1)

            if mode=='by_sent':
                empty_sample = True
                if filter_empty:
                    for sent in doc:
                        out = False
                        for tup in sent:
                            x,pos,y = tup
                            if y != 'O':    # si oracion no solo tiene O
                                empty_sample = False
                                out = True
                                break
                        if out:
                            break

                for i,sent in enumerate(doc):
                    sent_x = []
                    sent_y = []
                    sent_pos = []
                    #empty_sample = True
                    if START_END_TAGS:
                        sent_x   = [START    , START]
                        sent_y   = [START_TAG, START_TAG]
                        sent_pos = [START_TAG, START_TAG]
                    for tup in sent:
                        x,pos,y = tup
                        if extended_filter:
                            x = permanentFilter(x)
                        stem = stemAugmented(x.lower())
                        if x not in filter_names and stem_vocab!=[] and stem not in stem_vocab:
                            x = assignFilterTag(x)
                        if pos_vocab!=[] and pos not in pos_vocab:
                            pos = NOUN
                        #if y != 'O':    # si oracion no solo tiene O
                        #    empty_sample = False
                        if x not in corpus.word_dict:
                            corpus.word_dict.add(x)
                        if y not in corpus.ne_dict:
                            corpus.ne_dict.add(y)
                        if pos not in corpus.pos_dict:
                            corpus.pos_dict.add(pos)
                        sent_x.append(x)
                        sent_y.append(y)
                        sent_pos.append(pos)
                    if START_END_TAGS:
                        sent_x.append(END)
                        sent_y.append(END_TAG)
                        sent_pos.append(END_TAG)
                    if any([not empty_sample and filter_empty,
                            not filter_empty]):
                        seq_list.append([sent_x,sent_y,sent_pos])
                        file_ids.append(name_folder)
                        br_pos_list.append(i)
            else:
                sent_x = []
                sent_y = []
                sent_pos = []
                br_positions = []
                if START_END_TAGS:
                    sent_x   = [START    , START]
                    sent_y   = [START_TAG, START_TAG]
                    sent_pos = [START_TAG, START_TAG]
                    br_positions.append(1)  # segundo START como BR
                empty_sample = True
                for i,tup in enumerate(doc):
                    x,pos,y = tup
                    if x != BR:
                        if extended_filter:
                            x = permanentFilter(x)
                        stem = stemAugmented(x.lower())
                        if x not in filter_names and stem_vocab!=[] and stem not in stem_vocab:
                            x = assignFilterTag(x)
                    else:
                        br_positions.append(i+2) # desfase por START,START
                    if pos_vocab!=[] and pos not in pos_vocab:
                        pos = NOUN
                    if y != 'O' and y != BR:    # si oracion no solo tiene O & BR
                        empty_sample = False
                    if x not in corpus.word_dict:
                        corpus.word_dict.add(x)
                    if y not in corpus.ne_dict:
                        corpus.ne_dict.add(y)
                    if pos not in corpus.pos_dict:
                        corpus.pos_dict.add(pos)
                    sent_x.append(x)
                    sent_y.append(y)
                    sent_pos.append(pos)
                if sent_x[-1] == BR:
                    sent_x.pop()
                    sent_y.pop()
                    sent_pos.pop()
                if START_END_TAGS:
                    sent_x.append(END)
                    sent_y.append(END_TAG)
                    sent_pos.append(END_TAG)
                    br_positions.append(len(sent_x)-1)
                seq_list.append([sent_x,sent_y,sent_pos])
                file_ids.append(name_folder)
                br_pos_list.append(br_positions)

        sequence_list = SequenceList(corpus.word_dict, corpus.pos_dict, corpus.ne_dict, corpus.stem_vocabulary)

        for i,(x,y,pos) in enumerate(seq_list):
            sequence_list.add_sequence(x, y, pos,file_ids[i],br_pos_list[i])
        return sequence_list
    return makeCorpus(data=raw_data, idx=train_idx, mode=mode, START_END_TAGS=START_END_TAGS,
                        stem_vocab=stem_vocab, pos_vocab=pos_vocab, filter_empty=filter_empty, extended_filter=extended_filter)

def make_word_counts(data,idx,mode='by_sent',extended_filter=True):
    stem_vocab = {}
    POS_vocab = {}
    for id in idx:
        doc = data[id]
        if mode == 'by_sent':
            for sent in doc:
                for tup in sent:
                    w = tup[0].lower()
                    if extended_filter:
                        w = permanentFilter(w)
                    if w not in filter_names:
                        stem = stemAugmented(w)
                        if stem in stem_vocab:
                            stem_vocab[stem] += 1
                        else:
                            stem_vocab[stem] = 1
                    pos = tup[1]
                    if pos in POS_vocab:
                        POS_vocab[pos] += 1
                    else:
                        POS_vocab[pos] = 1
        else:
            stem_vocab[BR] = 1
            for tup in doc:
                ww = tup[0]
                if ww == BR:
                    stem_vocab[ww] += 1
                    continue
                w = ww.lower()
                if extended_filter:
                    w = permanentFilter(w)
                if w not in filter_names:
                    stem = stemAugmented(w)
                    if stem in stem_vocab:
                        stem_vocab[stem] += 1
                    else:
                        stem_vocab[stem] = 1
                pos = tup[1]
                if pos in POS_vocab:
                    POS_vocab[pos] += 1
                else:
                    POS_vocab[pos] = 1
    res_stem = [stem for stem,freq in stem_vocab.items() if freq>=RARE_THR or stem==BR]
    res_pos = [pos for pos,freq in POS_vocab.items() if freq>=RARE_POS_THR]

    return res_stem,res_pos


from nltk import FreqDist

def custom_train_test(test_size = 0.4):
    if test_size==0:
        return range(0,800),[]
    res = []
    train_size = int(800*(1.0-test_size))
    res  = [idx for idx in range(400) if idx%5!=3 and idx%5!=4]
    test = [idx for idx in range(400) if idx not in res]

    if len(res) < train_size:       
        train_size -= len(res)
        test_size = 1.0 - train_size/400.0
        left,temp = train_test_split(range(400,800),test_size=test_size,random_state = RANDOM_STATE)
        res += left
        test += temp
        return res,test
    elif len(res)==train_size:
        return res,test+list(range(400,800))
    else:
        # no entra
        return train_test_split(range(800),test_size=test_size, random_state=RANDOM_STATE)

########################################
## TAG PROPORTION ANALYSIS
def print_tag_proportion(data,SE=True):
    temp = []
    for sent in data.seq_list:
        if SE:
            temp.extend(sent.y[2:-1])
        else:
            temp.extend(sent)
    tag_count = FreqDist(temp).most_common()
    total = sum([v for id,v in tag_count])
    for id,count in tag_count:
        tag_name = data.y_dict.get_label_name(id)
        print("   %s: %.2f(%i)" % (tag_name,100.0*count/total,count))


###############################################################################################
def parallel_function(f):
    def easy_parallize(f, sequence):
        """ assumes f takes sequence as input, easy w/ Python's scope """
        pool = Pool(processes=NPROCESSORS) # depends on available cores
        result = pool.map(f, sequence) # for i in sequence: result[i] = f(i)
        cleaned = [x for x in result if not x is []] # getting results
        pool.close() # not optimal! but easy
        pool.join()
        return cleaned
    
    return partial(easy_parallize, f)


###############################################################################################

def getData(test=0.1, val=0.1, mode='by_sent', tags=TAGS, target='BIO', START_END_TAGS=True, extended_filter=True, filter_empty=True):
    print(":: Reading parameters")
    print("Mode  : ",mode)
    print("Target: ",target)
    print("Extended filter tags: ",extended_filter)
    print("Filter empty samples: ",filter_empty)
    print("NEs: ",tags)
    print("=======================================")
    startTime = datetime.now()

    idx = range(800)
    val_size = 0
    if val+test != 0.0:
        val_size = val/(val+test)

    #train_idx,temp = train_test_split(idx ,test_size = test+val, random_state=RANDOM_STATE)
    train_idx,temp = custom_train_test(test_size = test+val)
    try:
        test_idx,val_idx = train_test_split(temp,test_size = val_size, random_state=RANDOM_STATE)
    except:
        test_idx = val_idx = []

    raw_data = reader(tags=tags, mode=mode,path=careers_dir, target=target)
    temp     = reader(tags=tags, mode=mode,path=random_dir , target=target)
    raw_data.extend(temp)

    stem_vocab,pos_vocab = make_word_counts(raw_data,train_idx,mode=mode, extended_filter=extended_filter)

    #print("Stem_Vocab saved!!!!")
    #saveObject(stem_vocab,'train_stem_vocab')

    param1 = (tags,mode,careers_dir,target)
    param2 = (tags,mode,random_dir ,target)

    ## PARALELIZANDO MAKE_CORPUS: SAPEEE!
    makeCorpus_wrapper.parallel = parallel_function(makeCorpus_wrapper)
    train_param = [raw_data,train_idx,mode,START_END_TAGS,stem_vocab,pos_vocab,filter_empty,extended_filter]
    test_param  = [raw_data,test_idx ,mode,START_END_TAGS,stem_vocab,pos_vocab,False,extended_filter]
    val_param   = [raw_data,val_idx ,mode,START_END_TAGS,stem_vocab,pos_vocab,False,extended_filter]
    
    """
    train = makeCorpus(data=raw_data, idx=train_idx, mode=mode, START_END_TAGS=START_END_TAGS,
                        stem_vocab=stem_vocab, pos_vocab=pos_vocab, filter_empty=filter_empty, extended_filter=extended_filter)
    test  = makeCorpus(data=raw_data, idx=test_idx , mode=mode, START_END_TAGS=START_END_TAGS,
                        stem_vocab=stem_vocab, pos_vocab=pos_vocab, extended_filter=extended_filter)
    val   = makeCorpus(data=raw_data, idx=val_idx  , mode=mode, START_END_TAGS=START_END_TAGS,
                        stem_vocab=stem_vocab, pos_vocab=pos_vocab, extended_filter=extended_filter)
    """
    train,test,val = makeCorpus_wrapper.parallel([train_param,test_param,val_param])

    print("Dataset analysis:")
    print(":: Training set")
    print("Size training set: ",len(train.seq_list))
    print_tag_proportion(train)
    print("---------------------------------------")
    print(":: Testing set")
    print("Size testing set: ",len(test.seq_list))
    print_tag_proportion(test)
    print("---------------------------------------")
    print(":: Validation set")
    print("Size validation set: ",len(val.seq_list))
    print_tag_proportion(val)
    print("---------------------------------------")
    print("Execution time: ",datetime.now()-startTime)
    print("=======================================")

    return train,test,val

#############################################################################################################################
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA

from scipy import sparse
from sklearn.preprocessing import normalize

def getStandart_NERC(data, feature_mapper,mode='by_sent'):
    n = 0
    BR_x_id = data.x_dict.get_label_id(BR)
    if mode=='by_doc':
        for sequence in data.seq_list:
            for pos in range(2, len(sequence.x)-1):
                x = sequence.x[pos]
                if x == BR_x_id:
                    continue
                n += 1
    else:
        n = sum([len(seq.x)-2-1 for seq in data.seq_list])
    n_features = feature_mapper.get_num_features()

    row = []
    col = []
    values = []
    Y = np.zeros(n, dtype=np.int_)

    sample = 0
    ini = True
    X_total = []
    for sequence in data.seq_list:
        mx = 0
        for pos in range(2, len(sequence.x)-1):
            x = sequence.x[pos]
            if x == BR_x_id:
                continue
            y_1 = sequence.y[pos-1]
            y_2 = sequence.y[pos-2]
            y = sequence.y[pos]
            Y[sample] = y
            features = []
            features = feature_mapper.get_features(sequence, pos, y_1, y_2,features)
            mx = max(mx,len(features))
            #ipdb.set_trace()
            """
            row = sample*np.ones(len(features))
            values = np.ones(len(features),dtype=np.bool_)
            if ini:
                X_total = sparse.csr_matrix((values,(row,features)),shape=(n,n_features),dtype=np.bool_)
                ini = False
            else:
                X_temp = sparse.csr_matrix((values,(row,features)),shape=(n,n_features),dtype=np.bool_)
                X_total = X_total + X_temp
            """

            row.extend([sample for i in range(len(features))])
            col.extend(features)
            
            sample += 1
        print("       nr_seq",mx)
    values = np.ones(len(row))
    X_total = sparse.csr_matrix( (values,(row,col)), shape=(n,n_features))
    return X_total,Y

##############################################################################################
##########          NEC UTILS
import classifiers.id_features_NEC as featNEC

class Chunk:
    def __init__(self, sequence_id, pos, length, entity):
        self.sequence_id = sequence_id
        self.pos = pos
        self.length = length
        self.entity = entity        # NOMBRE, NO ID

    def __repr__(self):
        return "<\nSequence id:%i\n" \
               "Pos inicial: %i\n" \
               "Longitud: %i\n" \
               "Entidad: %s\n>" % (self.sequence_id, self.pos, self.length, self.entity)

class ChunkSet:
    def __init__(self, dataset):
        self.dataset = dataset
        self.chunk_list = []
        self.entity_classes = LabelDictionary()

        self.chunk_data()

    def chunk_data(self):
        for seq_id in range(len(self.dataset)):
            sequence = self.dataset.seq_list[seq_id]
            pos = 0
            open = False
            n = len(sequence.x)
            ne = ''
            for (i, w) in enumerate(sequence.x):
                tag = self.dataset.y_dict.get_label_name(sequence.y[i])
                if len(tag)>1:
                    ne = tag[2:]
                else:
                    ne = tag

                if ne!= 'O' and tag != START_TAG and tag != END_TAG and tag != BR:
                    self.entity_classes.add(ne)
                prev_ne = ne

                if i>0:
                    prev_tag = self.dataset.y_dict.get_label_name(sequence.y[i-1])
                    if len(prev_tag)>1:
                        prev_ne = prev_tag[2:]
                if tag.find('B') == 0:
                    if open and i>0:
                        chunk = Chunk(sequence_id = seq_id, pos = pos, length = i - pos, entity = prev_ne)
                        self.chunk_list.append(chunk)
                    pos = i
                    open = True
                elif tag.find('I') != 0 and open:
                    open = False
                    chunk = Chunk(sequence_id = seq_id, pos = pos, length = i - pos, entity = prev_ne)
                    self.chunk_list.append(chunk)
            if open:
                chunk = Chunk(sequence_id = seq_id, pos = pos, length = n - pos, entity = ne)
                self.chunk_list.append(chunk)


def getStandart(chunks, feature_mapper, mode = 'BINARY'):
    X = np.zeros((len(chunks.chunk_list),len(feature_mapper.feature_dict)))
    Y = []
    if mode=='BINARY':
        Y = np.zeros(len(chunks.chunk_list), dtype=np.int_)
    else:
        Y = np.zeros( (len(chunks.chunk_list),len(chunks.entity_classes)) )      # CUATRO ENTIDADES
    for i,chunk in enumerate(chunks.chunk_list):
        features = feature_mapper.get_features(chunk, chunks.dataset)
        X[i,features] = 1
        ne_id = chunks.entity_classes.get_label_id(chunk.entity)
        if mode == 'BINARY':
            Y[i] = ne_id
        else:
            Y[i,ne_id] = 1
    return X,Y


def getData_NEC(test=0.2, val=0.2, mode='by_sent',target='TODO'):
    print("Reading data...")
    train,test,val = getData(test=test, val=val, mode=mode,target=target)

    print("Reading chunks...")
    chunks_train = ChunkSet(dataset=train)
    chunks_test = ChunkSet(dataset=test)

    print("Building features...")
    idf = featNEC.IDFeatures(dataset = train, chunkset = chunks_train)
    idf.build_features()

    ###############################################################################
    print("Standarizing dataset...")
    X_train,Y_train = getStandart(chunks_train, idf)
    X_test,Y_test = getStandart(chunks_test, idf)

    # sparse representation and normalize
    X_train = sparse.csr_matrix(X_train)
    X_train = normalize(X_train, copy = False)

    X_test = sparse.csr_matrix(X_test)
    X_test = normalize(X_test, copy = False)

    return X_train,Y_train,X_test,Y_test, chunks_train


############################################################

def saveObject(obj, name='model'):
    with open(name + '.pickle', 'wb') as fd:
        pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name):
    # Load tagger
    with open(obj_name + '.pickle', 'rb') as fd:
        obj = pickle.load(fd)
    return obj



def join_data_tags(data_list):
    res = []
    for sent in data_list:
        #res.extend(sent.y[2:-1])
        res.extend(sent.y)
    return res

def join_data_tags_bio(data_list):
    res = []
    mn = min([
            data_list[0].sequence_list.y_dict.get_label_id("B"),
            data_list[0].sequence_list.y_dict.get_label_id("I"),
            data_list[0].sequence_list.y_dict.get_label_id("O"),
        ])
    for sent in data_list:
        res.extend( [id-mn for id in sent.y if sent.sequence_list.y_dict.get_label_name(id) in ["B",'I','O']] )
    return res

############################################################
is_punct = re.compile(r'^[%s]+$' % PUNCTUATION, re.UNICODE)  # IS PUNCTUATION

##################################################

def filterTokens(tokens, word_dict):
    res = []
    for sent in tokens:
        new_sent = []
        for word in sent:
            if word.lower() in word_dict:
                new_sent.append(word)
            else:
                new_sent.append(assignFilterTag(word))
        res.append(new_sent)

    return res



def make_term_frequency(tokens,word_dict,stem=False):
    collapsed = []
    stopwords = getStopwords(stem=stem)
    for sent in tokens:
        if stem:
            sent = [stemAugmented(word.lower()) for word in sent if word.lower() not in stopwords and 
                                                     not is_punct.match(word) and
                                                     word not in filter_names and
                                                     word.lower() in word_dict]
        else:
            sent = [word.lower() for word in sent if word.lower() not in stopwords and 
                                                     not is_punct.match(word) and
                                                     word not in filter_names and
                                                     word.lower() in word_dict]
        collapsed.extend(sent)
    ff = FreqDist(collapsed).most_common()
    #res = [(word_id[word.lower()],freq) for word,freq in ff if word.lower() in word_id]
    res = dict(ff)
    return res

def getStopwords(stem=False):
    res = set()
    stopwords_dir = os.path.join(BASE_DIR,'stopwords')
    for root, dirs, filenames in os.walk(stopwords_dir):
        for f in filenames:
            if f[-1]!='~':
                for line in open(os.path.join(stopwords_dir,f)):
                    line = line.strip('\n').lower()
                    if line!='':
                        if stem:
                            res.add(stemAugmented(line))
                        else:
                            res.add(line)
    return res


def readDoc(doc_file):
    res = []
    try:
        for line in open(doc_file):
            line = line.strip('\n').replace('<br>','')
            if line!='':
                sent = line.split(' ')
                res.append(sent)
    except:
        pass
    return res


def make_freqdist(doc_names, docs_dir,word_dict_filtered,stem=False,THR=10):
    collapsed = []
    stopwords = getStopwords(stem=stem)

    for name in doc_names:
        tokens = readDoc(os.path.join(docs_dir,name))
        if len(tokens)==0:
            continue
        filtered_tokens = filterTokens(tokens,word_dict_filtered)
        for sent in filtered_tokens:
            if stem:
                sent = [word.lower() for word in sent if stemAugmented(word.lower()) not in stopwords and 
                                                        not is_punct.match(word) and
                                                        word not in filter_names]
            else:
                sent = [word.lower() for word in sent if word.lower() not in stopwords and 
                                                     not is_punct.match(word) and
                                                     word not in filter_names]
            collapsed.extend(sent)
    res = FreqDist(collapsed).most_common()

    res = [(w,c) for w,c in res if c>=THR]

    return dict(res)
