import os,sys
import json
import unicodedata
import re
import numpy as np

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_utils)

from utils_new import *

#################################################################################################################
IDENTIFIER_STEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'identifiers')

# docname : {docname : true name}
nameByFile = json.loads(open('hierarchy/ident_names.json','r').read())

## normalizando ingenieria
norm_ing = re.compile(r'\bing[a-z.]*\b')

#################################################################################################################
def makeSequence_doc(doc, _id='',START_END_TAGS=True):
    corpus = Corpus()
    if START_END_TAGS:
        corpus.word_dict.add(START)
        corpus.word_dict.add(END)
        corpus.word_dict.add(BR)
        corpus.ne_dict.add(START_TAG)
        corpus.ne_dict.add(END_TAG)
        corpus.ne_dict.add(BR)
        corpus.pos_dict.add(START_TAG)
        corpus.pos_dict.add(END_TAG)
        corpus.pos_dict.add(BR)
    corpus.ne_dict.add('B')
    corpus.ne_dict.add('I')
    corpus.ne_dict.add('O')
    corpus.ne_dict.add(BR)
    stem_vocab = uploadObject('train_stem_vocab')
    corpus.stem_vocabulary = stem_vocab

    sent_x = []
    sent_y = []
    sent_pos = []
    br_positions = []
    if START_END_TAGS:
        sent_x   = [START    , START]
        sent_y   = [START_TAG, START_TAG]
        sent_pos = [START_TAG, START_TAG]
        br_positions.append(1)  # segundo START como BR
    k = 2
    for sentence in doc:
        k+=len(sentence)
        for x in sentence:
            x = permanentFilter(x)
            stem = stemAugmented(x.lower())
            if x not in filter_names and stem_vocab!=[] and stem not in stem_vocab:
                x = assignFilterTag(x)
            if x not in corpus.word_dict:
                corpus.word_dict.add(x)
            sent_x.append(x)
            sent_y.append('O')
            sent_pos.append(0)
        sent_x.append(BR)
        sent_y.append(BR)
        sent_pos.append(BR)
        br_positions.append(k)
        k+=1

    if sent_x[-1] == BR:
        sent_x.pop()
        sent_y.pop()
        sent_pos.pop()
    if START_END_TAGS:
        sent_x.append(END)
        sent_y.append(END_TAG)
        sent_pos.append(END_TAG)

    sequence_list = SequenceList(corpus.word_dict, corpus.pos_dict, corpus.ne_dict, corpus.stem_vocabulary)
    sequence_list.add_sequence(sent_x, sent_y, sent_pos,_id,br_positions)
    
    return sequence_list.seq_list[0]


def makeSequence(sentence, _id='',START_END_TAGS=True):
    corpus = Corpus()
    if START_END_TAGS:
        corpus.word_dict.add(START)
        corpus.word_dict.add(END)
        corpus.ne_dict.add(START_TAG)
        corpus.ne_dict.add(END_TAG)
        corpus.pos_dict.add(START_TAG)
        corpus.pos_dict.add(END_TAG)
    corpus.ne_dict.add('B')
    corpus.ne_dict.add('I')
    corpus.ne_dict.add('O')
    
    stem_vocab = uploadObject('train_stem_vocab')
    corpus.stem_vocabulary = stem_vocab

    sent_x = []
    sent_y = []
    sent_pos = []

    if START_END_TAGS:
        sent_x = [START    , START]
        sent_y = [START_TAG, START_TAG]
        sent_pos = [START_TAG, START_TAG]
    for token in sentence:
        x = permanentFilter(token)
        stem = stemAugmented(x.lower())
        if x not in filter_names and stem_vocab!=[] and stem not in stem_vocab:
            x = assignFilterTag(x)
        if x not in corpus.word_dict:
            corpus.word_dict.add(x)
        sent_x.append(x)
        sent_pos.append(0)
        sent_y.append(0)
    if START_END_TAGS:
        sent_x.append(END)
        sent_y.append(END_TAG)
        sent_pos.append(END_TAG)

    sequence_list = SequenceList(corpus.word_dict, corpus.pos_dict, corpus.ne_dict, corpus.stem_vocabulary)
    sequence_list.add_sequence(sent_x, sent_y, sent_pos,_id)
    
    return sequence_list.seq_list[0]



def getNameEntities(sequence):
    chunks = ChunkSet(sequence.sequence_list)
    nes = []
    for chunk in chunks.chunk_list:
        ini = chunk.pos
        fin = ini + chunk.length
        ne = ' '.join([sequence.sequence_list.x_dict.get_label_name(xi) for xi in sequence.x[ini:fin]])
        nes.append(ne)
    return nes


#################################################################################################################
def getDocnameByCareer():
    """ :return diccionario de nombres de documentos mapeados por nombre de carreras
    """
    docnameByCareer = {}

    for root, dirs, filenames in os.walk(IDENTIFIER_STEM_DIR):
        for f in filenames:
            if f[-1]!='~':
                for line in open(os.path.join(IDENTIFIER_STEM_DIR, f),'r'):
                    line = line.strip('\n')
                    if line not in docnameByCareer:
                        docnameByCareer[line] = set()
                    docnameByCareer[line].add(f)
    return docnameByCareer

fileByIdent = getDocnameByCareer()

def discretizeCareers(name_entities):
    if name_entities==[]:
        return ['otros']
    carreras = [unicodedata.normalize('NFKD', ne.lower()).encode('ascii','ignore').decode('utf-8') for ne in name_entities]

    ing_post = False
    temp = []
    for car in carreras:
        new_car = norm_ing.sub('ing',car)
        if car != new_car:
            ing_post = True
        new_car = ' '.join([stemAugmented(word) for word in new_car.split(' ')])
        temp.append(new_car)
    carreras = list(temp)
    
    """
    print(name_entities)
    print(carreras)
    ipdb.set_trace()
    """

    found_areas = []
    ING_STEM  = 'ing'
    # buscar en identificadores | probar agregando ingenieria +
    for option in carreras:
        # variaciones nombres de ingenierias
        op1 = ING_STEM + " " + option
        op2 = ING_STEM + " de " + option
        op3 = ING_STEM + " en " + option
        op = [option,op1,op2,op3]

        for car in op:
            if car in fileByIdent.keys():
                # CASO ESPECIAL : QUIMICA | ING QUIMICA
                if any([car=='ing quimic',
                        car=='ing en quimic',
                        car=='ing de quimic',
                        car=='quimic']):                    # detect case
                    if ing_post:                            # es post de ingenieria?
                        found_areas.append('quimica')       # ing quimica
                    else:
                        found_areas.append('quimico')       # quimico puro
                elif car in fileByIdent:                                       # caso general
                    found_areas.extend(list(fileByIdent[car]))
    if found_areas == []:
        found_areas.append('otros')
    return list(set(found_areas))

def load_doc_topic_matrix(topics = 10,folder='jobs_1000',n_docs=1000):
    load_path = '/home/ronald/clustering/' +folder+ '_'+ str(topics)
    docs_topics = np.zeros((n_docs,topics))
    k = 0
    for line in open(os.path.join(load_path,'final.gamma'),'r'):
        line = line.strip('\n')
        if line!='':
            topics = np.array([float(a) for a in line.split(' ')])
            docs_topics[k] = topics
            k += 1
    return docs_topics


def insert(_dict, key, val = 1):
    if key not in _dict:
        _dict[key] = val
    else:
        _dict[key] += val

stopwords = getStopwords()

def mallet_text(doc):
    res = []
    for sent in doc:
        res_sent = [unicodedata.normalize('NFKD', word.lower()).encode('ascii','ignore').decode('utf-8') \
                        for word in sent if stemAugmented(word.lower()) not in stopwords and 
                                                        not is_punct.match(word) and
                                                        word not in filter_names]
        if len(res_sent)!=0:
            res.append(' '.join(res_sent))
    return '\n'.join(res)