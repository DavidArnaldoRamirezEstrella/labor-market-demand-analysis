# -*- coding: utf-8 -*-
__author__ = 'ronald'

import os, sys
import time
import re, string
import pymongo
from pymongo import MongoClient
import nltk

from nltk import UnigramTagger, BigramTagger, TrigramTagger
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.probability import *
import pickle
import ancora
import pdb, ipdb

EXT_CORPUS_PATH = '../../nlp/'

START = '_START_'
END = '_END_'
START_TAG = '<START>'
END_TAG = '<STOP>'

BR = '**'
RARE = "<RARE>"

NO_LABELS = [
    START_TAG,
    END_TAG,
    BR,
    RARE,
]

############################################################################################

GAZETTERS_DIR = 'external_gazetters/'

############################################################################################
vinetas = ['?',
           '◦',  # agregar aki + viñetas q sean caracter especial en regex
           '+',
           '.',
           '~',
           '*',  # limite para viñetas regex-special
           '•',
           'º',
           '»',
           '·',
           '°',
           '¬',
           'ª',
           '¨',
           '§',
           'ð',
           '→',
           '×',
           '-',  # SIEMPRE AL FINAL -- NO MOVER
]


thr = vinetas.index('*')
norm_vineta = []
for (i, vin) in enumerate(vinetas):
    if i > thr:
        norm_vineta.append(r'\n\s*%s+' % vin)
    else:
        norm_vineta.append(r'\n\s*\%s+' % vin)

# norm_vineta.append(r'\n\s*[0-9]{0,2}[.)]-?')
norm_vineta.append(r'^ *\d+[.)-](?:\d[.)-]*)*')  # match con cualquier numero del formato 1.12.12.-
norm_vineta.append(r'^ *[a-zA-Z]\d*[.)-](?:[a-zA-Z]?\d*[.)-]+)*')  # match con a) A) a.1) a.2.- etc..
roman = r'(?:X(?:X(?:V(?:I(?:I?I)?)?|X(?:I(?:I?I)?)?|I(?:[VX]|I?I)?)?|V(?:I(?:I?I)?)?|I(?:[VX]|I?I)?)?|' \
        r'V(?:I(?:I?I)?)?|I(?:[VX]|I?I)?)'
norm_vineta.append(r'^ *' + roman + r'[\da-zA-Z]{0,2}[.)-](' + roman + '?[.)-]*[\da-zA-Z]{0,2}[.)-]+)*')

# REGEX CASES
# Normalizar viñetas
norm_vineta = re.compile('|'.join(norm_vineta), re.UNICODE | re.MULTILINE)
# SI O SI VIÑETA
unicode_vineta = re.compile('[•º»·°¬ª¨§~ð→×]', re.UNICODE)


META_VINETA = '*'

# curar lineas en blanco
blank_line = re.compile(r'\n(\s*[.:,;-]*\s*\n)*', re.UNICODE)

# Caso de dinero
# money = re.compile(r'(?P<total>(?P<currency>[sS$])\s*/\s*\.?\s*(?P<number>[0-9]+))')
soles = r'(?P<totalSoles>(?P<currencySoles>[sS])(?=.*[.,/])\s*/?\s*\.?\s*/?\s*(?P<numberSoles>(\d+ ?[.,]? ?)+))'
dolar = r'(?P<totalDolar>(?P<currencyDolar>[$])\s*/?\s*\.?\s*/?\s*(?P<numberDolar>(\d+ ?[.,]? ?)+))'
money = re.compile(soles + r'|' + dolar)
dinero = re.compile(r'(?P<number>(\d+ ?[.,]? ?)+)')
hora = re.compile(r'(^|\s)\d{1,2} ?: ?\d{1,2}')

# Normalizar comillas
comillas = ['"', "'", '`', '“', '”','«','»','´']   # ¨ se hace strip nomas
norm_comillas = re.compile(r'[%s]+' % ''.join(comillas), re.UNICODE)
META_COMILLA = '"'

# reducir puntuaciones duplicadas

doubled = [',', ';', ':', '_', '!', '¡', '?', '¿'] + vinetas[:vinetas.index('+')] + vinetas[vinetas.index('+') + 1:]
clean_doubled = re.compile(r'(?P<multi>[%s]+)' % (''.join(doubled)), re.UNICODE)

# CORRECION DE TILDES
tildes = [
    ('à', 'á'),
    ('è', 'é'),
    ('ì', 'í'),
    ('ò', 'ó'),
    ('ù', 'ú'),
    ('À', 'Á'),
    ('È', 'É'),
    ('Ì', 'Í'),
    ('Ò', 'Ó'),
    ('Ù', 'Ú'),

    ('úu', 'ú'),
    ('çc', 'c'),
    ('étc', 'etc'),
    ('¸', ' '),
    ('®', ''),
    ('©', '@'),

    ('ä', 'a'),
    ('ë', 'e'),
    ('ï', 'i'),
    ('ö', 'o'),
    ('ü', 'u'),
    ('Ä', 'A'),
    ('Ë', 'E'),
    ('Ï', 'I'),
    ('Ö', 'O'),
    ('Ü', 'U'),
    ('ÿ', 'y'),

    ('â', 'á'),
    ('ê', 'é'),
    ('î', 'í'),
    ('ô', 'ó'),
    ('û', 'ú'),
    ('Â', 'Á'),
    ('Ê', 'É'),
    ('Î', 'Í'),
    ('Ô', 'Ó'),
    ('Û', 'Ú'),
]

########################################################################################################################################################################################


def separate_number(match):
    for key in match.groupdict():
        if key == 'prefix':
            return match.group('prefix') + ' ' + match.group('number')
    if len(match.group('number')) == 1 and len(match.group('word')) == 1:
        return match.group('number') + match.group('word')
    return match.group('number') + ' ' + match.group('word')


def separate_suffix(match):
    return match.group('word') + ' '


def clean_space(text):
    import re

    text = re.sub('[ \r\t\f\v]+', ' ', text)
    text = re.sub('(\n+ *)+', '\n', text)
    return text


def corrections_tokenizer(text):
    # reemplazar puntuacion – -> -
    text = text.replace('–', '-')
    text = text.replace('…', '.')

    # correciones de tildes
    for wrong, good in tildes:
        text = text.replace(wrong, good)

    ##########################################
    pattern3 = re.compile(r'\b(?P<first>[0-9]{1,2})y(?P<second>[0-9]{1,2})\b')  # 1y30, 08y30
    match = pattern3.search(text)
    while match:
        ntext = text[:match.start()] + match.group('first') + ':' + match.group('second') + text[match.end():]
        match = pattern3.search(ntext)
        if text == ntext:
            break
        text = ntext

    ##########################################
    pat_number = re.compile(r'(?P<number>[\d]{1,2})(?P<word>[a-zA-Z]+)')  # 6meses 1hora
    text = pat_number.sub(separate_number, text)

    text = clean_space(text)

    #ipdb.set_trace()

    ##########################################
    pattern3 = re.compile(r'\b[oó]\s*/\s*[yý]\b')  # y/0
    match = pattern3.search(text)
    while match:
        ntext = text[:match.start()] + 'y/o' + text[match.end():]
        match = pattern3.search(ntext)
        if text == ntext:
            break
        text = ntext

    ##########################################
    pattern3 = re.compile(r'\b[yeoý]\s*7\s*[opó0]\b') 
     # y/0
    match = pattern3.search(text)
    while match:
        ntext = text[:match.start()] + 'y/o' + text[match.end():]
        match = pattern3.search(ntext)
        if text == ntext:
            break
        text = ntext

    ##########################################
    pattern3 = re.compile(r'\b[yeoý]\s*/\s*[pó0]\b')  # y/0
    match = pattern3.search(text)
    while match:
        ntext = text[:match.start()] + 'y/o' + text[match.end():]
        match = pattern3.search(ntext)
        if text == ntext:
            break
        text = ntext

    ##########################################
    pattern3 = re.compile(r'(?P<first>[a-z]) ?/ (?P<second>[a-z])')  # y / o y/ o
    match = pattern3.search(text)
    while match:
        ntext = text[:match.start()] + match.group('first') + '/' + match.group('second') + text[match.end():]
        match = pattern3.search(ntext)
        if text == ntext:
            break
        text = ntext

    pattern3 = re.compile(r'(?P<first>[a-z]) / ?(?P<second>[a-z])')  # y / o y /o
    match = pattern3.search(text)
    while match:
        ntext = text[:match.start()] + match.group('first') + '/' + match.group('second') + text[match.end():]
        match = pattern3.search(ntext)
        if text == ntext:
            break
        text = ntext
    ############################################

    # pat_parenthesis = re.compile(r'^[(] ?([aA][sS]|[eE][sS]|[aA]|[oO]|[oO][sS]) ?[)]$')
    pat_parenthesis = re.compile(r'(?P<word>[a-zA-Z]{3,} ?)(?P<suffix>[(] ?[a-zA-z]{0,2} ?[)])')
    text = pat_parenthesis.sub(separate_suffix, text)
    text = clean_space(text)
    pat_slash = re.compile(r'(?P<word>[a-zA-Z]{3,} ?)(?P<suffix>[/] ?([a-zA-z]{1,2}) )')
    text = pat_slash.sub(separate_suffix, text)
    text = clean_space(text)

    ###########################################
    pat_number = re.compile(r'(?P<prefix>[nN]°)(?P<number>\d+)')  # N°123
    text = pat_number.sub(separate_number, text)
    ###########################################

    pattern3 = re.compile(r'x{3,}')  # xxxxxxxxxxxxxxxxxxxx
    match = pattern3.search(text)
    while match:
        ntext = text[:match.start()] + 'xxx' + text[match.end():]
        match = pattern3.search(ntext)
        if text == ntext:
            break
        text = ntext
    ##########################################

    text = clean_space(text)
    return text


def specialCases(words):
    ans = []
    change = False
    dolar = False
    for (i, word) in enumerate(words):
        # eliminar tilde no intencional
        word = word.strip('´')
        match = money.search(word)
        if match:  # is it currency?
            curr = match.group('currencySoles')
            if curr is None:
                curr = match.group('currencyDolar')
            if curr.lower() == 's' and '/.' not in word:
                word = word.replace('/', '/.')
                change = True
            ans.append(word)
            continue

        if word == '$':
            dolar = True
            ans.append(word)
            continue

        if dolar:
            match = dinero.search(word)
            if match is None:
                dolar = False
                ans.append(word)
                continue
            ans[-1] += word
            change = True
            dolar = False
            continue

        if '#' == word:
            if i - 1 >= 0 and len(ans) > 0:
                if ans[-1] == 'c' or ans[-1] == 'C':
                    ans[-1] += word
                    change = True
                else:
                    ans.append(word)
            continue

        if '*' in word and len(word) > 1:
            splitWord = word.split('*')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('*')
                change = True
            ans.pop()
            continue

        if '¡' in word and len(word) > 1:
            splitWord = word.split('¡')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('¡')
                change = True
            ans.pop()
            continue

        if '¿' in word and len(word) > 1:
            splitWord = word.split('¿')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('¿')
                change = True
            ans.pop()
            continue

        if ':' in word and len(word) > 1:
            splitWord = word.split(':')
            match = hora.search(word)
            if match:
                ans.append(word)
                continue
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append(':')
                change = True
            ans.pop()
            continue

        if '_' in word and len(word) > 1:
            splitWord = word.split('_')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('_')
                change = True
            ans.pop()
            continue

        if '\\' in word and len(word) > 1:
            splitWord = word.split('\\')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('\\')
                change = True
            ans.pop()
            continue

        if '\'' in word and len(word) > 1:
            splitWord = word.split('\'')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('\'')
                change = True
            ans.pop()
            continue

        if '|' in word and len(word) > 1:
            splitWord = word.split('|')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('|')
                change = True
            ans.pop()
            continue

        if '/' in word and len(word) > 1:
            if word.count('/') >= 2:
                splitWord = word.split('/')
                for sw in splitWord:
                    if len(sw) > 0:
                        ans.append(sw)
                    ans.append('/')
                    change = True
                ans.pop()
            else:
                slashPos = word.find('/')
                if len(word[:slashPos]) > 1 and len(word[slashPos + 1:]) > 1:
                    ans.extend([word[:slashPos], '/', word[slashPos + 1:]])
                    change = True
                elif len(word[:slashPos]) == 1 and len(word[slashPos + 1:]) == 1:
                    ans.append(word)
                else:
                    if word[:slashPos]:
                        ans.append(word[:slashPos])
                    ans.append('/')
                    if word[slashPos + 1:]:
                        ans.append(word[slashPos + 1:])
                    change = True
            continue

        if ',' in word and len(word) > 1:
            splitWord = word.split(',')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append(',')
                change = True
            ans.pop()
            continue

        if '-' in word and len(word) > 1:
            splitWord = word.split('-')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('-')
                change = True
            ans.pop()
            continue

        if '+' in word and len(word) > 1:
            if ('c++' in word) or ('C++' in word):
                ans.append(word)
                continue
            splitWord = word.split('+')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('+')
                change = True
            ans.pop()
            continue


        if '.' in word and len(word) > 1:
            #print('asdasd')
            if word.count('.') > 1:
                if any([word == '...',
                        word.lower() == 'a.m.',
                        word.lower() == 'p.m.',
                        ]):  # no separar ...
                    ans.append(word)
                    continue

                splitWord = word.split('.')
                for sw in splitWord:
                    if len(sw) > 0:
                        ans.append(sw)
                    ans.append('.')
                    change = True
                ans.pop()
            else:
                #print('asdasd')
                if word == 'm.' or '.net' == word:
                    ans.append(word)
                    continue

                dotPos = word.find('.')
                if len(word[:dotPos]) >= 1 and len(word[dotPos + 1:]) >= 1:
                    if word[:dotPos].isdigit() and word[dotPos + 1:].isdigit():
                        #print(word)
                        ans.append(word)
                    else:
                        ans.extend([word[:dotPos], '.', word[dotPos + 1:]])
                        #print(word[:dotPos], '.', word[dotPos + 1:])
                        change = True
                elif dotPos == len(word) - 1:
                    ans.extend([word[:-1], '.'])
                    #print(word[:-1], '.')
                    change = True
                else:
                    ans.extend(['.', word[1:]])
                    #print('.', word[1:])
                    change = True
            continue

        ans.append(word)
    return (ans, change)


def tokenizer(text):
    '''
    :param text: raw text to tokenize
    :return: list of words
    '''
    res = []
    if type(text) != str:
        text = text.decode('utf8')

    text = clean_space(text)

    # Curar lineas en blanco
    text = blank_line.sub('\n', text)

    text = list(text)
    import unicodedata
    for i in range(len(text)):
        try:
            text[i].encode('latin-1')
            if text[i] != '\n' and unicodedata.category(text[i])[0] == 'C':
                text[i] = '*'
        except UnicodeEncodeError:
            text[i] = '*'
    if text[-1] == '*':
        text.pop()
    text = "".join(text)

    text = corrections_tokenizer(text)

    # caso vinetas duplicadas
    match = clean_doubled.search(text)
    text = list(text)
    if text[-1] == '*':
        text.pop()
    text = "".join(text)
    temp = ''
    while (match):
        multis = match.group('multi')
        f = text.find(multis)
        temp += text[:f] + multis[0]
        text = text[f + len(multis):]
        match = clean_doubled.search(text)
    text = temp + text

    # caso dinero
    match = money.search(text)
    temp = ''
    while (match):
        total = match.group('totalSoles')
        if total is None:
            total = match.group('totalDolar')
        curr = match.group('currencySoles')
        if curr is None:
            curr = match.group('currencyDolar')
        number = match.group('numberSoles')
        if number is None:
            number = match.group('numberDolar')

        f = text.find(total)
        sub_text = ''
        if curr.lower() == 's':
            sub_text = 'S/'
        else:
            sub_text = curr
        sub_text += number.replace(' ', '')
        temp += text[:f] + ' ' + sub_text + ' '
        text = text[f + len(total):]
        match = money.search(text)
    text = temp + text

    text = clean_space(text)
    # normalizar viñetas
    text = norm_vineta.sub('\n' + META_VINETA, text)
    text = unicode_vineta.sub(META_VINETA, text)  # si o si es viñeta pero no tiene \n antes

    # normalizar comillas
    text = norm_comillas.sub(META_COMILLA, text)

    # sent_tokenizer : \n
    sents = [line + ' ' for line in text.split('\n') if len(line.strip(' ')) > 0]

    for line in sents:
        temp = []
        for chunk in line.split(META_COMILLA):
            temp.extend(nltk.word_tokenize(chunk))
            temp.append(META_COMILLA)
        temp.pop()
        ## casos especiales aqui
        (temp, change) = specialCases(temp)
        #print(temp)
        while change:
            (temp, change) = specialCases(temp)
        res.append(temp)
    res = [r for r in res if len(r) > 0]

    return res


########################################################################################################################################################################################
def saveModel(model, name='model'):
    with open(name + '_model.pickle', 'wb') as fd:
        pickle.dump(model, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadModel(name_model):
    # Load tagger
    with open(name_model + '.pickle', 'rb') as fd:
        tagger = pickle.load(fd)
    return tagger

########################################################################################################################################################################################

def simplifyTagset(data):
    res = []
    for doc in data:
        new_sent = []
        for meta in doc:
            word = meta[0]
            pos = meta[1].lower()
            new_pos = pos

            if pos == 'unk':
                if word.isdigit():
                    new_pos = 'z'
                else:
                    new_pos = 'np00'
            if pos[0] == 'a':  # adjetivo
                new_pos = pos[:2] + pos[3:5]
            elif pos[0] == 'd':  # determinante
                new_pos = pos[:5]
            elif pos[0] == 'n':  # sustantivo
                new_pos = pos[:4]
            elif pos[0] == 'v':  # verbo
                new_pos = pos[:3]
            elif pos[0] == 'p':  # pronombre
                new_pos = pos[:5]
                if pos[1] == 'e':  # PE -> PT
                    new_pos = list(new_pos)
                    new_pos[1] = 't'
                    new_pos = ''.join(new_pos)
            elif pos[0] == 's':  # preposicion
                new_pos = pos[:2]
            elif pos[0] == 'f':  # puntuacion
                if pos[:2] == 'fa' or pos[:2] == 'fi':
                    new_pos = 'fa'
                if (pos[:2] == 'fc' or pos[:2] == 'fl') and len(pos) > 2:
                    new_pos = 'fl'
                if pos == 'fc' or pos == 'fx':
                    new_pos == 'fc'
                if pos == 'fe' or pos[:2] == 'fr':
                    new_pos == 'fe'
            elif pos[0] == 'z':  # numerico
                if pos == 'zd':
                    new_pos = 'z'

            # dividir multi-words
            try:
                words = word.split('_')
            except:
                pdb.set_trace()

            for w in words:
                new_sent.append(tuple([w, new_pos]))
        # build new data
        res.append(new_sent)
    return res


def getDataAncora(max_docs='inf'):
    # get data
    # Data : lista de documentos
    # doc  : lista de tuples | oraciones separadas por tuple (BR,BR_LABEL)
    reader = ancora.AncoraCorpusReader(EXT_CORPUS_PATH + 'ancora-2.0/')
    docs = reader.tagged_sents(max_docs)

    data = []
    for doc in docs:
        jd = []
        doc = simplifyTagset(doc)
        for sent in doc:
            jd.extend(sent)
            jd.append(tuple([BR, BR_LABEL]))
        jd.pop()
        data.append(jd)

    return data


def getDataWikicorpus(max_docs='inf'):
    res = []
    sent = []
    cont = 0
    doc_cont = 0

    with open(EXT_CORPUS_PATH + 'wikicorpus/data_wc') as file:
        for line in file:
            if '<doc' not in line and '</doc' not in line:
                line = line.strip('\n')
                if line:
                    (multiwords, lemma, pos, num) = line.strip('\n').split(' ')
                    for word in multiwords.split('_'):
                        sent.append(tuple([word, pos.lower()]))
                elif len(sent) > 0:
                    res.append(list(sent))
                    doc_cont += 1

                    if cont % 10 == 0:
                        print("--->", cont)
                    cont += 1

                    if max_docs != 'inf' and doc_cont == max_docs:
                        break
                    sent = []
    return res


########################################################################################################################################################################################

def train_and_test(train, test, est):
    '''
    :train : training dataset
    :test : testing dataset
    :est  : NLTK Prob object
  '''
    tag_set = list(set([tag for sent in train for (word, tag) in sent]))
    symbols = list(set([word for sent in train for (word, tag) in sent]))

    # trainer HMM
    trainer = nltk.HiddenMarkovModelTrainer(tag_set, symbols)
    hmm = trainer.train_supervised(train, estimator=est)
    res = 100 * hmm.evaluate(test)

    return res

# estimadores prob
mle = lambda fd, bins: MLEProbDist(fd)
laplace = LaplaceProbDist
ele = ELEProbDist
witten = WittenBellProbDist
gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)


def lidstone(gamma):
    return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)


########################################################################################################################################################################################
##                          EXTENDED FEATURES : GAZETTERS

from nltk.stem.snowball import SpanishStemmer

span_stemmer = SpanishStemmer()


def getCareerGazetter():
    '''
    :return: return list with STEMMED careers unigrams
    '''
    careers = [span_stemmer.stem(line.strip()) for line in open(os.path.join(GAZETTERS_DIR, 'carreras'))]
    careers = [w for w in careers if len(w) > 3]
    careers.extend(['mina', 'mecan', 'moda', 'ing'])
    careers = list(set(careers))

    return careers


def getPlacesGazetter():
    return list(set([line.strip() for line in open(os.path.join(GAZETTERS_DIR, 'lugares'))]))


########################################################################################################################################################################################


import codecs
import gzip
from sequences.label_dictionary import *
from sequences.sequence import *
from sequences.sequence_list import *
from os.path import dirname
import numpy as np
from sklearn.cross_validation import train_test_split

## Directorie where the data files are located.
careers_dir = "careers tagged/"
random_dir = "random/"

WINDOW = 3
TAGS_PROY = ['AREA', 'REQ', 'JOB', 'CARR']

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class JobDBCorpus(object):
    def __init__(self):
        # Word dictionary.
        self.word_dict = LabelDictionary()
        self.pos_dict = LabelDictionary(['nc'])
        self.ne_dict = LabelDictionary()
        self.ne_dict.add(BR)

        # Initialize sequence list.
        self.sequence_list = SequenceList(self.word_dict, self.pos_dict, self.ne_dict)

        # Initialize word & tag dicts
        self.word_dict.add(START)
        self.word_dict.add(END)
        self.ne_dict.add(START_TAG)
        self.ne_dict.add(END_TAG)
        self.pos_dict.add(START_TAG)
        self.pos_dict.add(END_TAG)

    def read_sequence_list(self, target='BIO', START_END_TAGS=True, entities=TAGS_PROY):
        '''
        :param target: BIO : IBO tagging, Y = B,I,O
                       NE : Y = NE names
        :return: list of sentences
        '''
        seq_list = []
        file_ids = []
        for i in range(1, 401):
            sent_x = []
            sent_y = []
            sent_pos = []
            if START_END_TAGS:
                sent_x = [START, START]
                sent_y = [START_TAG, START_TAG]
                sent_pos = [START_TAG, START_TAG]
            for line in open(careers_dir + str(i) + '.tsv'):
                line = line.strip('\n')
                x = ''
                y = ''
                pos = ''
                if len(line) > 0:
                    temp = line.split('\t')
                    pos = temp[1]
                    x = temp[0]
                    if temp[-1][2:] in entities:
                        if target == 'BIO':
                            y = temp[-1][0]
                        else:
                            y = temp[-1]  # temp[-1][2:]
                    else:
                        y = 'O'
                else:
                    x, pos, y = (BR, BR, BR)

                if x not in self.word_dict:
                    self.word_dict.add(x)
                if y not in self.ne_dict:
                    self.ne_dict.add(y)

                if pos not in self.pos_dict:
                    self.pos_dict.add(pos)
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

            seq_list.append([sent_x, sent_y, sent_pos])
            file_ids.append("car_tag_" + str(i))

        for i in range(1, 401):
            sent_x = []
            sent_y = []
            sent_pos = []
            if START_END_TAGS:
                sent_x = [START, START]
                sent_y = [START_TAG, START_TAG]
                sent_pos = [START_TAG, START_TAG]
            for line in open(random_dir + str(i) + '.tsv'):
                line = line.strip('\n')
                x = ''
                y = ''
                pos = ''
                if len(line) > 0:
                    temp = line.split('\t')
                    x = temp[0]
                    pos = temp[1]

                    if temp[-1][2:] in entities:
                        if target == 'BIO':
                            y = temp[-1][0]
                        else:
                            y = temp[-1]  # temp[-1][2:]
                    else:
                        y = 'O'
                else:
                    x, pos, y = (BR, BR, BR)
                if x not in self.word_dict:
                    self.word_dict.add(x)
                if y not in self.ne_dict:
                    self.ne_dict.add(y)
                if pos not in self.pos_dict:
                    self.pos_dict.add(pos)
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

            seq_list.append([sent_x, sent_y, sent_pos])
            file_ids.append("random_" + str(i))

        self.sequence_list = SequenceList(self.word_dict, self.pos_dict, self.ne_dict)

        for i, (x, y, pos) in enumerate(seq_list):
            self.sequence_list.add_sequence(x, y, pos, file_ids[i])
        return self.sequence_list


    def train_test_data(self, test_size=0.1):
        train = SequenceList(self.word_dict, self.pos_dict, self.ne_dict)
        test = SequenceList(self.word_dict, self.pos_dict, self.ne_dict)
        tn, tt = train_test_split(self.sequence_list.seq_list, test_size=test_size)
        train.seq_list = tn
        test.seq_list = tt

        return train, test


    def TTCV_data(self, test_size=0.2, cv_size=0.2):
        train = SequenceList(self.word_dict, self.pos_dict, self.ne_dict)
        test = SequenceList(self.word_dict, self.pos_dict, self.ne_dict)
        cv = SequenceList(self.word_dict, self.pos_dict, self.ne_dict)

        tn, temp = train_test_split(self.sequence_list.seq_list, test_size=test_size + cv_size,
                                    random_state=RANDOM_STATE)
        tst, cvt = train_test_split(temp, test_size=cv_size / (cv_size + test_size), random_state=RANDOM_STATE)

        train.seq_list = tn
        test.seq_list = tst
        cv.seq_list = cvt

        return train, test, cv

    ## Dumps a corpus into a file
    def save_corpus(self, name):
        with open(name + '_corpus.pickle', 'wb') as fd:
            pickle.dump(self, fd, protocol=pickle.HIGHEST_PROTOCOL)

    ## Loads a corpus from a file
    def load_corpus(self, name):
        with open(name + '_corpus.pickle', 'rb') as fd:
            loaded = pickle.load(fd)

        self.word_dict = loaded.word_dict
        self.pos_dict = loaded.pos_dict
        self.ne_dict = loaded.ne_dict
        self.sequence_list = loaded.sequence_list


    def trimTrain(self, data, percentage=1.0):
        '''
        :param data: Seq List object
        :param num_docs: porcentaje a conservar
        :return: Sequence List Object
        '''
        train = SequenceList(self.word_dict, self.pos_dict, self.ne_dict)
        res, _ = train_test_split(data.seq_list, test_size=1.0 - percentage, random_state=RANDOM_STATE)
        train.seq_list = res
        return train


##############################################################################################
##########          NEC UTILS
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from scipy import sparse

import classifiers.id_features_NEC as featNEC


class Chunk:
    def __init__(self, sequence_id, pos, length, entity):
        self.sequence_id = sequence_id
        self.pos = pos
        self.length = length
        self.entity = entity  # NOMBRE, NO ID

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
                if len(tag) > 1:
                    ne = tag[2:]
                else:
                    ne = tag

                if ne != 'O' and tag != START_TAG and tag != END_TAG and tag != BR:
                    self.entity_classes.add(ne)
                prev_ne = ne

                if i > 0:
                    prev_tag = self.dataset.y_dict.get_label_name(sequence.y[i - 1])
                    if len(prev_tag) > 1:
                        prev_ne = prev_tag[2:]
                if tag.find('B') == 0:
                    if open and i > 0:
                        chunk = Chunk(sequence_id=seq_id, pos=pos, length=i - pos, entity=prev_ne)
                        self.chunk_list.append(chunk)
                    pos = i
                    open = True
                elif tag.find('I') != 0 and open:
                    open = False
                    chunk = Chunk(sequence_id=seq_id, pos=pos, length=i - pos, entity=prev_ne)
                    self.chunk_list.append(chunk)
            if open:
                chunk = Chunk(sequence_id=seq_id, pos=pos, length=n - pos, entity=ne)
                self.chunk_list.append(chunk)


def getStandart(chunks, feature_mapper, mode='BINARY'):
    X = np.zeros((len(chunks.chunk_list), len(feature_mapper.feature_dict)))
    Y = []
    if mode == 'BINARY':
        Y = np.zeros(len(chunks.chunk_list), dtype=np.int_)
    else:
        Y = np.zeros((len(chunks.chunk_list), len(chunks.entity_classes)))  # CUATRO ENTIDADES

    for i, chunk in enumerate(chunks.chunk_list):
        features = feature_mapper.get_features(chunk, chunks.dataset)
        X[i, features] = 1
        ne_id = chunks.entity_classes.get_label_id(chunk.entity)
        if mode == 'BINARY':
            Y[i] = ne_id
        else:
            Y[i, ne_id] = 1
    return X, Y


def getData_NEC():
    reader = JobDBCorpus()
    data = reader.read_sequence_list(target='TODO')
    np.seterr(all='ignore')

    train, test = reader.train_test_data(test_size=0.2)

    print("Reading chunks...")
    chunks_train = ChunkSet(dataset=train)
    chunks_test = ChunkSet(dataset=test)

    print("Building features...")
    idf = featNEC.IDFeatures(dataset=train, chunkset=chunks_train)
    idf.build_features()

    ###############################################################################
    print("Standarizing dataset...")
    X_train, Y_train = getStandart(chunks_train, idf)
    X_test, Y_test = getStandart(chunks_test, idf)

    # sparse representation and normalize
    X_train = sparse.csr_matrix(X_train)
    X_train = normalize(X_train, copy=False)

    X_test = sparse.csr_matrix(X_test)
    X_test = normalize(X_test, copy=False)

    return X_train, Y_train, X_test, Y_test, chunks_train

##############################################################################################
import classifiers.id_features_NERC as featNERC


def getStandart_NERC(data, feature_mapper):
    BR_x_id = data.x_dict.get_label_id(BR)
    n = 0
    for sequence in data.seq_list:
        for pos in range(2, len(sequence.x) - 1):
            x = sequence.x[pos]
            if x == BR_x_id:
                continue
            n += 1

    n_features = feature_mapper.get_num_features()

    row = []
    col = []
    values = []
    Y = np.zeros(n, dtype=np.int_)

    sample = 0
    for sequence in data.seq_list:
        for pos in range(2, len(sequence.x) - 1):
            x = sequence.x[pos]
            if x == BR_x_id:
                continue
            y_1 = sequence.y[pos - 1]
            y_2 = sequence.y[pos - 2]
            y = sequence.y[pos]

            features = feature_mapper.get_features(sequence, pos, y_1, y_2)
            row.extend([sample for i in range(len(features))])
            col.extend(features)
            Y[sample] = y

            sample += 1
    values = np.ones(len(row))
    X = sparse.csr_matrix((values, (row, col)), shape=(n, n_features))
    return X, Y


def toSeqList(data, seq_array):
    ST_id = data.y_dict.get_label_id(START_TAG)
    END_id = data.y_dict.get_label_id(END_TAG)
    BR_id = data.y_dict.get_label_id(BR)
    BR_x_id = data.x_dict.get_label_id(BR)
    res = []
    pos = 0

    for sequence in data.seq_list:
        y = [ST_id, ST_id]
        for i in range(2, len(sequence.y) - 1):
            id = sequence.y[i]
            if id == BR_id:
                y.append(BR_id)
            else:
                y.append(seq_array[pos])
                pos += 1
        y.append(END_id)
        seq = sequence.copy_sequence()
        seq.y = y
        res.append(seq)

    return res
