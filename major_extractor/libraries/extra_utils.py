#!/usr/bin/python
# -*- coding: utf-8 -*-

import re, string
import unicodedata
import nltk
import pickle

RARE = "<RARE>"

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
permanent_filters = list(range(2,10)) + [11,14,15,16]
filter_names = list(set(['<'+w+'>' for w in filter_tags] + [RARE]))

def permanentFilter(word, filter_idx=permanent_filters):
    wordL = ''
    if word==u'¡' or word==u'¿':
        wordL=word
    else:
        wordL = unicodedata.normalize('NFKD',word.lower()).encode('ascii','ignore').decode('utf8')
    for idx in filter_idx:
        pat = filters[idx]
        if pat.search(wordL):
            return '<' + filter_tags[idx] + '>'
    return word

##############################################################################################################################
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
"""
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
"""
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
    text = text.replace(u'–', '-')
    text = text.replace(u'…', '.')

    """
    # correciones de tildes
    for wrong, good in tildes:
        text = text.replace(wrong, good)
    """

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
        word = word.strip(u'´')
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

        if u'¡' in word and len(word) > 1:
            splitWord = word.split('¡')
            for sw in splitWord:
                if len(sw) > 0:
                    ans.append(sw)
                ans.append('¡')
                change = True
            ans.pop()
            continue

        if u'¿' in word and len(word) > 1:
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

########################################################################################
is_punct = re.compile(r'^[%s]+$' % PUNCTUATION, re.UNICODE)  # IS PUNCTUATION

########################################################################################

def saveObject(obj, name='model'):
    with open(name + '.pickle', 'wb') as fd:
        pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)


def uploadObject(obj_name):
    # Load tagger
    with open(obj_name + '.pickle', 'rb') as fd:
        obj = pickle.load(fd)
    return obj