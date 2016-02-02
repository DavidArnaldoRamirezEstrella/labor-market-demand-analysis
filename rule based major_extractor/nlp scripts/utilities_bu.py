import os,sys
import nltk
import unicodedata
import re
import math
import json
import numpy
import pdb
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import SpanishStemmer
import datetime

RANDOM = 42
numpy.random.seed(RANDOM)

#####################################################################################################
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.join(BASE_DIR, 'www')
CRAWLER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDENTIFIER_DIR = os.path.join(CRAWLER_DIR, 'Identifiers')
IDENTIFIER_STEM_DIR = os.path.join(CRAWLER_DIR, 'Identifiers_stem')
UTIL_DIR = os.path.dirname(os.path.abspath(__file__))

keyword_path  = os.path.join(UTIL_DIR,'smartlist/keywords/ingenieria')
# Not available
#engdata_path = os.path.join(UTIL_DIR,'engineering/engineering')
nltk_stoplist = os.path.join(UTIL_DIR,'stopwords/spanish')
verb_stoplist = os.path.join(UTIL_DIR,'stopwords/verbos')
career_stoplist = os.path.join(UTIL_DIR,'stopwords/carreras')
smart_stoplist = os.path.join(UTIL_DIR,'stopwords/smartlist')
unigram_stoplist = os.path.join(UTIL_DIR,'stopwords/unigram_stopword')
lugares_stoplist = os.path.join(UTIL_DIR,'stopwords/lugares')
syllabus_stoplist = os.path.join(UTIL_DIR,'stopwords/syllabus')

no_stemming_list = os.path.join(UTIL_DIR,'stopwords/no_stemming')       # LISTA  DE PALABRAS Q NO DEBEN SER STEMMEADAS

######################################################################################################################
sys.path.append(PROJECT_DIR)
os.environ['DJANGO_SETTINGS_MODULE'] = 'project.dev'


from core.models import Details, Description

######################################################################################################################
VECTOR_HASH_DIR = os.path.join(UTIL_DIR,'vector_hash')
PSEUDO_PROF = 'vh_pseudo_prof'
PROF = 'vh_prof'
NO_PROF = 'vh_no_prof'
TECNICO = 'vh_tecnico'
NO_TECNICO = 'vh_no_tec'
PURE = 'pure'
JOIN = 'join'

######################################################################################################################
TFIDF = 'tfidf'
FREQ_DOC = 'freq_doc'
FREQ_TOTAL = 'freq_total'

######################################################################################################################
LOG = 'logaritmic'
SQRT = 'sqrt'
TRUNC = 'truncation'

######################################################################################################################
##                              STEMMING SET UP
span_stemmer = SpanishStemmer()
eng_stemmer = SnowballStemmer("english")
punctuation = re.compile(r'[-.?!,":;()|0-9]')
sep_multiword_Lemma = r'(?P<prev>.*\b)(?P<hypen>\s*-\s*)(?P<post>\b.*)'
sep_multiword_Lemma = re.compile(sep_multiword_Lemma)

######################################################################################################################
SPECIAL = ['_','-',"'",'"','`',',', '\n',
           ':',';',"...","..",'.',
           '*','+','/','$','#','=','^','~',
           '(',')',']','[',
           '\n','\r','\t','\a','\b',
           ]
especial_strip = ['_','-',"'",'"','`',',', '\n',
           ':',';',"...","..",'.',
           '*','+','/','$','#','=','^','~',
           ']','[',
           '\n','\r','\t','\a','\b',
           ]
NUM_SPECIAL = ['hora','horas','hr','hrs','am','pm',
               'ano','anos','dia','dias','kg','gr','tn','sol','soles',
               'mm','cm','km','mt',
               'ro','ra','er','ro','ra','ero','era','do','ndo','ero','to','mo','vo','of',
               'ene','feb','mar','abr','may','jun','jul','ago','set','oct','nov','dic',
               ]
FREQ_THR = 5
Vocabulary = {}
rareList = []
numList = []

Sentences = []
termFreq = {}
RARE_TAG = "RARE"
NUM_TAG = "NUM"
START_TAG = '*'
PUNCT_TAG = 'PUNCT'
TAGS = [RARE_TAG,
        NUM_TAG,
        START_TAG,
        PUNCT_TAG,] 

Bigrams  = {}
Trigrams = {}

################################################################################################################################################################################
###                                                         LECTURA ARCHIVOS, STOPWORDS, IDENTIFIERS, STEMMING

def readList(corpus,inverted=False,header=False):
    """ input: list of strings, one ngram per line | inverted read of tuples
        output: dict ngram:score
        describ: reconstruye lista de tuples (TF-IDF, ngram)
    """
    res = []
    for (i,line) in enumerate(corpus):
        if header and i==0:
            continue
        if len(line)<1:
            continue
        idx = 0
        tfidf = 0.0
        ngram = ""
        if inverted:
            idx = line.find(' ')
            tfidf = float(line[:idx])
            ngram = line[idx+1:].split(',')
        else:
            idx = line[::-1].find(',')
            idx= len(line) - idx
            tfidf = float(line[idx:])
            ngram = line[:idx].split(',')
        ngram = [n for n in ngram if len(n)>0]
        res.append( (tfidf,ngram) )
    dd = {}
    for n in res:
        insert(dd,tuple(n[1]),n[0])
    return dd


def readLatin(path):
    """ lee en codificacion UTF-8 e ignora tildes/enhes
    """
    temp = unicode(open(path).read().decode('utf-8'))
    temp = unicodedata.normalize('NFKD', temp).encode('ascii','ignore')
    return temp.lower()


def leer_tags(data):
    """ input : file object
        output: lista de str (lineas)
        descrip: lee los delimitadores de los archivos tags
    """
    ans = []
    line = data.readline()
    while line:
        if len(line) > 0:
            ans.append( line )
        line = data.readline()
    return ans


def getStopWords(stemming = False, ignore_smart = False, unigrams=False, syllabus = False,ignore_careers=False):
    """ input:  stemming : stem stopwords
                ignore_smart : not consider smart stoplist
                unigrams : get only unigram-specific stopwords list
        output: lista de stopwords, lower, ascii - fija y smart
    """
    normal= []
    verb = []
    sml = []
    uni_list = []
    syl = []
    career = []
    if not ignore_careers:
        career   = strip_encode([unicode(line.lower()) for line in readLatin(career_stoplist).split('\n')])

    if syllabus:
        syl   = strip_encode([unicode(line.lower()) for line in readLatin(syllabus_stoplist).split('\n')])
        syl = [w for w in syl  if len(w)>0]
        tt = []
        for phrase in syl:
            temp = phrase.split()
            tt.extend(temp)
        syl.extend(tt)
        syl = list(set(syl))

    if unigrams:
        uni_list = strip_encode([unicode(line.lower()) for line in readLatin(unigram_stoplist).split('\n')])
        if stemming:
            career = [stemAugmented(line.lower()) for line in career]
            uni_list   = [stemAugmented(line.lower()) for line in uni_list]

        career = [w for w in career  if len(w)>0]
        uni_list = [w for w in uni_list if len(w)>0]

        for phrase in career:
            temp = phrase.split()
            uni_list.extend(temp)
        uni_list = list(set(uni_list))
        return uni_list
    else:
        verb  = strip_encode([unicode(line.lower()) for line in readLatin(verb_stoplist).split('\n')])
        places   = strip_encode([unicode(line.lower()) for line in readLatin(lugares_stoplist).split('\n')])
        normal = []
        if stemming:
            normal = [stemAugmented(line.lower()) for line in open(nltk_stoplist).read().split('\n')]
            verb   = [stemAugmented(line.lower()) for line in verb]
            career = [stemAugmented(line.lower()) for line in career]
            places = [stemAugmented(line.lower()) for line in places]
            sml    = [stemAugmented(line.lower()) for line in open(smart_stoplist).read().split('\n')]
            syl    = [stemAugmented(line.lower()) for line in syl]
        else:
            normal = [line.lower() for line in open(nltk_stoplist).read().split('\n')]
            sml    = [line.lower() for line in open(smart_stoplist).read().split('\n')]

        normal = [w for w in normal   if len(w)>0]
        verb   = [w for w in verb if len(w)>0]
        career = [w for w in career  if len(w)>0]
        sml    = [w for w in sml  if len(w)>0]

        if not ignore_smart:
            normal.extend(sml)
        normal.extend(verb)
        normal.extend(career)
        normal.extend(places)
        if syllabus:
            syl = [w for w in syl  if len(w)>0]
            normal.extend(syl)

        normal = list(set(normal))
        return normal


def getStopWords_Syllabus(stemming = False, ignore_smart = False, unigrams=False):
    """
    """
    stopwords = getStopWords(stemming = stemming, ignore_smart = ignore_smart, unigrams=unigrams)
    return stopwords


def getStopWordsEnglish(stemming = False,path=''):
    """ input : stemming :  stem words
                dir : path of list dir
        output: (list) english stopwords
    """
    if stemming:
        stopwords = [ stemAugmented(phrase, eng_stemmer) for phrase in open(path).read().split('\n') if len(phrase) > 0]
    else:
        stopwords = [ phrase for phrase in open(path).read().split('\n') if len(phrase) > 0]
    return stopwords


def getProfIdentifiers():
    """ output: lista de identificadores stemeados
        description: contiene los identificadores de todas las carreras profesionales no tecnicas
    """
    profesional = os.path.join(IDENTIFIER_STEM_DIR,'prof_todo.txt')
    prof_ident = readLatin(profesional).split('\n')
    prof_ident = [stemAugmented(' '.join(punctuation.sub(" ",line).split()),degree=1) for line in prof_ident    if len(line) > 0]

    # Identificar carreras que contengan esos tags
    Group_Identifiers = []

    for root, dirs, filenames in os.walk(IDENTIFIER_STEM_DIR):
        for f in filenames:
            if f[-1]!='~' and f != 'tecnico':
                ident = os.path.join(IDENTIFIER_STEM_DIR, f)
                text_input = strip_encode(leer_tags(open(ident)), False)

                prof_ident.extend(text_input)

    prof_ident = list(set(prof_ident))

    return prof_ident


def getDocnameByCareer(only_majors=False):
    """ :return diccionario de nombres de documentos mapeados por nombre de carreras
    """
    docnameByCareer = {}
    exclude_files = ["prof_todo.txt", "no_tecnico"]
    if only_majors:
        exclude_files += ["ing"]
    for root, dirs, filenames in os.walk(IDENTIFIER_STEM_DIR):
        for f in filenames:
            if f[-1]!='~' and f not in exclude_files:
                ident = os.path.join(IDENTIFIER_STEM_DIR, f)
                #text_input = [unicode(line.lower().strip(' ')) for line in readLatin(ident).split('\n') if len(line) > 0]
                text_input = [unicode(line.lower()) for line in readLatin(ident).split('\n') if len(line) > 0]
                for line in text_input:
                    docnameByCareer[line] = f
    return docnameByCareer


def addSuffixRegex(ident):
    """
    :param ident: lista de oraciones. A cada palabra de cada oracion se le agregara sufijos y prefijos para busqueda RE
    :return: lista de oraciones modificada para busqueda RE
    """
    temp = []

    for line in ident:
        temp_phrase = []
        match = sep_multiword_Lemma.search(line)
        if match:
            prev = [match.group('prev')]
            post = [match.group('post')]
            hypen = match.group('hypen')
            temp_phrase.append(addSuffixRegex(prev)[0].lstrip('(').rstrip(')') +
                               hypen +
                               addSuffixRegex(post)[0].lstrip('(').rstrip(')') )
        else:
            phrase = line.split()
            for word in phrase:
                word = word.replace('.', '[.]')
                if len(word)>2:
                    temp_phrase.append(word + '[a-z().]*')
                else:
                    temp_phrase.append(word)
        temp.append('(\\b' + '\\s+'.join(temp_phrase) + ')')
    return temp


################################################################################################################################################################################
###                                                         ESCRITURA DE NGRAMS A TXT, JSON, CSV

def normalizeFeature(features):
    """ input: dict of features - uni,bi or trigrams
        output: dict of normalized features
        desc: normaliza de acuerdo a (x-x^)/std
    """
    mean = 1#numpy.mean(features.values())
    std  = 2#numpy.std(features.values())

    norm = {}
    for (k,v) in features.iteritems():
        insert(norm,k,float(v-mean)/float(std))

    return norm


def joinFeatures(UG,BG,TG,scale=False):
    """ input:  dicts - unigrams | bigrams | trigrams
                scale: (bool) escalar de 0 a 100
        output: dict  ngrams : normalized_tfidf
        descrip: une los tres conjuntos en uno solo para comparacion
                 Si normalize = true -> normaliza las listas antes de unirlas (usar en analisis de tfidf)
    """
    # normalize
    joined = dict(UG.items() + BG.items() + TG.items())

    joined = [(v,k) for (k,v) in joined.iteritems()]
    joined.sort(reverse=True)

    if scale:
        # Cambiar de rango 0 - 100
        mn = joined[-1][0]
        mx = joined[0][0]

        temp = []
        for (x,k) in joined:
            val = ((x-mn)*100.0)/float(mx-mn)
            if type(k)==str:
                k = [k]
            if val > 1e-5:
                temp.append((val,k))
        joined = temp

    # Filtrar KW repetidos en KW + rankeados
    keywords = []
    for dato in joined:
        s = ' '.join(dato[1])
        keywords.append(s)

    temp = []
    for (i,cur) in enumerate(keywords):
        repetido = False
        for j in range(i):
            if cur in keywords[j]:
                repetido = True
        if not repetido:
            temp.append(joined[i])
    joined = temp

    res = {}
    for n in joined:
        insert(res,n[1],n[0])
    return res


def makeJSON(NG):
    """ input: (str) formato: lista de  {name:ngram,size:score}
        output: string en formato json
        descrip: construye dict tipo json
    """
    json = []
    for (k,v) in NG.iteritems():
        req = ' '.join(k)
        json.append('{"name":"%s","size":%d}' % ( req,round(v) ) )
    json = "[" + ',\n'.join(json) + "]"
    return json


def makeText(NG, separator=':'):
    """ input: dict ngram:score
        output: string formato ngram:NUM
        descrip: construye texto para usar en pag. de wordcloud. score entrada debe ser float
    """
    res = []
    temp = [(v,k) for (k,v) in NG.iteritems()]
    temp.sort(reverse=True)
    for t in temp:
        k = t[1]
        v = t[0]
        req = ' '.join(k)
        res.append('%s%s%.6f' % ( req,separator,v ) )
    res = '\n'.join(res)
    return res


def scale(NG,out_range=[0,75], scales=[], inflectionTH=20, width=0.1):
    """ input: dict ngram:TFIDF | rango de valores de salida | scalas a aplicar | umbral de inflexion
        output: lista de tuples (scaled,ngram)
        descrip: escalamiento (log + truncation) y redondea los pesos para poder graficarlos
                 scale infletion thresh: 20
    """
    temp = []
    for (k,v) in NG.iteritems():
        temp.append( (v,k) )
    NG = temp
    NG.sort(reverse=True)

    #escalamiento logaritmico
    if LOG in scales:
        NG = [ (math.log(k[0]),k[1]) for k in NG ]

    if SQRT in scales:
        NG = [ (math.sqrt(k[0]),k[1]) for k in NG ]

    # escalar tfidf -> [1-75] (default)
    scaled = []
    # descendente
    mn = NG[-1][0]
    mx = NG[0][0]
    domain = [mn,mx]

    for ngram in reversed(NG):
        w = ngram[0]
        newW = 1.0*(out_range[1]-out_range[0])*(w-domain[0])/float(domain[1]-domain[0]) + out_range[0]
        scaled.append( newW )

    if TRUNC in scales:
        inflection_point = -1
        for i in range(len(scaled)-1,0,-1):
            if( scaled[i]<inflectionTH ):             #scale inflection threshold
                inflection_point = i
                break

        if(inflection_point != -1):
            fixed = []
            for i in range(inflection_point+1):
                #temp = scaled[i]

                fixed.append( width*scaled[inflection_point] *
                            ( (scaled[i] - scaled[0])/(scaled[inflection_point]-scaled[0]) )**3 +
                            (1 - width)*scaled[inflection_point] )

            for i in range(inflection_point+1):
                scaled[i] = fixed[i]
    scaled = list(reversed(scaled))
    res = {}
    for (i,v) in enumerate(NG):
        insert(res,tuple(v[1]),scaled[i])
    return res


def writeOutput(source_dir,results_dir,scaling=False,scales=[],normalize=False,handfiltered=True,score=FREQ_DOC, join=False,text=True,json=True):
    """ input: source_dir : (str) path absoluto de .csv filtrados a mano
               results_dir: (str) path absoluto donde grabar resultados
               scaling : (bool) aplicar scalamiento
               scales : (list) identificadores de scalamientos a aplicar
               normalize : normalizar scores de listas (usado en tf-idf)
               handfiltered : (bool) usar lista filtrada a mano (True) | usar lista original de ngramas (False)
               join : (bool) escribir archivo con lista unida de ngramas
               text : (bool) escribir docs .txt con formato para WordClouds Online (wordle,...)
               json : (bool) escribir docs formato json
        output: None
    """
    unigrams = []
    bigrams = []
    trigrams = []

    if not handfiltered:
        # Usando originales
        if score == FREQ_DOC or score == FREQ_TOTAL:
            unigrams = readLatin(os.path.join(source_dir,'freq_unigrams.csv')).split('\n')
            bigrams  = readLatin(os.path.join(source_dir,'freq_bigrams.csv' )).split('\n')
            trigrams = readLatin(os.path.join(source_dir,'freq_trigrams.csv')).split('\n')
        else:
            # score = TFIF
            unigrams = readLatin(os.path.join(source_dir,'tfidf_unigrams.csv')).split('\n')
            bigrams  = readLatin(os.path.join(source_dir,'tfidf_bigrams.csv' )).split('\n')
            trigrams = readLatin(os.path.join(source_dir,'tfidf_trigrams.csv')).split('\n')

        unigrams = readList(unigrams,header=True)
        bigrams  = readList(bigrams,header=True)
        trigrams = readList(trigrams,header=True)
    else:
        # usando hand-filtered
        unigrams = readLatin(os.path.join(source_dir,'ug_handfiltered.csv')).split('\n')
        bigrams  = readLatin(os.path.join(source_dir,'bg_handfiltered.csv')).split('\n')
        trigrams = readLatin(os.path.join(source_dir,'tg_handfiltered.csv')).split('\n')

        unigrams = readList(unigrams)
        bigrams  = readList(bigrams)
        trigrams = readList(trigrams)

    joined = []

    if normalize:
        unigrams = normalizeFeature(unigrams)
        bigrams  = normalizeFeature(bigrams)
        trigrams = normalizeFeature(trigrams)

    if join:
        joined = joinFeatures(unigrams, bigrams, trigrams)

    if scaling:
        unigrams = scale(unigrams,scales=scales)
        bigrams  = scale(bigrams ,scales=scales)
        trigrams = scale(trigrams,scales=scales)
        joined   = scale(joined,scales=scales)

    ####################################################################################
    if json:
        jsonUG = makeJSON(unigrams)
        jsonBG = makeJSON(bigrams)
        jsonTG = makeJSON(trigrams)
        jsonJ  = makeJSON(joined)

        open(os.path.join(results_dir ,'unigram.json'),'w').write(jsonUG)
        open(os.path.join(results_dir ,'bigram.json' ),'w').write(jsonBG)
        open(os.path.join(results_dir ,'trigram.json'),'w').write(jsonTG)
        if json:
            open(os.path.join(results_dir ,'joined.json' ),'w').write(jsonJ)

    if text:
        textUG = makeText(unigrams)
        textBG = makeText(bigrams)
        textTG = makeText(trigrams)
        textJ  = makeText(joined)

        open(os.path.join(results_dir ,'unigram.txt'),'w').write(textUG)
        open(os.path.join(results_dir ,'bigram.txt' ),'w').write(textBG)
        open(os.path.join(results_dir ,'trigram.txt'),'w').write(textTG)
        if json:
            open(os.path.join(results_dir ,'joined.txt' ),'w').write(textJ)


def writeAddOutput(unigrams,bigrams,trigrams,filename='',results_dir='',scaling=False,scales=[],normalize=False,handfiltered=True,score=FREQ_DOC, join=False,text=True,json=True):
    """ input: unigrams,bigrams,trigrams : dict de la forma <tag:score>
               filename : prefijo de nombre de output files
               results_dir: (str) path absoluto donde grabar resultados
               scaling : (bool) aplicar scalamiento
               scales : (list) identificadores de scalamientos a aplicar
               normalize : normalizar scores de listas (usado en tf-idf)
               handfiltered : (bool) usar lista filtrada a mano (True) | usar lista original de ngramas (False)
               join : (bool) escribir archivo con lista unida de ngramas
               text : (bool) escribir docs .txt con formato para WordClouds Online (wordle,...)
               json : (bool) escribir docs formato json
        output: None
    """
    joined = []

    if normalize:
        unigrams = normalizeFeature(unigrams)
        bigrams  = normalizeFeature(bigrams)
        trigrams = normalizeFeature(trigrams)

    if join:
        joined = joinFeatures(unigrams, bigrams, trigrams)

    if scaling:
        unigrams = scale(unigrams,scales=scales)
        bigrams  = scale(bigrams ,scales=scales)
        trigrams = scale(trigrams,scales=scales)
        joined   = scale(joined,scales=scales)

    ####################################################################################
    if json:
        jsonUG = makeJSON(unigrams)
        jsonBG = makeJSON(bigrams)
        jsonTG = makeJSON(trigrams)
        jsonJ  = makeJSON(joined)

        if score == TFIDF:
            open(os.path.join(results_dir ,filename + '_tfidf_unigram.json'),'w').write(jsonUG)
            open(os.path.join(results_dir ,filename + '_tfidf_bigram.json' ),'w').write(jsonBG)
            open(os.path.join(results_dir ,filename + '_tfidf_trigram.json'),'w').write(jsonTG)
            if json:
                open(os.path.join(results_dir ,filename + '_tfidf_joined.json' ),'w').write(jsonJ)
        else:
            open(os.path.join(results_dir ,filename + '_freq_unigram.json'),'w').write(jsonUG)
            open(os.path.join(results_dir ,filename + '_freq_bigram.json' ),'w').write(jsonBG)
            open(os.path.join(results_dir ,filename + '_freq_trigram.json'),'w').write(jsonTG)
            if json:
                open(os.path.join(results_dir ,filename + '_freq_joined.json' ),'w').write(jsonJ)

    if text:
        textUG = makeText(unigrams)
        textBG = makeText(bigrams)
        textTG = makeText(trigrams)
        textJ  = makeText(joined)

        if score == TFIDF:
            open(os.path.join(results_dir ,filename + '_tfidf_unigram.txt'),'w').write(textUG)
            open(os.path.join(results_dir ,filename + '_tfidf_bigram.txt' ),'w').write(textBG)
            open(os.path.join(results_dir ,filename + '_tfidf_trigram.txt'),'w').write(textTG)
            if json:
                open(os.path.join(results_dir ,filename + '_tfidf_joined.txt' ),'w').write(textJ)
        else:
            open(os.path.join(results_dir ,filename + '_freq_unigram.txt'),'w').write(textUG)
            open(os.path.join(results_dir ,filename + '_freq_bigram.txt' ),'w').write(textBG)
            open(os.path.join(results_dir ,filename + '_freq_trigram.txt'),'w').write(textTG)
            if json:
                open(os.path.join(results_dir ,filename + '_freq_joined.txt' ),'w').write(textJ)



################################################################################################################################################################################
###                                                         PROCESADO DE TEXTO


def especial(car):
    """retorna verdadero o falso si el caracter pertenece al conjunto
    """
    return car in especial_strip


def insert(dic, word, freq = 1):    
    if word=="":
        return
    
    if word in dic:
        dic[word] = dic[word] + freq
    else:
        dic[word] = freq


def strip_encode(text,flag_code = True):
    """ devuelve el ascii y ademas elimina
        los caracteres especiales
    """
    ans = []
    if type(text)==str or type(text)==unicode:
        text = unicodedata.normalize('NFKD', unicode(text)).encode('ascii','ignore')
        texto = text.lower().strip(' ')
        while(len(texto)>0 and (especial(texto[-1]) or especial(texto[0]) )):
            for special in especial_strip:
                texto = texto.strip(special).strip(' ')
        
        return texto
    if flag_code==True:
        text = [unicodedata.normalize('NFKD', unicode(line)).encode('ascii','ignore') for line in text]
    for texto in text:
        texto = texto.lower()
        while(len(texto)>0 and (especial(texto[-1]) or especial(texto[0]) )):
            for special in especial_strip:
                texto = texto.strip(special).strip(' ')
        
        if len(texto)>1:
            ans.append(texto)
        
    return ans


def isLabelNUM(word):
    """ input: string to analyse
        output: boolean
        descrip: si no hay digito, Falso
                 si # letras contiguas (mL) >= 2 buscar NUM-SP
    """
    hasDig = False
    lenC = 0
    maxLenC = 0
    for c in word:
        if c.isdigit():
            hasDig = True
            maxLenC = max(maxLenC,lenC)
            lenC = 0
        elif c.isalpha():
            lenC = lenC + 1
    maxLenC = max(maxLenC,lenC)
    
    if not hasDig:
        return False
    
    if maxLenC < 2:
        return True
    else:
        for nsp in NUM_SPECIAL:
            if nsp in word and maxLenC <= len(nsp)+2:
                return True
        return False


def clean_splitToken(word):
    """ input: raw token
        output: LIST of tokens free of special characters,
        descrip: recibe palabra y la cataloga en algun Label especial para reducir ruido
                 si no, devuelve el token limpio de SPECIALS, considera PUNCT_TAG
    """
    PUNCTUATION = [',',';',':']
    # condiciones de longitud de token
    #    long 1:no significativo si no es letra
    if len(word)==1 and word in PUNCTUATION:
        return [PUNCT_TAG]
    if len(word)==1 and not word.isalpha() and not word.isdigit():
        return []
    #    long > CONT: no hubo espacio, mal escrito
    
    for sp in SPECIAL:
        # casos especiales donde se requiere del special
        if(any([('++' in word and sp == '+' and len(word)<6),
               ('y/o' in word and sp == '/'),
               ('c#' in word and sp == '#'),
               (any(['asp.net' in word,
                    'rr.hh'    in word,
                    '.aa'      in word]) and sp=='.'),
           ])):
            continue
        word = word.replace(sp,' ')
    word = word.strip(' ')

    if word == '':
        return []
    return [ w for w in word.split() if w != '']
     

################################################################################################################################################################################
###                                                         STEMMING
# CASOS ESPECIALES: PALABRAS Q NO DEBEN SER LEMMATIZADAS
special_lemma_list = strip_encode([unicode(line.lower()) for line in readLatin(no_stemming_list).split('\n') if len(line)>0])

def stemAugmented(phrase,stemmer = span_stemmer,degree='inf'):
    """ input: str, stem object , degree
        ouput: string
        description: itera varias veces el stemmer del NLTK. NO funciona para todos los casos
    """
    if type(phrase) == tuple:
        phrase = ' '.join(phrase)

    stemmed = []
    phrase = phrase.split()

    for word in phrase:
        match = sep_multiword_Lemma.search(word)
        if match:
            prev = match.group('prev')
            post = match.group('post')
            stemmed.append(stemAugmented(prev) + match.group('hypen') + stemAugmented(post))
        else:
            res = ""
            found_spec = False
            for sp_lemma in special_lemma_list:     # si encuentra
                if word.find(sp_lemma) == 0:       # encuentra al inicio
                    res = sp_lemma
                    found_spec = True
                    break
            
            if not found_spec:
                if degree=='inf':
                    flag = True
                else:
                    flag = degree
                while flag:
                    res = stemmer.stem(word)
                    if res == word:
                        break
                    word = res
                    if degree != 'inf':
                        flag -= 1
            stemmed.append(res)
    return ' '.join(stemmed)


################################################################################################
ignore_file = os.path.join(UTIL_DIR,'stopwords/ignore_words')
ignore_words = [stemAugmented(unicode(w.lower()),degree=1) for w in readLatin(ignore_file).split('\n') if len(w)>0]
ignore_patts = addSuffixRegex(ignore_words)
################################################################################################

def searchIdentifier(text,ident):
    """ input  : list of str | list of str
        output : bool
        descrip: busca los tags de [:ident:] en [:text:]. preprocesa el lemma para correr RegEx
                 True si encuentra algun tag
    """
    if type(ident) == str or type(ident) == unicode:
        ident = [ident]
    if type(text) == str or type(ident) == unicode:
        text = [text]
    # DROP IGNORING_WORDS
    ign = '|'.join(ignore_patts)
    temp = []
    for line in text:
        temp.append( re.sub(ign,'...',line) )
    text = list(temp)
    ###########
    to_addsuff = []
    no_add = []
    for id in ident:
        if len(id.split())>1:
            to_addsuff.append(id)
        else:
            id = id.replace('.','[.]')
            if id[0]==' ':
                no_add.append(id)
            else:
                no_add.append('\\b'+id)

    to_addsuff = addSuffixRegex(to_addsuff)

    ident = to_addsuff + no_add
    ident = '|'.join(ident)
    pattern = ''
    pattern = re.compile(ident)
    
    for line in text:
        match = pattern.search(line)
        if match:
            return True
    return False


def stemData(V,RL,NL):
    """ input: TF: term frequency dictionary
               V: vocabulary dict
               RL: rare tokens list
               NL: num labeled tokens list
        output: (sTF,sV,sRL,sNL), stemmed versions of lists and dicts
        description: stem cada token en cada list/dict y lo acumula localmente
    """
    sV = {}
    sRL = []
    sNL= []
    for (k,v) in V.iteritems():
        token = ""
        if k==NUM_TAG or k==RARE_TAG:
            token = k
        else:
            token = stemAugmented(k)
        insert(sV,token,v)

    for k in RL:
        sRL.append(stemAugmented(k))

    for k in NL:
        sNL.append(stemAugmented(k))

    return (sV,sRL,sNL)


################################################################################################################################################################################
###                                                         ESTRUCTURACION DE TEXTO

def augmentedTokenizer(data):
    """ input: document as string 
        ouput: list of tokens
        description: create list of unique pre processed terms 
    """
    tokens = nltk.word_tokenize(data)
    vocab = []
    for (i,word) in enumerate(tokens):
        # caso especial con c#: tokenizer lo separa
        if word == '#' and i>0:
            word = tokens[i-1] + word
        for tk in clean_splitToken(word):
            vocab.append(tk)
    #vocab = list(set(vocab))
    return vocab


def sentenceTokenizer(Document,RL,NL,filters=[],stem_flag=False):
    """ input : Document corpus - can be whole corpus data or post document
                rare list , num labeled list,
                filters & to stem doc flag
        output: sentence list of document
        desc: divide en lista de oraciones, considera texto entre parentesis como oracion a parte

    """
    BOUNDARIES_INI = ['(','[','{']
    BOUNDARIES_FIN = [')',']','}']
    sentence_list = []
    seq = []

    for line in Document:
        seq = [START_TAG,START_TAG]
        extra = []
        tokens = nltk.word_tokenize(line)
        bound_ini = -1
        bound_fin = -1
        for (i,metaword) in enumerate(tokens):
            if bound_ini == -1 and bound_fin == -1:
                if i+1<len(tokens) and tokens[i+1]=='#':
                    continue
                if metaword == '#' and i>0:
                    metaword = tokens[i-1]+metaword

                words = clean_splitToken(metaword)
                for w in words:
                    if stem_flag:
                        w = stemAugmented(w)
                    if NUM_TAG in filters and w in NL:
                        seq.append(NUM_TAG)
                    elif RARE_TAG in filters and w in RL:
                        seq.append(RARE_TAG)
                    else:
                        seq.append(w)

            if metaword in BOUNDARIES_INI:
                temp_idx = 0
                if i>0:
                    temp_idx = line.find(tokens[i-1])
                bound_ini = line[temp_idx:].find(metaword) + temp_idx

            if metaword in BOUNDARIES_FIN:
                temp_idx = 0
                if i>0:
                    temp_idx = line.find(tokens[i-1])
                bound_fin = line[temp_idx:].find(metaword) + temp_idx

                extra.extend(sentenceTokenizer( [line[bound_ini+1:bound_fin]],RL,NL,filters,stem_flag) )
                bound_ini = -1
                bound_fin = -1

        if bound_ini!=-1 and bound_fin==-1:
            # parentesis sin cerrar
            extra.extend(sentenceTokenizer( [line[bound_ini+1:]],RL,NL,filters,stem_flag) )

        sentence_list.append(seq)
        if len(extra)!=0:
            sentence_list.extend(extra)

    return sentence_list


def updateTermFreq(document,term_dict):
    """ input : Document corpus - can be whole corpus data or post document
        output: updated TermFreq list
        desc: clean, preprocess and tokenize.
              arma term_freq dictionary. terminos de una sola palabra.
    """
    for line in document:
        tokens = augmentedTokenizer(line)
        for tk in tokens:
            insert(term_dict,tk)


def getVocabulary(term_dict,filters=[]):
    """ input : term frequency dictionary, list of filters to apply over raw data
        output: (vocabulary dict , rare terms list, num tagged terms list)
        desc  : extracts all term whose frequency is higher than a threshold
                if a term freq is lower than FREQ_THR it is labeled as RARE_TAG
    """
    rare = []
    nums = []
    modifiedTD = {}

    # NUM Filter
    if NUM_TAG in filters:
        if len(modifiedTD)==0:
            modifiedTD = term_dict
        temp = {}
        for k,v in modifiedTD.iteritems():
            if isLabelNUM(k):
                insert(temp,NUM_TAG,v)
                nums.append(k)
            else:
                insert(temp,k,v)
        modifiedTD = temp

    # RARE Filter
    if RARE_TAG in filters:
        if len(modifiedTD)==0:
            modifiedTD = term_dict
        temp = {}
        for k,v in modifiedTD.iteritems():
            if v < FREQ_THR or len(k)>20:
                insert(temp,RARE_TAG,v)
                rare.append(k)
            else:
                insert(temp,k,v)
        modifiedTD = temp

    if len(filters)==0:
        modifiedTD = term_dict


    rare = list(set(rare))
    nums = list(set(nums))

    rare.sort()
    nums.sort()
    return (modifiedTD,rare,nums)


def getVocabSentences(corpus,option="bydoc",stem_flag=False,filters=[]):
    """ input: corpus:  string  | list of docs
               option:  all     | byDoc
               stem_flag: get stemmed data
               filters to use on data - NUM, RARE
        ouput: (vocabulary, sentence list)
        description: resume todo las pruebas hasta ahora. Usado con corpus en forma de String
                     Para corpus en DB usar getPreprocessedData
    """
    sentences = []
    RL = []
    NL = []
    V = {}
    TF = {}

    print 'Building vocab...'
    if option=='all':
        updateTermFreq(corpus.split('\n'),TF)
    else:
        k = 0
        for doc in corpus:
            updateTermFreq(doc.split('\n'),TF)
            if k%1000==0:
                print '->',k
            k += 1
    print 'Filtering vocab...'
    (V,RL,NL) = getVocabulary(TF,filters)
    if stem_flag:
        (V,RL,NL) = stemData(V,RL,NL)

    print 'Tokenizing with filter lists...'
    if option=='all':
        sentences = sentenceTokenizer(corpus.split('\n'), RL, NL, filters, stem_flag)
    else:
        k = 0
        for i,doc in enumerate(corpus):
            if i<=24267:
                continue
            try:
                _doc = (line for line in doc.split('\n') if line!='')
                pdb.set_trace()
                sentences.append( sentenceTokenizer(_doc,RL,NL,filters,stem_flag) )
            except:
                pdb.set_trace()
            if k%1000==0:
                print '->',k
            #if k>=24000:
            #    print '->',k
            k += 1


    return (V,sentences)


################################################################################################################################################################################
###                                                         VECTOR HASH RETRIEVING | processing

def writeDebug(jobs, path,name,ids=[]):
    """
    :param jobs: lista de trabajos, formato (details_hash,description_hash)
    :param path: path de carpeta donde guardar archivo
    :param name: nombre del archivo
    :param ids: lista de indices a escribir
    :return: None
    """
    if ids==[] or len(ids)> len(jobs):
        ids = range(len(jobs))

    ff = open(os.path.join(path,name),'w')
    c = 0
    for (i,post) in enumerate(jobs):
        if i in ids:
            det_pk = post[0]
            desc_pk = post[1]
            job = Details.objects.filter(pk=det_pk)[0]
            desc = job.description_set.filter(pk=desc_pk)[0]
            ff.write('''************************************************     [%d]      ***********************************************************
***************     Title     ***************
%s\n
***************     Requerimientos **********
%s\n
***************     Funciones ***************
%s\n\n\n''' % (c,job.title,desc.requirements.strip('\n'),desc.functions.strip('\n')))
            c += 1


def writeHashVector(data,name):
    """
    :param data: lista de tuples (hash_det,hash_descr)
    :param name: nombre con el cual guardar
    :return: None
    """
    ff = open(os.path.join(VECTOR_HASH_DIR,name),'w')
    for post in data:
        ff.write("%s,%s\n" % (post[0],post[1]))


def readHashVector(name):
    """
    :param name: nombre de vh a buscar
    :return: lista de tuples (hdet, hdesc)
    """
    res = []
    full_name = os.path.join(VECTOR_HASH_DIR,name)
    file = open(full_name)
    line = file.readline()
    while line:
        if len(line) > 0:
            res.append(tuple(line.strip('\n').split(',')))
        line = file.readline()
    return res


def updateVHfromDB(career='all',data=[], query_date=[]):
    '''
    :param career: <nombre de archivo> de carrera cuyo VH actualizar
    :param data: lista de <job objects> de la tabla Details
    :return:
    '''
    if career == 'all':
        if query_date==[]:
            data = Details.objects.all()
        else:
            [d,m,y] = [int(aa) for aa in query_date[0].split('-')]
            _from = datetime.date(y,m,d)
            _to = ''
            
            if len(query_date) > 1:
                [d,m,y] = [int(aa) for aa in query_date[1].split('-')]
                _to = datetime.date(y,m,d)
            else:
                _to = datetime.date.today()
            data = Details.objects.filter(
                    date__gte = _from
                ).filter(
                    date__lte = _to
                )
    vh_list = []

    for job in data:
        hash_det = job.hash
        hash_descrip_list = job.description_set.all()

        for desc in hash_descrip_list:
            hash_desc = desc.hash
            vh_list.append([hash_det,hash_desc])
    name = 'vh_' + career

    writeHashVector(vh_list,name)


def existVectorHash(name):
    """
    :param name: nombre de vector hash de carrera a buscar
    :return: Bool
    """
    vh_path = os.path.join(VECTOR_HASH_DIR,name)
    return os.path.exists(vh_path)


def sample_write_VectorHash(name,results_dir=UTIL_DIR,samples=50,explicit=False):
    """
    :param name: nombre de la carrera cuyo vector hash se va a muestrear
    :param samples: numero de muestras aleatoreas a escribir
    :param results_dir: directorio donde escribir los resultados | default: mismo directorio q script
    :return: None
    :descrip: Escribe en un archivo con el titulo, req y funciones de cada trabajo, tomando <samples> muestras aleatoreamente
    """

    if explicit:
        name = "vh_" + name
    else:
        docnameByCareer = getDocnameByCareer()
        name = "vh_" + docnameByCareer[name]

    if not existVectorHash(name):
        print("Vector Hash para %s no existe!! Corra la funcion getOnlyCareer para esta carrera" % name)
    else:
        jobs_filtered = readHashVector(name)
        id_posts = numpy.random.random_integers(0,len(jobs_filtered),samples)
        name = "sample_" + name
        writeDebug(jobs_filtered,results_dir,name,id_posts)

        print("N = %d muestras aleatoreas escritas" % samples)
        print("Archivo : %s" % os.path.join(results_dir,name))


def sampleVectorHash(data,samples=50):
    """
    :param data: vector hash como lista de tuples
    :param samples: numero de muestras aleatoreas a escribir
    :return: vector hash muestreado aleatoreamente
    :descrip: Muestrea aleatoreamente <samples> muestras y las retorna
    """
    id_posts = numpy.random.random_integers(0,len(data),samples)
    if len(data) <= samples:
            return data
    res = []
    for (i,post) in enumerate(data):
        if i in id_posts:
            res.append(post)
    return res


def preprocessVectorHashData(data, option='bydoc'):
    """
    :param data: lista de tuples(hash_det,hash_desc) obtenidos de leer los vector hash
    :param option: 'all':   str, todos los documentos en un string
                   'bydoc': lista de str - Cada doc es un str
    :return:
    """
    Corpus = []
    k = 0
    for i,post in enumerate(data):
        det_pk = post[0]
        desc_pk = post[1]
        job = Details.objects.filter(pk=det_pk)[0]
        desc = job.description_set.filter(pk=desc_pk)[0]
            
        # armar texto
        title = strip_encode([job.title])
        title.append('')
        description = strip_encode([w for w in desc.description.split('\n')])
        cuerpo = list(title) # copia explicita
        cuerpo.extend(description)

        if option=='all':
            Corpus.extend(cuerpo)
        else:
            Corpus.append('\n'.join(cuerpo))
        if k%1000==0:
            print "->",k
        k += 1

    if option=='all':
        Corpus = '\n'.join(Corpus)

    return Corpus


################################################################################################################################################################################
###                                                         SETUP DE FILTRO POR REG_EX

pattern_dir = os.path.join(UTIL_DIR,'patterns')
sys.path.append(pattern_dir)
from regex_patterns import *

docnameByIdent = getDocnameByCareer()
###################################################################################################
# CASOS ESPECIALES - para que no los detecten/separen los patrones
# Mecanica Electrica
ident_mec_elect = [w for w in docnameByIdent.keys() if docnameByIdent[w]=='mecanica_electrica']
ident_mec_elect = '|'.join(addSuffixRegex(ident_mec_elect))

# Tecnologia Medica
ident_tec_med = [w for w in docnameByIdent.keys() if docnameByIdent[w]=='tecnologia_medica']
ident_tec_med = '|'.join(addSuffixRegex(ident_tec_med))

spec_case_patterns = [
    re.compile(addSuffixRegex(['ing'])[0] ),
    re.compile(ident_mec_elect),
    re.compile(ident_tec_med),
    re.compile(addSuffixRegex([stemAugmented('redes y comunicaciones')] )[0] ),
    re.compile(addSuffixRegex(['computacion e informatica']             )[0] ),
    re.compile(addSuffixRegex(['logistica y operaciones']               )[0] ),
    re.compile(addSuffixRegex(['operaciones y logistica']               )[0] ),
    re.compile(addSuffixRegex(['sistemas e informacion']                )[0] ),
    re.compile(addSuffixRegex([stemAugmented('medico veterinario')]     )[0] ),
    re.compile(addSuffixRegex([stemAugmented('estudio de abogado')] )[0] ),
    re.compile(r'^educacion\s*:'),
    re.compile(r'\([a-z]\)'),
]

spec_case_subs = [
    'ing',                      # normalizacion de ident de ingenieria
    'mecanica electrica',
    'medizintechnik',           # dafuq, tecnologia medica en Aleman xD
    'redes Y comunicaciones',
    'computacion E informatica',
    'logistica Y operaciones',
    'operaciones Y logistica',
    'sistemas E informacion',
    'veterinaria',
    'estudio_de_abogado',
    'formacion:',
    '',                         # elimina (a) o (o)
]


################################################################################################################################################################################
###                                                         FILTRADO Y OBTENCION DE POSTS y SYLLABUS | FILTRO: CONJUNTOS

def filterData(data,identifiers=[],ignore=True, complement = False):
    """
    :param data: lista de tuples (Det_hash,Desc_hash) a filtrar
    :param identifiers: lista de identificadores
    :param ignore: True: ignorar identificadores
    :return: Complemente = False: lista de tuples (Det_hash,Desc_hash) filtrada
             Complemente = True: (respuesta,complemento)
    """
    res = []
    no_res = []
    for post in data:
        det_pk = post[0]
        desc_pk = post[1]
        job = Details.objects.filter(pk=det_pk)[0]
        requirements = job.description_set.filter(pk=desc_pk)[0].requirements

        tituloInput = punctuation.sub(" ",job.title).split()
        descInput = punctuation.sub(" ",requirements).split()
        if ignore:
            if not searchIdentifier(tituloInput,identifiers) and not searchIdentifier(descInput,identifiers):
                res.append(post)
            elif complement:
                no_res.append(post)
        else:
            if searchIdentifier(tituloInput,identifiers) or searchIdentifier(descInput,identifiers):
                res.append(post)
            elif complement:
                no_res.append(post)
    if complement:
        return (res,no_res)
    else:
        return res


def getOnlyCareer(career='no tecnico',debug=True):
    """ input: career: (str) nombre de carrera a filtrar
                 'profesional' : todas las carreras
                 'tecnico' : carreras tecnicas
                 'otros' : demas
        output: lista de tuples (Det_hash,Desc_hash)
        description: objetos filtrados de la forma A* = (A U B) \ B (solo A)
    """
    # Utilizar data ya existente si existe
    career_stem = stemAugmented(career,degree=1)
    docnameByCareer = getDocnameByCareer()

    filename_career = 'vh_' + docnameByCareer[career_stem]
    if existVectorHash(filename_career):
        return readHashVector(filename_career)

    # Crear carpeta de VECTOR_HASH_DIR si no existe
    if not os.path.exists(VECTOR_HASH_DIR):
        os.makedirs(VECTOR_HASH_DIR)

    ###########################################################################
    # Obtener identificadores
    ident_no_tecnico = strip_encode(leer_tags(open(os.path.join(IDENTIFIER_STEM_DIR,'no_tecnico'))), False)
    ident_tecnico    = strip_encode(leer_tags(open(os.path.join(IDENTIFIER_STEM_DIR,'tecnico'))), False)
    ident_prof = getProfIdentifiers()


    ###########################################
    # 1er filtro : Pseudo profesional (P) y  no profesional (P~)
    if existVectorHash(PSEUDO_PROF):
        primer_filtro = readHashVector(PSEUDO_PROF) # P
    if existVectorHash(NO_PROF):
        no_primer_filtro = readHashVector(NO_PROF)  # P~

    if not existVectorHash(PSEUDO_PROF) or not existVectorHash(NO_PROF):
        jobs = Details.objects.all()
        temp = []
        for job in jobs:
            for desc in job.description_set.all():
                temp.append(tuple([job.hash,desc.hash]))
        data = temp

        (primer_filtro,no_primer_filtro) = filterData(data=data,identifiers=ident_prof,ignore=False,complement=True)
        writeHashVector(primer_filtro,PSEUDO_PROF)
        writeHashVector(no_primer_filtro,NO_PROF)

    if debug:
        print("---> 1er filtro: Pseudo profesional | No profesional")
        print("--->    size   :  %d                |     %d" % (len(primer_filtro), len(no_primer_filtro)))

    if career == 'no tecnico':
        # 2do filtro : No tecnico sobre No profesional
        res = filterData(data=no_primer_filtro,identifiers=ident_tecnico,ignore=True)
        if debug:
            print("Target: No tecnico | Num posts: %d" % len(res))
        writeHashVector(res,NO_TECNICO)
    elif career == 'tecnico':
        # 2do filtro : Solo_tecnico sobre No profesional
        res = filterData(data=no_primer_filtro,identifiers=ident_tecnico,ignore=False)

        if debug:
            print("Target: Tecnico | Num posts: %d" % len(res))
        writeHashVector(res,TECNICO)
    else:
        # Carreras profesionales
        if debug:
            print("---> Target: Carrera profesional de %s" % career)

        #### 2do filtro : no tecnico sobre Pseudo prof
        segundo_filtro = []
        if existVectorHash(PROF):
            segundo_filtro = readHashVector(PROF)
        else:
            segundo_filtro = filterData(data=primer_filtro,identifiers=ident_tecnico,ignore=True)
            writeHashVector(segundo_filtro,PROF)

        if debug:
            print("--->    2do filtro: Profesionales | num posts: %d" % len(segundo_filtro))

        #### 3er filtro : Pseudo career | no career ~
        # Separar identificadores de carrera del total de profesionales
        ignore_list = []
        career_tags = []
        for root, dirs, filenames in os.walk(IDENTIFIER_STEM_DIR):
            for f in filenames:
                if f[-1]!='~' and f != "prof_todo.txt":
                    ident = os.path.join(IDENTIFIER_STEM_DIR, f)
                    text_input = strip_encode(leer_tags(open(ident)), False)

                    if career_stem in text_input:
                        career_tags.extend(text_input)
                    else:
                        ignore_list.extend(text_input)
        ignore_list = list(set(ignore_list))
        career_tags = list(set(career_tags))

        # no solicitado en query pero util en vector de hash
        pseudo_career = "vh_pseudo_" + docnameByCareer[career_stem]
        # usado en tercer filtro
        comp_no_career = "vh_comp_no_" + docnameByCareer[career_stem]
        tercer_filtro = []

        if not existVectorHash(pseudo_career):
            temp = filterData(data=segundo_filtro,identifiers=career_tags,ignore=False)
            writeHashVector(temp,pseudo_career)

        if existVectorHash(comp_no_career):
            tercer_filtro = readHashVector(pseudo_career)
        else:
            tercer_filtro = filterData(data=segundo_filtro,identifiers=ignore_list,ignore=True)
            writeHashVector(segundo_filtro,comp_no_career)

        if debug:
            print("--->    3er filtro: Complemento de <no %s> | num posts: %d" % (career,len(tercer_filtro)))

        # 4to filtro: only <career>
        only_career = "vh_" + docnameByCareer[career_stem]
        if existVectorHash(only_career):
            cuarto_filtro = readHashVector(only_career)
        else:
            cuarto_filtro = filterData(data=tercer_filtro,identifiers=career_tags,ignore=False)
            writeHashVector(cuarto_filtro,only_career)

        if debug:
            print("--->    4to filtro: Only <%s> | num posts: %d'" % (career, len(cuarto_filtro)))

        res = cuarto_filtro
    return res


def getPreprocessedData(option='bydoc',career_tags=[], max_docs = 100000):
    """ input  : option (all, bydoc) | lista de tags de carreras a filtrar | maxima cantidad de job a retornar
        output : list of lines all-in-one or by-docs mode
        descrip: Arma lista de posts de las carreras deseadas.
                 all: string unico con todo el corpus
                 bydoc: lista de strings (docs).
    """
    # Get identifiers by group
    cont = 0
    career_tags = [stemAugmented(tag) for tag in career_tags]
    query_areas = []
    # Identificar carreras que contengan esos tags
    for root, dirs, filenames in os.walk(IDENTIFIER_STEM_DIR):
        for f in filenames:
            if f[-1]!='~':
                ident = os.path.join(IDENTIFIER_STEM_DIR, f)
                text_input =  strip_encode([unicode(line.lower()) for line in readLatin(ident).split('\n') if len(line)>0 ])
                if searchIdentifier(text_input,career_tags):
                    query_areas.append(f)       # Guardar el docname

    #Obtener la data
    data = []
    update_area = []
    for area in query_areas:
        temp = 'vh_' + area
        if existVectorHash(temp):
            data.extend(readHashVector(temp))
        else:
            update_area.append(area)

    if len(update_area) > 0:
        all_data = readHashVector('vh_all')
        vh_list = [[] for i in range(len(update_area))]
        patrones = compilePatterns()
        for post in all_data:
            det_pk = post[0]
            desc_pk = post[1]
            job = Details.objects.filter(pk=det_pk)[0]
            desc = job.description_set.filter(pk=desc_pk)[0]
            # armar texto
            title = strip_encode([job.title])
            title.append('')
            description = strip_encode([w for w in desc.description.split('\n')])
            cuerpo = list(title) # copia explicita
            cuerpo.extend(description)
            # Variables para identificacion de carreras
            carreras = set()
            found_areas = []
            cuerpo_temp = []
            ing_post = False       # flag si posiblemente es un aviso de ingenieria

            for line in cuerpo:
                # preprocesado de casos especiales (ver header)
                for i,pat in enumerate(spec_case_patterns):
                    line = pat.sub(spec_case_subs[i],line)
                if spec_case_patterns[0].search(line):
                    ing_post = True                         # aviso pide ingenieros
                cuerpo_temp.append(line)
                # busqueda de patrones
                for (i,pattern) in enumerate(patrones):
                    for car in careersFromPatterns(line,patrones,i):
                        carreras.add(car.lower())
            cuerpo = list(cuerpo_temp)
            carreras = list(carreras)       # set -> list
            carreras = [stemAugmented(line).strip(' ') for line in carreras]
            carreras = list(set(carreras))
            ING_STEM  = 'ing'
            # buscar en identificadores | probar agregando ingenieria +
            for option in carreras:
                # variaciones nombres de ingenierias
                op1 = ING_STEM + " " + option
                op2 = ING_STEM + " de " + option
                op3 = ING_STEM + " en " + option
                op = [option,op1,op2,op3]
                for car in op:
                    if car in docnameByIdent.keys():
                        # CASO ESPECIAL : QUIMICA | ING QUIMICA
                        if any([car=='ing quimic',
                                car=='ing en quimic',
                                car=='ing de quimic',
                                car=='quimic']):                    # detect case
                            if ing_post:                            # es post de ingenieria?
                                found_areas.append('quimica')       # ing quimica
                            else:
                                found_areas.append('quimico')       # quimico puro
                        else:                                       # caso general
                            found_areas.append(docnameByIdent[car])
            if len(found_areas)==0:
                # buscar ident de tecnico menos 'tecnico'
                if searchIdentifier(cuerpo,ident_tecnico):
                    found_areas.append('tecnico')
                else:
                    # buscar ident de carreras en titulo
                    found = False
                    special_med = any([ searchIdentifier(cuerpo[0],stemAugmented('ventas')          ),
                                        searchIdentifier(cuerpo[0],stemAugmented('producto medico') ),
                                        searchIdentifier(cuerpo[0],stemAugmented('visitador medico')),
                                        ])
                    for (ident,doc) in docnameByIdent.iteritems():
                        # CASO ESPECIAL DE VENDEDORES DE PRODUCTOS MEDICOS
                        if doc == 'medicina' and special_med:
                            continue
                        if searchIdentifier(cuerpo[0],ident):
                            found_areas.append(doc)
                            found = True
                    if not found:
                        found_areas.append('otros')
            found_areas = list(set(found_areas))
            for (i,a) in enumerate(update_area):
                if a in found_areas:
                    tp = [job.hash,desc.hash]
                    vh_list[i].append(tp)
                    # agregar VH tuples a DATA
                    if tp not in data:
                        data.append(tp)
        #ENDFOR ALL_DATA
        for (i,area) in enumerate(update_area):
            writeHashVector(vh_list[i],"vh_" + area)
    #ENDIF UPDATE_AREAS

    # Load Data
    Corpus = preprocessVectorHashData(data,option=option)

    return Corpus


def getSyllabusData(option='bydoc',path = '',max_docs = 100000):
    """ input : option (str) (all, bydoc)
                path : (str) donde se encuentran los syllabus
                max_docs : (int) maxima cantidad de sylllabus a retornar
        output: list of lines all-in-one or by-docs mode
    """
    corpus = []
    for root, dirs, filenames in os.walk(path):
        for f in filenames:
            if f[-1]!='~':
                syllabo = os.path.join(path, f)
                temp =  strip_encode([unicode(line.lower()) for line in readLatin(syllabo).split('\n') if len(line)>0 ])
                temp = [t for t in temp if len(t)>0]
                temp = '\n'.join(temp)

                if len(temp) > 0:
                    if option == 'all':
                        corpus.extend(temp)
                    else:
                        corpus.append(temp)
    if option == 'all':
        corpus = '\n'.join(corpus)
    return corpus

################################################################################################################################################################################
###                                                         PROCESADO Y FILTRADO DE NGRAMAS

def TF_IDF(term_freq, total_tokens,docs,NUM_DOCS):
    """ input : term frequency in doc
                total_tokens in doc
                # docs where token appears
                # total docs
        output: TF_IDF feature as float
        descrip: classic TF IDF
    """
    tf = term_freq / float(total_tokens)
    idf = 0
    if docs == 0:
        idf = 1.0
    else:
        idf = 1.0 + math.log(NUM_DOCS/float(docs))

    return tf * idf


def filterNgrams(NG,stopwords,ngrams=1,stem_flag = False,stemmer = span_stemmer):
    """ input:  dict de ngrams: <ngram> : freq
                lista de stopwords - smart & normal
                stemming flag
                stemmer : funcion de stemming
        output: dict ngrams,     ngram_filtered:score
        description: Unigramas: no TAG(PUNCT,NUM,RARE) ni stopword
                     Bigramas:  WORD WORD / no tag ni stopword
                     Trigramas: WORD <ANY> WORD / any pude ser palabra o stopword - no tag
                     Si stem_flag es True compara lemmas en vez de palabras | stopwords debe estar stemmed si True
    """
    temp = {}
    unigrams_stopwords = []
    if ngrams == 1:
        # util solamente para espanhol, en cuenta de ingles no tiene efecto alguno
        unigrams_stopwords = getStopWords(stemming = stem_flag, unigrams=True)

    for (k,v) in NG.iteritems():
        if ngrams==1:
            lemma = k
            if stem_flag:
                lemma = stemAugmented(k,stemmer=stemmer)
            if (all([lemma not in stopwords,
                     lemma not in unigrams_stopwords,
                     k not in TAGS,
                     ])):
                insert(temp,k,v)
        elif ngrams==2:
            lem1 = k[0]
            lem2 = k[1]
            if stem_flag:
                lem1 = stemAugmented(k[0],stemmer=stemmer)
                lem2 = stemAugmented(k[1],stemmer=stemmer)
            if (all([lem1 not in stopwords,
                     lem2 not in stopwords,
                     ' '.join([lem1,lem2]) not in stopwords,
                     k[0] not in TAGS,
                     k[1] not in TAGS,
                     ])):
                insert(temp,k,v)
        else:
            lem1 = k[0]
            lem2 = k[1]
            lem3 = k[2]
            if stem_flag:
                lem1 = stemAugmented(k[0],stemmer=stemmer)
                lem3 = stemAugmented(k[2],stemmer=stemmer)
            if (all([lem1 not in stopwords,
                     k[0] not in TAGS,
                     k[1] != PUNCT_TAG,          #NO CONSIDERADO EN WORDCLOUD
                     lem3 not in stopwords,
                     ' '.join([lem1,lem2,lem3]) not in stopwords,
                     k[2] not in TAGS,
                     ])):
                insert(temp,k,v)

    return temp


def getLemmaToken_dict(sentence_list,stopwords,option='all',stemmer=span_stemmer):
    """ input : sentence_list: lista de oraciones, estructura dada por option
                option: | all: lista de oraciones | bydoc : lista de docs
                stemmer : funcion encargada del stemming (span_stemmer o eng_stemmer)
        output: diccionario de la forma <lemma> : <token>
        description: crea diccionario para mapear lemma con palabra original mas frecuente en corpus de sentence_list
    """

    (ug,bg,tg) = countNgrams(sentence_list,option=option)

    ug = filterNgrams(ug,stopwords,ngrams=1, stem_flag = True,stemmer=stemmer)
    bg = filterNgrams(bg,stopwords,ngrams=2, stem_flag = True,stemmer=stemmer)
    tg = filterNgrams(tg,stopwords,ngrams=3, stem_flag = True,stemmer=stemmer)

    ug_temp = [(v,k) for (k,v) in ug.iteritems()]
    ug_temp.sort(reverse=True)
    bg_temp = [(v,k) for (k,v) in bg.iteritems()]
    bg_temp.sort(reverse=True)
    tg_temp = [(v,k) for (k,v) in tg.iteritems()]
    tg_temp.sort(reverse=True)

    lemmas = {}
    for kk in ug_temp:
        k = kk[1]
        v = kk[0]

        lem = stemAugmented(k,stemmer=stemmer)
        if lem not in lemmas:
            lemmas[lem] = k

    for kk in bg_temp:
        k = kk[1]
        v = kk[0]

        lem = stemAugmented(k,stemmer=stemmer)
        if lem not in lemmas:
            lemmas[lem] = k

    for kk in tg_temp:
        k = kk[1]
        v = kk[0]

        lem = stemAugmented(k,stemmer=stemmer)
        if lem not in lemmas:
            lemmas[lem] = k

    return lemmas


def processTFIDF(Sentences, lemma_token, stopwords, ngrams=1, stem_flag=False,stemmer=span_stemmer):
    """ input: Sent,Vocab, | stowords list | ngram n | stemmed comparison with stopwords
        output: ngram dict with TFIDF scores
        description: generaliza procedimiento TFIDF para ngramas
    """
    TermDocMatrix = {}
    termDocFreq = {}
    tokens_per_doc = []
    NG_total = countNgrams(Sentences, ngrams, option="bydoc")

    NUM_DOCS = len(Sentences)

    for (k,v) in NG_total.iteritems():
        TermDocMatrix[k] = [0 for i in range(NUM_DOCS)]
        termDocFreq[k] = 0

    for (i,doc) in enumerate(Sentences):
        L = 0
        NG = countNgrams(doc, ngrams, option='all')
        for (w,f) in NG.iteritems():
            TermDocMatrix[w][i] = f
            termDocFreq[w] = termDocFreq[w] + 1
            L = L + f
        tokens_per_doc.append(L)

    # TF_IDF Normalize
    for (token,v) in TermDocMatrix.iteritems():
        TermDocMatrix[token] = [ TF_IDF(TF,tokens_per_doc[i],termDocFreq[token],NUM_DOCS) for (i,TF) in enumerate(v)]
        sums = 0
        for i in range(NUM_DOCS):
            sums = sums + TermDocMatrix[token][i]

        TermDocMatrix[token] = sums

    # Filtrado
    res = {}
    TermDocMatrix = filterNgrams(TermDocMatrix,stopwords,ngrams,stem_flag,stemmer=stemmer)
    for (k,v) in TermDocMatrix.iteritems():
        lemma = ""
        if stem_flag:
            lemma = stemAugmented(k,stemmer=stemmer)
        else:
            lemma = ' '.join(k)
        insert(res,lemma_token[lemma],v)

    return res


################################################################################################################################################################################
###                                                         CONTEO DE NGRAMAS - GENERACION DE LISTA DE TAGS

def countNgrams(sentences_list,ngrams = 0,option="all"):
    """ input:  lista de oraciones
                ngrams: | 0 : uni,bi y trigrams
                          1: only unigrams | 2: only bigrams | 3: only trigrams
                option: | all : lista de oraciones | bydoc : lista de docs
        output: diccionarios (Unigrams,Bigrams,Trigrams) segun |ngrams|
        desc:   cuenta los Unigrams, Bigramas y Trigramas en todas las oraciones
    """
    unigram = {}
    bigram = {}
    trigram = {}
    if option=="all":
        for sentence in sentences_list:
            L = len(sentence)
            # Unigrams
            if ngrams == 0 or ngrams == 1:
                for i in range(L):
                    insert(unigram,sentence[i])

            # Bigrams
            if ngrams == 0 or ngrams == 2:
                for i in range(1,L):
                    insert(bigram,(sentence[i-1],sentence[i]))

            # Trigrams
            if ngrams == 0 or ngrams == 3:
                for i in range(2,L):
                    insert(trigram,(sentence[i-2],sentence[i-1],sentence[i]))
        if ngrams == 0 or ngrams == 1:
            insert(unigram,START_TAG,len(sentences_list)*2)
    else:
        for doc in sentences_list:
            for sentence in doc:
                L = len(sentence)
                # Unigrams
                if ngrams == 0 or ngrams == 1:
                    for i in range(L):
                        insert(unigram,sentence[i])

                # Bigrams
                if ngrams == 0 or ngrams == 2:
                    for i in range(1,L):
                        insert(bigram,(sentence[i-1],sentence[i]))

                # Trigrams
                if ngrams == 0 or ngrams == 3:
                    for i in range(2,L):
                        insert(trigram,(sentence[i-2],sentence[i-1],sentence[i]))
            if ngrams == 0 or ngrams == 1:
                insert(unigram,START_TAG,len(doc)*2)

    if ngrams == 1:
        return unigram
    elif ngrams == 2:
        return bigram
    elif ngrams == 3:
        return trigram
    else:
        return (unigram,bigram,trigram)


def getCount_FreqbyDoc(sentence_list,lemma_token,stopwords,ngrams=1,stem_flag = False,stemmer=span_stemmer):
    """ input : sentence_list : lista de docs (IMPORTANTE)
                lemma_token : dict <lemma> : <most frequent token>
                stopwords: lista total de stopwords
                ngrams: uni,bi o trigrams
                stem_flag: contar por lemmas
    """
    # Si sentence_list es generada por 'all', lista de oraciones y no de documentos
    if len(sentence_list) > 0 and type(sentence_list[0]) == str:
        print("Lista de Oraciones debe ser generada por documentos <option='bydoc'>")
        return None

    if stem_flag:
        res = {}
        for doc in sentence_list:
            NG = countNgrams(doc,ngrams,option = 'all')
            NG = filterNgrams(NG,stopwords,ngrams,stem_flag,stemmer=stemmer)

            lemmas = [stemAugmented(ng,stemmer=stemmer) for ng in NG.keys()]
            lemmas = list(set(lemmas))
            for lemma in lemmas:
                insert(res,lemma_token[lemma])
    else:
        NG = countNgrams(sentence_list, ngrams,option = 'bydoc')
        res = filterNgrams(NG,stopwords,ngrams,stem_flag,stemmer=stemmer)
    return res


def getCount_TotalFreq(sentence_list,lemma_token,stopwords,ngrams=1,stem_flag = False,stemmer=span_stemmer):
    """ input : sentence_list : lista de lista de str | lista de str
                lemma_token : dict <lemma> : <most frequent token>
                stopwords: lista total de stopwords
                ngrams: uni,bi o trigrams
                stem_flag: contar por lemmas
                stemmer: stemmer function
    """
    # Si sentence_list es generada por 'all', lista de oraciones y no de documentos
    if len(sentence_list) == 0:
        print("Lista de Oraciones Vacia")
        return None
    res = {}
    if type(sentence_list[0][0]) == str:
        ## opcion all
        NG = countNgrams(sentence_list,ngrams,option = 'all')
    elif type(sentence_list[0][0]) == list:
        # opcion 'bydoc'
        NG = countNgrams(sentence_list,ngrams,option = 'bydoc')

    NG = filterNgrams(NG,stopwords,ngrams,stem_flag,stemmer=stemmer)

    for (k,v) in NG.iteritems():
        lemma = ""
        if stem_flag:
            lemma = stemAugmented(k,stemmer=stemmer)
        else:
            lemma = ' '.join(k)
        insert(res,lemma_token[lemma],v)
    return res


def counterJobData(output_prefix='', extracting_option='bydoc',career_tags=[],filters=[NUM_TAG],stemming=False,score=TFIDF,
                   results_dir='' , join=True, text=True, json=False):
    """
    :param output_prefix : prefijo del nombre de los archivos
    :param extract_option: (str)
            bydoc : estructura data en lista de documentos, es decir, lista de lista de oraciones
            all   : estructura data en lista de oraciones
    :param career_tags: (str list)identificadores de carreras a incluir en cuenta.
    :param filters : (str list) filtros de ruido en palabras (NUM, RARE)
    :param stemming: (bool) comparacion con stemming y normalizado unicode
    :param score : (str)
                TF_IDF : ranking de palabras por TF_IDF
                FREQ_DOC :  ranking de palabras por frecuencia documental (1 vez / documento)
    :param results_dir: (str) path de carpeta donde crear los archivos de resultados
    """
    corpus = getPreprocessedData(option = extracting_option,career_tags=career_tags)
    V = {}
    S = []
    (V,S) = getVocabSentences(corpus, option = extracting_option,stem_flag=False, filters=filters)

    print("Vocabulary and Sentences extracted")
    # Get stopwords
    stopwords = getStopWords(stemming=True)
    ug = {}
    bg = {}
    tg = {}
    if score == TFIDF:
        lemma_token = getLemmaToken_dict(S, stopwords, option=extracting_option)
        # Score TF IDF
        if extracting_option == 'all':
            print("Debe llamar funcion con opcion de extracion 'bydoc'.")
            return None
        else:
            ug = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=1,stem_flag=stemming)
            bg = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=2,stem_flag=stemming)
            tg = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=3,stem_flag=stemming)
    else:
        # get Lemma - most freq token dictionary
        lemma_token = getLemmaToken_dict(S, stopwords, option=extracting_option)
        if score == FREQ_DOC:
            # Score por frecuencia por documento
            ug = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 1, stem_flag = True)
            bg = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 2, stem_flag = True)
            tg = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 3, stem_flag = True)
        else:
            # Score por frecuencia total
            ug = getCount_TotalFreq(S,lemma_token,stopwords, ngrams = 1, stem_flag = True)
            bg = getCount_TotalFreq(S,lemma_token,stopwords, ngrams = 2, stem_flag = True)
            tg = getCount_TotalFreq(S,lemma_token,stopwords, ngrams = 3, stem_flag = True)

    print("Ngram counting done!")

    # prepare to write
    min_freq = -5
    uni = [(v,k) for (k,v) in ug.iteritems() if v > min_freq]
    bb  = [(v,k) for (k,v) in bg.iteritems() if v > min_freq]
    tt  = [(v,k) for (k,v) in tg.iteritems() if v > min_freq]
    uni.sort(reverse=True)
    bb.sort(reverse=True)
    tt.sort(reverse=True)

    RESULTS_DIR = results_dir
    if score == TFIDF:
        csv_uni = os.path.join(RESULTS_DIR,output_prefix + '_tfidf_unigrams.csv')
        csv_bi  = os.path.join(RESULTS_DIR,output_prefix + '_tfidf_bigrams.csv')
        csv_tri = os.path.join(RESULTS_DIR,output_prefix + '_tfidf_trigrams.csv')
    else:
        csv_uni = os.path.join(RESULTS_DIR,output_prefix + '_freq_unigrams.csv')
        csv_bi  = os.path.join(RESULTS_DIR,output_prefix + '_freq_bigrams.csv')
        csv_tri = os.path.join(RESULTS_DIR,output_prefix + '_freq_trigrams.csv')

    # write in
    out  = open(csv_uni,'w')
    outB = open(csv_bi,'w')
    outT = open(csv_tri,'w')

    out.write("token,score\n")
    outB.write("v,u,score\n")
    outT.write("w,v,u,score\n")

    print("Writing ...")

    if score == TFIDF:
        temp = ''
        for k in uni:
            temp = "%s,%.4f\n" % (k[1],k[0])
            out.write(temp.encode('utf-8'))
        for k in bb:
            temp = "%s,%s,%.4f\n" %(k[1][0],k[1][1],k[0])
            outB.write(temp.encode('utf-8'))
        for k in tt:
            temp = "%s,%s,%s,%.4f\n" %(k[1][0],k[1][1],k[1][2],k[0])
            outT.write(temp.encode('utf-8'))
    else:
        temp = ''
        for k in uni:
            temp = "%s,%d\n" % (k[1],k[0])
            out.write(temp.encode('utf-8'))
        for k in bb:
            temp = "%s,%s,%d\n" %(k[1][0],k[1][1],k[0])
            outB.write(temp.encode('utf-8'))
        for k in tt:
            temp = "%s,%s,%s,%d\n" %(k[1][0],k[1][1],k[1][2],k[0])
            outT.write(temp.encode('utf-8'))

    writeAddOutput(ug,bg,tg,results_dir=results_dir,output_prefix=output_prefix,score=score,join=join,text=text,json=json)
    print("Done!!")


def counterSyllabusData(extracting_option='all',data_path='',stopwords_path='',results_dir='',
                        filters=[NUM_TAG],stemming=True,score = TFIDF, language='spanish'):
    """
    """
    corpus = getSyllabusData(option = extracting_option,path=data_path)
    V = {}
    S = []
    (V,S) = getVocabSentences(corpus, option = extracting_option,stem_flag=False, filters=filters)
    # Get stopwords
    stopwords = []
    stemmer = ""
    if language == 'spanish':
        stopwords = getStopWords(stemming=True,syllabus=True)
        stemmer = span_stemmer
    else:
        stopwords = getStopWordsEnglish(stemming=True,path = stopwords_path)
        stemmer = eng_stemmer


    if score == TFIDF:
        lemma_token = getLemmaToken_dict(S, stopwords, option=extracting_option,stemmer=stemmer)

        # Score TF IDF
        if extracting_option == 'all':
            print("Debe llamar funcion con opcion de extracion 'bydoc'.")
            return None
        else:
            ug = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=1,stem_flag=stemming,stemmer=stemmer)
            bg = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=2,stem_flag=stemming,stemmer=stemmer)
            tg = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=3,stem_flag=stemming,stemmer=stemmer)
    else:
        # get Lemma - most freq token dictionary
        lemma_token = getLemmaToken_dict(S, stopwords, option=extracting_option,stemmer=stemmer)

        if score == FREQ_DOC:
            # Score por frecuencia
            ug = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 1, stem_flag = stemming,stemmer=stemmer)
            bg = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 2, stem_flag = stemming,stemmer=stemmer)
            tg = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 3, stem_flag = stemming,stemmer=stemmer)
        else:
            # FREQ_TOTAL
            ug = getCount_TotalFreq(S,lemma_token,stopwords, ngrams = 1, stem_flag = stemming,stemmer=stemmer)
            bg = getCount_TotalFreq(S,lemma_token,stopwords, ngrams = 2, stem_flag = stemming,stemmer=stemmer)
            tg = getCount_TotalFreq(S,lemma_token,stopwords, ngrams = 3, stem_flag = stemming,stemmer=stemmer)

    # prepare to write
    min_freq = 0
    uni = [(v,k) for (k,v) in ug.iteritems() if v > min_freq]
    bb  = [(v,k) for (k,v) in bg.iteritems() if v > min_freq]
    tt  = [(v,k) for (k,v) in tg.iteritems() if v > min_freq]
    uni.sort(reverse=True)
    bb.sort(reverse=True)
    tt.sort(reverse=True)

    RESULTS_DIR = results_dir
    if score == TFIDF:
        csv_uni = os.path.join(RESULTS_DIR,'tfidf_unigrams.csv')
        csv_bi  = os.path.join(RESULTS_DIR,'tfidf_bigrams.csv')
        csv_tri = os.path.join(RESULTS_DIR,'tfidf_trigrams.csv')
    else:
        csv_uni = os.path.join(RESULTS_DIR,'freq_unigrams.csv')
        csv_bi  = os.path.join(RESULTS_DIR,'freq_bigrams.csv')
        csv_tri = os.path.join(RESULTS_DIR,'freq_trigrams.csv')

    # write in
    out  = open(csv_uni,'w')
    outB = open(csv_bi,'w')
    outT = open(csv_tri,'w')

    out.write("token,score\n")
    outB.write("v,u,score\n")
    outT.write("w,v,u,score\n")

    if score == TFIDF:
        for k in uni:
            out.write("%s,%.4f\n" % (k[1],k[0]))
        for k in bb:
            outB.write("%s,%s,%.4f\n" %(k[1][0],k[1][1],k[0]))
        for k in tt:
            outT.write("%s,%s,%s,%.4f\n" %(k[1][0],k[1][1],k[1][2],k[0]))
    else:
        # FREQ_DOC & FREQ_TOTAL
        for k in uni:
            out.write("%s,%d\n" % (k[1],k[0]))
        for k in bb:
            outB.write("%s,%s,%d\n" %(k[1][0],k[1][1],k[0]))
        for k in tt:
            outT.write("%s,%s,%s,%d\n" %(k[1][0],k[1][1],k[1][2],k[0]))


def writeTagLists(data=[],filename ='',extracting_option='bydoc',filter_option=PURE, career_tags=[],filters=[NUM_TAG],stemming=True,score=TFIDF,samples = 1000000,
                  join=True,text=True,json=True):
    """ input:
                data : lista de vector hash sobre el cual correr el contador
                filename : prefijo del nombre de los output files
            extract_option: (str)
                bydoc : estructura data en lista de documentos, es decir, lista de lista de oraciones
                all   : estructura data en lista de oraciones
            filter_option : (str)
                PURE : obtiene solo A - sin intersecciones
                JOIN : obtiene todo A
            career_tags: (str list)identificadores de carreras a incluir en cuenta.
            filters : (str list) filtros de ruido en palabras (NUM, RARE)
            stemming: (bool) comparacion con stemming y normalizado unicode
            score : (str)
                TF_IDF : ranking de palabras por TF_IDF
                FREQ_DOC :  ranking de palabras por frecuencia documental (1 vez / documento)
            results_dir: (str) path de carpeta donde crear los archivos de resultados
            samples : numero de muestras aleatoreamente tomadas de los trabajos filtrados
    """
    corpus = []

    print "Preprocessing step..."
    if len(data)>0:
    # vector hash pre-definido
        corpus = preprocessVectorHashData(data,option=extracting_option)
    else:
     # configuracion previa
        if filter_option == PURE:
          data = []
          for career in career_tags:
                temp = getOnlyCareer(career=career)
                data.extend(temp)
          data = list(set(data))
          data = sampleVectorHash(data,samples = samples)
          corpus = preprocessVectorHashData(data,option=extracting_option)
        else:
          corpus = getPreprocessedData(option = extracting_option,career_tags=career_tags, max_docs = samples)

    V = {}
    S = []
    print "Tokenization..."
    (V,S) = getVocabSentences(corpus, option = extracting_option,stem_flag=False, filters=filters)
    # Get stopwords
    stopwords = getStopWords(stemming=True)

    print "Counting..."
    if score == TFIDF:
        lemma_token = getLemmaToken_dict(S, stopwords, option=extracting_option)

        # Score TF IDF
        if extracting_option == 'all':
            print "Debe llamar funcion con opcion de extracion 'bydoc'."
            return None
        else:
            ug = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=1,stem_flag=stemming)
            bg = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=2,stem_flag=stemming)
            tg = processTFIDF(Sentences=S,lemma_token=lemma_token,stopwords=stopwords,ngrams=3,stem_flag=stemming)
    else:
        # get Lemma - most freq token dictionary
        lemma_token = getLemmaToken_dict(S, stopwords, option=extracting_option)

        # Score por frecuencia
        ug = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 1, stem_flag = True)
        bg = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 2, stem_flag = True)
        tg = getCount_FreqbyDoc(S,lemma_token,stopwords, ngrams = 3, stem_flag = True)

    # prepare to write
    min_freq = -5
    uni = [(v,k) for (k,v) in ug.iteritems() if v > min_freq]
    bb  = [(v,k) for (k,v) in bg.iteritems() if v > min_freq]
    tt  = [(v,k) for (k,v) in tg.iteritems() if v > min_freq]
    uni.sort(reverse=True)
    bb.sort(reverse=True)
    tt.sort(reverse=True)

    print "Writing results..."
    # crear carpeta de resultados, si no existe
    results_dir=os.path.join(UTIL_DIR,'tag_lists')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if score == TFIDF:
        csv_uni = os.path.join(results_dir,filename + '_tfidf_unigrams.csv')
        csv_bi  = os.path.join(results_dir,filename + '_tfidf_bigrams.csv')
        csv_tri = os.path.join(results_dir,filename + 'tfidf_trigrams.csv')
    else:
        csv_uni = os.path.join(results_dir,filename + '_freq_unigrams.csv')
        csv_bi  = os.path.join(results_dir,filename + '_freq_bigrams.csv')
        csv_tri = os.path.join(results_dir,filename + '_freq_trigrams.csv')

    # write in
    out  = open(csv_uni,'w')
    outB = open(csv_bi,'w')
    outT = open(csv_tri,'w')

    out.write("token,score\n")
    outB.write("v,u,score\n")
    outT.write("w,v,u,score\n")

    if score == TFIDF:
        for k in uni:
            out.write("%s,%.4f\n" % (k[1],k[0]))
        for k in bb:
            outB.write("%s,%s,%.4f\n" %(k[1][0],k[1][1],k[0]))
        for k in tt:
            outT.write("%s,%s,%s,%.4f\n" %(k[1][0],k[1][1],k[1][2],k[0]))
    else:
        for k in uni:
            tempu2 = u'%s,%d\n' % (k[1],k[0])
            out.write(tempu2.encode("utf-8"))
            #out.write("%s,%d\n" % (k[1],k[0]))
        for k in bb:
                tempb2 = "%s,%s,%d\n" %(k[1][0],k[1][1],k[0])
                outB.write(tempb2.encode("utf-8"))
            #outB.write("%s,%s,%d\n" %(k[1][0],k[1][1],k[0]))
        for k in tt:
            tempt2 = "%s,%s,%s,%d\n" %(k[1][0],k[1][1],k[1][2],k[0])
            outT.write(tempt2.encode("utf-8"))
            #outT.write("%s,%s,%s,%d\n" %(k[1][0],k[1][1],k[1][2],k[0]))

    # tuple -> json
    writeAddOutput(ug,bg,tg,filename=filename,results_dir=results_dir,handfiltered=False,score=score,join=join,text=text,json=json)


#############################################################################################
from multiprocessing import Pool
from functools import partial
PARALLELIZE = False
NPROCESSORS = 3

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


def getCareers_wrapper(params=[]):
    line,patrones,spec_case_patterns,fileByIdent,careersFromPatterns = params

    def getCareers( line=line, patrones=patrones, spec_case_patterns=spec_case_patterns, fileByIdent=fileByIdent,
                    careersFromPatterns=careersFromPatterns):
        # Variables para identificacion de carreras
        carreras = set()
        found_areas = set()
        cuerpo_temp = []
        ing_post = False       # flag si posiblemente es un aviso de ingenieria

        # preprocesado de casos especiales (ver header)
        for i,pat in enumerate(spec_case_patterns):
            line = pat.sub(spec_case_subs[i],line)
        if spec_case_patterns[0].search(line):
            ing_post = True                         # aviso pide ingenieros
        cuerpo_temp.append(line)
        # busqueda de patrones
        for (i,pattern) in enumerate(patrones):
            for car in careersFromPatterns(line,patrones,i):
                carreras.add(car.lower())

        cuerpo = list(cuerpo_temp)
        carreras = list(carreras)       # set -> list
        origc = list(carreras)  # copia explicita
        carreras = [stemAugmented(line).strip(' ') for line in carreras]
        carreras = list(set(carreras))

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
                            found_areas.add('quimica')       # ing quimica
                        else:
                            found_areas.add('quimico')       # quimico puro
                    else:                                       # caso general
                        found_areas.add(fileByIdent[car])
        return found_areas