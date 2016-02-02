'''
Created on Apr 21, 2014

@author: ronald
'''
import math
from counter import *

LOG = 'logaritmic'
SQRT = 'sqrt'
TRUNC = 'truncation'



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


if __name__=="__main__":

    # results & source dirs
    source_mit        = os.path.join(UTIL_DIR,'syllabus/results/MIT')
    source_uni        = os.path.join(UTIL_DIR,'syllabus/results/UNI')

    
    """ Cambia handfiltered = True cuando tengas la lista filtrada a mano
        *formato nombre de lista filtrada a mano:
              "<ug,bg,tg>_handfiltered.csv"
    """

    writeOutput(source_mit,source_mit,handfiltered=False,score=TFIDF,join=True,json=True)
    writeOutput(source_uni,source_uni,handfiltered=False,score=TFIDF,join=True,json=True)

    
    results_mercado = os.path.join(UTIL_DIR,'syllabus/mercado')
    writeOutput(results_mercado,results_mercado,handfiltered=False,score=TFIDF, join=True,json=True)
    
    
    """
    # solo para joined _ raw data
    OUTPUT_DIR = ''
    open(OUTPUT_DIR + 'joined_rawdata.txt' ,'w').write(textJ) 
    """