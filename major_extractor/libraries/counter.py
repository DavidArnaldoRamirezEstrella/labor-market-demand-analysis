from utilities import *
import math
import re
import json
from kombu.async.hub import Stop


if __name__ == "__main__":
    """
    ###   SCRIPT CONTEO EN SYLLABUS
    EXTRACT_OPTION = 'bydoc'
    results_dir = os.path.join(UTIL_DIR,'syllabus/mercado')
    career_tags = ['ingenieria mecanica']
    
    
    counterJobData(extracting_option=EXTRACT_OPTION,career_tags=career_tags, filters=[NUM_TAG],
                   stemming = True, score = FREQ_DOC,results_dir = results_dir)

    EXTRACT_OPTION = 'bydoc'
    syllabus_stopwords = os.path.join(UTIL_DIR,'stopwords_english/english')
    syllabus_data      = os.path.join(UTIL_DIR,'syllabus/MIT')
    results_mit        = os.path.join(UTIL_DIR,'syllabus/results/MIT')
    results_uni        = os.path.join(UTIL_DIR,'syllabus/results/UNI')


    # MIT SYLLABUS
    counterSyllabusData(extracting_option=EXTRACT_OPTION,data_path=syllabus_data,stopwords_path=syllabus_stopwords,results_dir=results_mit,
                        filters=[NUM_TAG],stemming = True, score = TFIDF,language='english')


    syllabus_data      = os.path.join(UTIL_DIR,'syllabus/UNI')
    # UNI SYLLABUS
    counterSyllabusData(extracting_option=EXTRACT_OPTION,data_path=syllabus_data,results_dir=results_uni,
                        filters=[NUM_TAG],stemming = True, score = TFIDF,language='spanish')
    """

    ###   SCRIPT CONTEO EN BASE DE DATOS
    results_dir = os.path.join(UTIL_DIR,'PARTY_O_KE_AZE')
    results_dir = UTIL_DIR
    career_tags = ['mecanica',       # --> extraera tod0 ingenieria
                   ]
    output_name = 'KE_AZE'  #  prefijo de nombre de archivos con resultados: prex_freq_unigram.csv ...

    counterJobData(output_prefix = output_name,career_tags = career_tags,
                   score = FREQ_DOC,results_dir = results_dir,
                   join = True, text = True, json = False)
    #:join: crea archivos aparte <join> con ug + bg + tg
    # text: crea txt con forma para Wordle
    #:json: crea archivos .json con formato para wordcloud.js
