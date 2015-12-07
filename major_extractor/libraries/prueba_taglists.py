from utilities import *

############ Seteo de parametros


# carreras a filtrar. Utilize solo minusculas sin tildes ni enhes

"""
 Que hace esta funcion exactamente?
 1. Filtra las carreras presentes en career_tags
 2. Estructura la data en oraciones. Cada oracion es una lista de palabras
 3. Divide las oraciones y cuenta unigramas, bigramas y trigramas
 4. Asigna un score a cada ngrama de acuerdo al parametro <score>:
       TFIDF: score estadistico. Tiene en cuenta la frecuencia relativa del ngrama con respecto a cada documento
              y la frecuencia absoluta con respecto de todo el corpus
       FREQ_DOC : Cuenta en cuantos documentos aparece cada ngrama
       FREQ_TOTAL : Cuenta cuantas veces aparece el ngrama en total en toda la data.
 5. Escribe los resultados en la carpeta 'tag_lists/', archivos de la forma:
      freq_<ngrama>.csv  -> para score tipo frecuencia
      tfidf_<ngrama>.csv -> para score TFIDF

 6. Parametros de output adicional
    Se puede obtener output adicional:
      <ngrama>.txt       -> de la forma <tag> : <score>
      <ngrama>.json

    Controlable mediante los parametros:
    - text: Si True, escribe .txt para cada lista
    - json: Si True, escribe .json para cada lista
    - join : Combina los unigramas, bigramas y trigramas en una sola lista.
            Si True, escribe join.txt y join.json.
"""

DATA_SOURCE = 'ing'
data = readHashVector('vh_'+DATA_SOURCE)

writeTagLists(data=data,filename ='ing',stemming=False,score=FREQ_DOC)