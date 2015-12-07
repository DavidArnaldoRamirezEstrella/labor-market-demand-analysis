import os, sys
import json
import copy
import numpy
import pdb

temp = os.path.dirname(os.path.abspath(__file__))
pattern_dir = os.path.join(temp,'nlp scripts/patterns')

sys.path.append(pattern_dir)
#from regex_patterns import *
from regex_patterns_no_tec import *

################################################################################################
## Datos externos
ident_tecnico = strip_encode(leer_tags(open(os.path.join(IDENTIFIER_STEM_DIR,'tecnico'))), False)
tec = stemAugmented('tecnico',degree=1)
ident_tecnico = [w for w in ident_tecnico if w != tec]

fileByIdent = getDocnameByCareer()
identByFile = {}
for k,v in fileByIdent.items():
    if v not in identByFile:
        identByFile[v] = []
    identByFile[v].append(k)


################################################################################################
# docname : {docname : true name}
nameByFile = json.loads(unicode(open('ident_names.json').read().decode('utf-8')))
temp = {}
for (file,name) in nameByFile.items():
    temp[file.strip(' ')] = name.strip(' ')
nameByFile = dict(temp)
################################################################################################

idByfile = {}
fileById = {}
idByName = {}
num_areas = 0

for (file,name) in nameByFile.items():
    idByfile[file] = num_areas
    idByName[name] = num_areas
    fileById[num_areas] = file
    num_areas += 1


###################################################################################################
# CASOS ESPECIALES - para que no los detecten/separen los patrones
# Mecanica Electrica
ident_mec_elect = [w for w in fileByIdent.keys() if fileByIdent[w]=='mecanica_electrica']
ident_mec_elect = '|'.join(addSuffixRegex(ident_mec_elect))

# Tecnologia Medica
ident_tec_med = [w for w in fileByIdent.keys() if fileByIdent[w]=='tecnologia_medica']
ident_tec_med = '|'.join(addSuffixRegex(ident_tec_med))

spec_case_patterns = [
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
##############################################################################################
##############################################################################################
###################################################################################################
##                                  SETUP DE MUESTREO
MUESTREO = True
UPDATE_VH = False
UPDATE_ALL = False     # hace vector hash de todas las carreras
UPDATE_DB = False     # solo cuando se carga nueva data a la DB | genera vh_all

##############################################################################################
# DEFINIR VECTOR HASH FUENTE | ALL para total
DATA_SOURCE = 'all'

carrera_muestreo = 'ing'
cm_list=[
    'ing',
]
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
vh_total = {}
if UPDATE_ALL:
    for cm in idByfile.keys():
        vh_total[cm] = []
else:
    for cm in cm_list:
        vh_total[cm] = []


num_posts = 100
total = 0
contador_jobs = 0

###################################################################################################
#                                   SETUP DE CONTEO CIRCLE PLOT Y TREEMAP
COUNT_ALL = False				# TRUE PARA ACTUALIZAR CARRERAS.JSON  Y  ADJMATRIX.JSON
AdjMatrix = numpy.zeros([num_areas,num_areas])
TotalSize = numpy.zeros(num_areas)

def sorter(T,sizeById, idByName):
    if "children" not in T:
        T["size"] = int(sizeById[ idByName[T["name"].strip(' ')] ])
        return T["size"]

    children = T["children"]
    temp = []
    _total = 0
    for child in children:
        subt_sum = sorter(child,sizeById, idByName)
        _total += subt_sum
        temp.append(tuple([subt_sum,child]))
    temp.sort(reverse=True)
    T["children"] = [k[1] for k in temp]
    return _total


def getSortedLeaves(T, V):
  if "children" not in T:
    V.append(T["name"])
    return
  for child in T["children"]:
    getSortedLeaves(child,V)

###################################################################################################
#                                   SETUP DE CONTEO STACKED BARS
STACKED_BARS_COUNT = False
sb_Count = [[] for i in range(num_areas)]

###################################################################################################
##                                                                  OBTENCION DE DATA

if UPDATE_DB:
    #                           dd-mm-yyyy
    updateVHfromDB(query_date=['01-06-2014',])        # solo cuando se carga nueva data


# Normalizando data en forma de vector hash
data = []
if UPDATE_VH or COUNT_ALL or UPDATE_ALL:
    data = readHashVector('vh_'+DATA_SOURCE)
else:
    data = readHashVector('vh_' + carrera_muestreo)
    if MUESTREO:
        data = sampleVectorHash(data,samples = num_posts)


print "%% Data loaded"
print "Source data len:",len(data)
######################################################################################################################################################################################################
######################################################################################################################################################################################################
#                                                            HILO PRINCIPAL
#patrones = compilePatterns()

for post in data:
    det_pk = post[0]
    desc_pk = post[1]
    job = Details.objects.filter(pk=det_pk)[0]
    desc = job.description_set.filter(pk=desc_pk)[0]

    pdb.set_trace()

    # armar texto
    title = strip_encode([job.title])
    title.append('')

    description = strip_encode([w for w in desc.description.split('\n')])

    cuerpo = list(title) # copia explicita
    cuerpo.extend(description)

    # Variables para identificacion de carreras
    found_areas = set()
    cuerpo_temp = []
    ing_post = False       # flag si posiblemente es un aviso de ingenieria

    for line in cuerpo:
        # preprocesado de casos especiales (ver header)
        for i,pat in enumerate(spec_case_patterns):
            line = pat.sub(spec_case_subs[i],line)
        if spec_case_patterns[0].search(line):
            ing_post = True                         # aviso pide ingenieros
        cuerpo_temp.append(line)

    cuerpo = list(cuerpo_temp)

    for career in cm_list:
        if searchIdentifier(cuerpo,identByFile[career]):
            found_areas.add(career)

    if len(found_areas)==0:
        found_areas.add('otros')
    
    found_areas = list(found_areas)

    ##########################################################
    # CONTEO DE ARISTAS PARA CIRCLE PLOT
    if COUNT_ALL:
        if len(found_areas)>1:
            for (i,area) in enumerate(found_areas):
                u = idByfile[found_areas[i]]
                TotalSize[u] += 1
                for j in range(i+1,len(found_areas)):
                    v = idByfile[found_areas[j]]
                    AdjMatrix[u][v] += 1
                    AdjMatrix[v][u] += 1
        else:
            u = idByfile[found_areas[0]]
            AdjMatrix[u][u] += 1
            TotalSize[u] += 1

    ##########################################################
    # CONTEO DE PALABRAS PARA BARRAS APILADAS
    if STACKED_BARS_COUNT:
        term_freq = {}
        updateTermFreq(cuerpo,term_freq)
        total_words = sum(term_freq.values())
        for area in found_areas:
            sb_Count[idByfile[area]].append(total_words)

    ##########################################################
    # UPDATE DE VECTOR HASH DE CARRERAS SELECCIONADAS
    if UPDATE_VH or UPDATE_ALL:
        for cm in vh_total.keys():
            if cm in found_areas:
                vh_total[cm].append([job.hash,desc.hash])

    ##########################################################
    # MUESTREO DE CARRERA MENCIONADA
    if MUESTREO and not UPDATE_VH and not UPDATE_ALL:
        print '''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
(--   %d  ---)
%s
::: Carreras
%s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
''' % (contador_jobs,'\n'.join(cuerpo),found_areas)
        contador_jobs += 1
    ##################################################################

    if total % 1000==0:
            print 'Total --> ',total
    total += 1
# ENDFOR
####################################################################################################################################

if UPDATE_VH or UPDATE_ALL:
    for cm in vh_total.keys():
        writeHashVector(vh_total[cm],"vh_" + cm)

    print "Vector Hash escritos"

####################################################################################################################################
if COUNT_ALL:
    print 'Campo de trabajo, Numero de trabajos'

    # debug total count
    for i in range(num_areas):
        print "%s : %d" % (fileById[i],TotalSize[i])

    #pdb.set_trace()
    output = []
    unw_output = []
    for i in range(num_areas):
        u = fileById[i]
        item = {}
        item["name"] = nameByFile[u]
        item["size"] = TotalSize[i]

        # Self loop edge: only career
        only_carrer = AdjMatrix[i][i]
        self_loop = {}
        item["imports"] = []

        #UNweighted graph
        unw_item = copy.deepcopy(item)
        unw_item["imports"] = []

        for j in range(num_areas):
            v = fileById[j]
            if AdjMatrix[i][j] > 0:
                if i!=j:
                    temp = dict(name=nameByFile[v], weight=AdjMatrix[i][j])
                    item["imports"].append(temp)
                else:
                    temp = dict(name=nameByFile[v], weight=only_carrer)
                    self_loop = temp
                unw_item["imports"].append(temp["name"])
        if self_loop == {}:     # caso q no haya carrera_solo
            self_loop= dict(name=item["name"],weight=0)

        item["imports"].append(self_loop)

        output.append(item)
        unw_output.append(unw_item)

    # Actualiza cuentas y orden relativo en carreras json
    tree = json.loads(unicode(open('carreras.json').read().decode('utf-8')))
    suma = sorter(tree, TotalSize, idByName)
    open("carreras.json",'w').write(json.dumps(tree,ensure_ascii=False, encoding='utf-8', indent = 2).encode('utf-8'))

    sorted_leaves = []
    getSortedLeaves(tree,sorted_leaves)

    temp = []
    for name in sorted_leaves:
        for car in output:
            if car["name"] == name:
                temp.append(car)
                break
    output = list(temp)

    temp = []
    for name in sorted_leaves:
        for car in unw_output:
            if car["name"] == name:
                temp.append(car)
                break
    unw_output = list(temp)

    open("adjmatrix.json",'w').write(json.dumps(output,ensure_ascii=False, encoding='utf-8',indent = 2).encode('utf-8'))
    open("unw_adjmatrix.json",'w').write(json.dumps(unw_output,ensure_ascii=False, encoding='utf-8',indent = 2).encode('utf-8'))

    print "ADJM & Carreras actualizadas"

####################################################################################################################################
if STACKED_BARS_COUNT:
    max_Length = 0
    for i,area in enumerate(sb_Count):
        area.sort()
        if len(area)>0:
            max_Length = max(max_Length,area[-1])
    output = []
    for i in range(num_areas):
        u = fileById[i]
        item = {}
        item["name"] = nameByFile[u]
        item["data"] = sb_Count[i]
        output.append(item)

    open("raw_stackedBars.json",'w').write(json.dumps(output,ensure_ascii=False, encoding='utf-8').encode('utf-8'))

    print "----------------------------"
    print "Maxima cantidad de palabras: ",max_Length
