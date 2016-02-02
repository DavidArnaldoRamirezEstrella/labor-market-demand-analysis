import os, sys
import json
import copy
import numpy
import random
from multiprocessing import Pool
import pdb

temp = os.path.dirname(os.path.abspath(__file__))
pattern_dir = os.path.join(temp,'nlp scripts/patterns')

sys.path.append(pattern_dir)
#from regex_patterns import *           # PATRONES UNIV + TECNICO
from regex_patterns_no_tec import *     # PATRONES UNIV (NO TECNICO)

numpy.random.seed(RANDOM)

################################################################################################
## Datos externos
ident_tecnico = strip_encode(leer_tags(open(os.path.join(IDENTIFIER_STEM_DIR,'tecnico'))), False)
tec = stemAugmented('tecnico',degree=1)
ident_tecnico = [w for w in ident_tecnico if w != tec]

fileByIdent = getDocnameByCareer(only_majors=True)

fnt_path = os.path.join(pattern_dir,'frase_no_tecnico')
frase_no_tecnico = strip_encode([ stemAugmented(unicode(line.lower()),degree=1) for line in readLatin(fnt_path).split('\n')])
frase_no_tecnico = set(frase_no_tecnico)
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
ident_mec_elect = []
# Mecanica fluidos
ident_mec_fluidos = []
for ident in fileByIdent.keys():
    w = ident.replace('ing de ','')
    w = w.replace('ing en ','')
    w = w.replace('ing ','')
    if fileByIdent[ident]=='mecanica_electrica':
        ident_mec_elect.append(w)
    if fileByIdent[ident]=='mecanica_fluidos':
        ident_mec_fluidos.append(w)
ident_mec_elect = '|'.join(addSuffixRegex(ident_mec_elect))
ident_mec_fluidos = '|'.join(addSuffixRegex(ident_mec_fluidos))

# Tecnologia Medica
ident_tec_med = [w for w in fileByIdent.keys() if fileByIdent[w]=='tecnologia_medica']
ident_tec_med = '|'.join(addSuffixRegex(ident_tec_med))
sep = '|'.join(separadores)
bett = '|'.join(bet)
spec_case_patterns = [
    re.compile(addSuffixRegex(['ing'])[0] ),
    re.compile(ident_mec_elect),
    re.compile(ident_tec_med),
    re.compile(ident_mec_fluidos),
    re.compile(addSuffixRegex([stemAugmented('redes y comunicaciones')] )[0] ),
    re.compile(addSuffixRegex(['computacion e informatica']             )[0] ),
    re.compile(addSuffixRegex(['logistica y operaciones']               )[0] ),
    re.compile(addSuffixRegex(['operaciones y logistica']               )[0] ),
    re.compile(addSuffixRegex(['sistemas e informacion']                )[0] ),
    re.compile(addSuffixRegex([stemAugmented('medico veterinario')]     )[0] ),
    re.compile(addSuffixRegex([stemAugmented('estudio de abogado')] )[0] ),
    re.compile(r'^educacion\s*:'),
    re.compile(r'\([a-z]\)'),
    re.compile(r'((rubro)|(sector)|(area)|(campo))[es]{,2}\s+((%s)\s+)*[a-z]+(\s*(%s)\s*((%s)\s+)*[a-z]+)*' % (bett,sep,bett)),
    re.compile(r'((rubro)|(sector)|(area)|(campo))[es]{,2}(\s+(%s)\s*)*:\s*((%s)\s+)*[a-z]+(\s*(%s)\s*((%s)\s+)*[a-z]+)*' % (bett,bett,sep,bett)),
    re.compile(r'- ((mina)|(minera)|(mineria))s?'),
]

spec_case_subs = [
    'ing',                      # normalizacion de ident de ingenieria
    'mecanica electrica',
    'medizintechnik',           # tecnologia medica en Aleman xD
    'ing fluidmechanik',            # mecanica de fluidos en Aleman xD + ing para q sea detectado
    'redes Y comunicaciones',
    'computacion E informatica',
    'logistica Y operaciones',
    'operaciones Y logistica',
    'sistemas E informacion',
    'veterinaria',
    'estudio_de_abogado',
    'formacion:',
    '',                         # elimina (a) o (o)
    ' <rubro> ',    # normalizar rubros
    ' <rubro> ',    # normalizar rubros
    'en_la_mina',   # referencia al lugar MINA
]

###################################################################################################
##                                  SETUP DE MUESTREO
MUESTREO = True
UPDATE_VH = False
UPDATE_ALL = False     # hace vector hash de todas las carreras
UPDATE_DB = False     # solo cuando se carga nueva data a la DB | genera vh_all
new_vh = []

DATA_SOURCE = 'ing'

carrera_muestreo = 'otros' # LOL
cm_list=[
    'mecanica'
]

vh_total = {}
if UPDATE_ALL:
    for cm in idByfile.keys():
        vh_total[cm] = []
else:
    for cm in cm_list:
        vh_total[cm] = []

num_posts = 60
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

def parallel_filter(line,index):
#def parallel_filter(line):
    if len(line)==0:
        return set(),set(),set()

    ing_post = False       # flag si posiblemente es un aviso de ingenieria
    carreras=set()
    found_areas=set()
    # preprocesado de casos especiales (ver header)
    for i,pat in enumerate(spec_case_patterns):
        line = pat.sub(spec_case_subs[i],line)
    if spec_case_patterns[0].search(line):
        ing_post = True

    # buscar casos de no tecnico
    spans = re.finditer(r'\btecnic[a-z]+',line.lower())
    breaks = []
    entro = False
    for m in spans:
        ini = m.span()[0]
        pre = m.span()[0]
        for i in xrange(ini-2,-1,-1):
            if not line[i].isalpha() and not line[i].isdigit():
                break
            pre=i
        if stemAugmented(line[pre:ini-1].lower(),degree=1) in frase_no_tecnico:
            breaks.append((pre,m.span()[1]))
            entro=True
    breaks.sort()
    new_body=[]
    ini = 0
    for br in breaks:
        new_body.append(line[ini:br[0]])
        ini=br[1]
    new_body.append(line[ini:])
    """
    if entro:
        print line
        print new_body
        pdb.set_trace()
    """

    for new_line in new_body:
        # busqueda de patrones
        for (i,pattern) in enumerate(patrones):
            #if i > TEC_TH: # desambiguar algunos casos de TECNICO como adjetivo modificativo
            for car in careersFromPatterns(new_line,patrones,i):
                carreras.add(car.lower())

    origc = set(carreras)  # copia explicita
    carreras = set([stemAugmented(line,degree=1).strip(' ') for line in carreras])

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


    if len(found_areas)==0:
        # buscar ident de tecnico menos 'tecnico'
        if searchIdentifier(new_body,ident_tecnico):
            found_areas.add('tecnico')
    """
    if index==3:
        print new_body
        print found_areas
        pdb.set_trace()
    """
    return found_areas,origc,carreras



######################################################################################################################################################################################################
######################################################################################################################################################################################################
#                                                            HILO PRINCIPAL
patrones = compilePatterns()

for post in data:
    det_pk = post[0]
    desc_pk = post[1]
    job = Details.objects.filter(pk=det_pk)[0]
    desc = job.description_set.filter(pk=desc_pk)[0]

    # armar texto
    cuerpo = strip_encode([job.title]) + []
    cuerpo_original = strip_encode([job.title]) + []

    description = strip_encode([w for w in desc.description.split('\n')])
    for line in description:
        temp = spec_case_patterns[0].sub(spec_case_subs[0],line)
        cuerpo.extend([ll.strip(' ') for ll in temp.split('.') if len(ll.strip(' '))>1])
        
        cuerpo_original.extend([ll.strip(' ') for ll in line.split('.') if len(ll.strip(' '))>1])


    # Variables para identificacion de carreras
    found_areas = set()
    carreras = set()
    origc = set()

    # filtro aqui
    parallelize = False
    
    if parallelize:
        pool = Pool(processes=10)
        majors = pool.map(parallel_filter,cuerpo)
        pool.close()
        pool.join()
        for maj,oc,car in majors:
            found_areas |= maj
            origc |= oc
            carreras |= car
        if len(found_areas)==0:
            found_areas.add('otros')
    else:
        for line in cuerpo:
            try:
                f,o,c = parallel_filter(line,contador_jobs)
                found_areas |= f
                origc |= o
                carreras |= c
            except:
                pdb.set_trace()
    
    
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
::: Carreras sin stemming
%s
::: Areas
%s
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
''' % (contador_jobs,'\n'.join(cuerpo),origc,found_areas)
        contador_jobs += 1
    ##################################################################

    if total % 1000==0:
            print 'Total --> ',total
    total += 1
# ENDFOR
####################################################################################################################################

if UPDATE_VH or UPDATE_ALL:
#    writeHashVector(new_vh,"vh_" + carrera_muestreo)
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
