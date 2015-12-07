from counter import *
import numpy

######################################################################################################################
VECTOR_HASH_DIR = os.path.join(CRAWLER_DIR,'vector_hash')
PSEUDO_PROF = 'vh_pseudo_prof'
PROF = 'vh_prof'
NO_PROF = 'vh_no_prof'
TECNICO = 'vh_tecnico'
NO_TECNICO = 'vh_no_tec'

docnameByCareer = {}

for root, dirs, filenames in os.walk(IDENTIFIER_STEM_DIR):
    for f in filenames:
        if f[-1]!='~' and f != "prof_todo.txt":
            ident = os.path.join(IDENTIFIER_STEM_DIR, f)
            text_input = strip_encode(leer_tags(open(ident)), False)

            for line in text_input:
                docnameByCareer[line] = f

######################################################################################################################



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


def writeDebug(jobs, path,name,ids=[]):
    """
    :param jobs: lista de trabajos, formato (details_hash,description_hash)
    :param path: path de carpeta donde guardar archivo
    :param name: nombre del archivo
    :param ids: lista de indices a escribir
    :return: None
    """
    if ids==[] or len(ids)> len(jobs):
        ids = [i for i in range(len(jobs))]

    ff = open(os.path.join(path,name),'w')
    c = 0
    for (i,post) in enumerate(jobs):
        if i in ids:
            det_pk = post[0]
            desc_pk = post[1]
            job = Details.objects.filter(pk=det_pk)[0]
            desc = job.description_set.filter(pk=desc_pk)[0]
            ff.write('''************************************************     [%d]      ******************************************************************************
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

    for root, dirs, filenames in os.walk(VECTOR_HASH_DIR):
        for f in filenames:
            if f[-1]!='~' and f==full_name:
                vh_file = f
                break
    for line in open(full_name).read().split('\n'):
        if len(line) > 0:
            res.append(line.split(','))
    return res


def existVectorHash(name):
    """
    :param name: nombre de vector hash de carrera a buscar
    :return: Bool
    """
    vh_path = os.path.join(VECTOR_HASH_DIR,name)
    return os.path.exists(vh_path)


def getOnlyCareer(career='no tecnico',debug=True):
    """ input: career: (str) nombre de carrera a filtrar
                 'profesional' : todas las carreras
                 'tecnico' : carreras tecnicas
                 'otros' : demas
        output: lista de tuples (Det_hash,Desc_hash)
        description: objetos filtrados de la forma A* = (A U B) \ B (solo A)
    """
    # Utilizar data ya existente si existe
    filename_career = 'vh_' + docnameByCareer[career]
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
                temp.append([job.hash,desc.hash])
        data = temp

        (primer_filtro,no_primer_filtro) = filterData(data=data,identifiers=ident_prof,ignore=False,complement=True)
        writeHashVector(primer_filtro,PSEUDO_PROF)
        writeHashVector(no_primer_filtro,NO_PROF)

    if debug:
        print "---> 1er filtro: Pseudo profesional | No profesional"
        print "--->    size   :  %d                |     %d" % (len(primer_filtro), len(no_primer_filtro))

    if career == 'no tecnico':
        # 2do filtro : No tecnico sobre No profesional
        res = filterData(data=no_primer_filtro,identifiers=ident_tecnico,ignore=True)
        if debug:
            print "Target: No tecnico | Num posts: %d" % len(res)
        writeHashVector(res,NO_TECNICO)
    elif career == 'tecnico':
        # 2do filtro : Solo_tecnico sobre No profesional
        res = filterData(data=no_primer_filtro,identifiers=ident_tecnico,ignore=False)

        if debug:
            print "Target: Tecnico | Num posts: %d" % len(res)
        writeHashVector(res,TECNICO)
    else:
        # Carreras profesionales
        if debug:
            print "---> Target: Carrera profesional de %s" % career

        career_stem = stemAugmented(career,degree=1)

        #### 2do filtro : no tecnico sobre Pseudo prof
        segundo_filtro = []
        if existVectorHash(PROF):
            segundo_filtro = readHashVector(PROF)
        else:
            segundo_filtro = filterData(data=primer_filtro,identifiers=ident_tecnico,ignore=True)
            writeHashVector(segundo_filtro,PROF)

        if debug:
            print "--->    2do filtro: Profesionales | num posts: %d" % len(segundo_filtro)

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
            print "--->    3er filtro: Complemento de <no %s> | num posts: %d" % (career,len(tercer_filtro))

        # 4to filtro: only <career>
        only_career = "vh_" + docnameByCareer[career_stem]
        if existVectorHash(only_career):
            cuarto_filtro = readHashVector(only_career)
        else:
            cuarto_filtro = filterData(data=tercer_filtro,identifiers=career_tags,ignore=False)
            writeHashVector(cuarto_filtro,only_career)

        if debug:
            print "--->    4to filtro: Only <%s> | num posts: %d'" % (career, len(cuarto_filtro))

        res = cuarto_filtro
    return res


def sampleVectorHash(name,results_dir=UTIL_DIR,samples=50):
    """
    :param name: nombre de la carrera cuyo vector hash se va a muestrear
    :param samples: numero de muestras aleatoreas a escribir
    :param results_dir: directorio donde escribir los resultados | default: mismo directorio q script
    :return: None
    :descrip: Escribe en un archivo con el titulo, req y funciones de cada trabajo, tomando <samples> muestras aleatoreamente
    """
    id_posts = numpy.random.random_integers(0,len(jobs_filtered),samples)
    name = "sample_vh_" + docnameByCareer[name]
    writeDebug(jobs_filtered,results_dir,name,id_posts)

    print "N=%d muestras aleatoreas escritas" % samples
    print "Archivo : %s" % os.path.join(results_dir,name)




######################################################################################################################

if __name__ == "__main__":
    debug = True
    filter_label = 'administracion'
    jobs_filtered = getOnlyCareer(career=filter_label,debug=debug)

    sampleVectorHash(filter_label)