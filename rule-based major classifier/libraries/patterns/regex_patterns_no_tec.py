import re
import os,sys
import pdb
import numpy as np

regex_dir = os.path.dirname(os.path.abspath(__file__))
util = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A_path = os.path.join(regex_dir,'patternA')
Atec_path = os.path.join(regex_dir,'patternAtec')
Apost_path = os.path.join(regex_dir,'pattern_postA')
Ainner_path = os.path.join(regex_dir,'patternAinner')
Amultiword_path = os.path.join(regex_dir,'Amultiword')
C_path = os.path.join(regex_dir,'patternC')
lug_path = os.path.join(regex_dir,'lugares')

sys.path.append(util)

from utilities import *

####################################################################################
sf = ['\\s+y\\s+',
      '\\s+o\\s+',
      '[-|,;/.]',
      '\\s+y/o\\s+',
      '\\s+o/y\\s+',
      '\\s+y/u\\s+',
      '\\s+c/s\\s+',
      '\\s+e\\s+',
      '\\s+u\\s+',
      '\\s+con\\s+',
      ]
temp = []
for w in sf:
    if '[' not in w:
        temp.append('('+w+')')
    else:
        temp.append(w)
sf = list(temp)

separadores = ['y', 'o', 'y/o', 'y/u', '[-|,;/]', 'e', 'u','con', 'a']

sep_list = ['y', 'o', 'y/o', 'y/u', '-','|',',',';','/', 'e', 'u','con', 'a']

bet = ['de','en','con', 'como', '-','del','los']

########################################################################################################################
stopwords = getStopWords(stemming = True,ignore_careers=True)

ident_tecnico = strip_encode(leer_tags(open(os.path.join(IDENTIFIER_STEM_DIR,'tecnico'))), False)

###########################################################################################################
# Tokenizador de carreras
regex_separadores = '(%s)' % '|'.join(sf)
pattern_sep = re.compile(regex_separadores)

# Depurador de separadores al inicio y final de linea
ext_sep = list(separadores)
ext_sep.extend(bet)
ext_sep = list(set(ext_sep))
ini = '^(%s)\\s+' % '|'.join(ext_sep)
fin = '\\s+(%s)$' % '|'.join(ext_sep)
ini = re.compile(ini)
fin = re.compile(fin)

#Depurar palabras en parentesis
clean_parentesis = '(\\(.*\\))|(\\(\w*\\b)'
clean_parentesis = re.compile(clean_parentesis)

# Limpiar caracteres no alfabeticos del inicio y final de lina
clean_ini = r'^[^a-z(]*(?P<open>[(]?)'
clean_fin = r'(?P<close>[)]?)[^a-z)]*$'
clean_ini = re.compile(clean_ini)
clean_fin = re.compile(clean_fin)

############################################################################################################
# Patrones internos
Ainner = [stemAugmented(line,degree=1) for line in readLatin(Ainner_path).split('\n') if len(line)>0]
Ainner = addSuffixRegex(Ainner)
Ainner = '|'.join(Ainner)
bett = '|'.join(bet)
#            Ainner \s+ ,?(\s+<bet>\s*)?   ,?        carrera
patternInner = '(%s)\\s*,?(\\s+(%s)\\s*)?[,:;]?\\b(?P<carreras>.*)' % (Ainner,bett)
patternInner =  re.compile(patternInner)

############################################################################################################
## CASO ESPECIAL : MEDICINA | TECNOLOGIA MEDICA
# Se parte la oracion en dos cuando encuentra un identificador de medico (de una palabra)
ident_med = strip_encode(leer_tags(open(os.path.join(IDENTIFIER_STEM_DIR,'medicina'))), False)
ident_med = [id for id in ident_med if len(id.split(' '))==1]
# incluir ident de tecnologia medica
ident_med.append('medizintechnik')
Med = addSuffixRegex(ident_med)
Med = '|'.join(Med)
# Patron medico de separado
med_pattern = r'(?P<prev>.*(%s))(?P<post>.*)' % Med
med_pattern = re.compile(med_pattern)

############################################################################################################
RAW_PATTERNS = []
############################################################################################################


def compilePatterns():
    """
    Lee patrones de los archivos patternA, patternC, patternAtec
    Compila regex de patrones.
    :return:Lista de pattern objects
    """
    # stemmed lists of pattern A
    A    = [stemAugmented(line,degree=1) for line in readLatin(A_path).split('\n')    if len(line)>0]
    C    = [stemAugmented(line,degree=1) for line in readLatin(C_path).split('\n')    if len(line)>0]
    Atec = [stemAugmented(line,degree=1) for line in readLatin(Atec_path).split('\n') if len(line)>0]
    Amultiword = [stemAugmented(line,degree=1) for line in readLatin(Amultiword_path).split('\n') if len(line)>0]
    A_post = [stemAugmented(line,degree=1) for line in readLatin(Apost_path).split('\n') if len(line)>0]
    
    lugares = ['('+line.strip()+')' for line in readLatin(lug_path).split('\n') if len(line)>0]

    A = list(set(A))
    C = list(set(C))
    Atec = list(set(Atec))
    Amultiword = list(set(Amultiword))
    A_post = list(set(A_post))
    lugares = list(set(lugares))

    A = addSuffixRegex(A)
    C = addSuffixRegex(C)
    Atec = addSuffixRegex(Atec)
    A_post = addSuffixRegex(A_post)
    Amultiword = addSuffixRegex(Amultiword)

    A = '|'.join(A)
    C = '|'.join(C)
    Atec = '|'.join(Atec)
    A_post = '|'.join(A_post)
    Amultiword = '|'.join(Amultiword)
    lugares = '|'.join(lugares)


    bett = '|'.join(bet)
    sep = '|'.join(separadores)

    # Primero : ya no se usa!!
    #** <A> <de,en> <B>,<B>
    #             A  \s  (bet \s)?  carrera
    #             A  \s  bet \s  (carrera.*) CORRECION   11/05/2015
    pattern1 = '(%s)\\s+(%s)\\s+(?P<carreras>.*)\.?' % (A,bett)
    pattern1 = re.compile(pattern1)

    # Segundo           0
    #** <A> <C>? (o C)? <de,en,->? <B> si <B> sf <B>
    #             A  \s  C   \s*  (sep \s+  C)* ,* \s* (bet \s+)? (carreras.*)   .?

    #** <A> <C>? (o C)? <de,en,-> (carrera.*) CORRECION   11/05/2015
    #             A  \s+(bett \s+)* (C\s+)*   (sep \s+  C\s+)*   (bett \s+)+ (carreras.*)   .?
    pattern2 = '(%s)\\s+((%s)\\s+)*((%s)\\s+)*((%s)\\s+(%s)\\s+)*((%s)\\s+)+(?P<carreras>.*)\\.?' % (A,bett,C,sep,C,bett)
    pattern2 = re.compile(pattern2)

    # Segundo multiword           1
    #** <Amw> <C>? (o C)? <de,en,->? <B> si <B> sf <B>
    #             A  \s  C   \s*  (sep \s+  C)* ,* \s* (bet \s+)? (carreras.*)   .?
    pattern25 = '(%s)\\s+(%s)*\\s*((%s)\\s+(%s))*\\s*((%s)\\s+)?(?P<carreras>.*)\\.?' % (Amultiword,C,sep,C,bett)
    pattern25 = re.compile(pattern25)


    ### Tercero yeahhh!         2
    #** <A> <C>? (o C)? <de,en>? (:|;) <B> sf <B>
    #            A  \s+ ( C  \s+ ( o  \s+  C )* )*(\s+ bet \s*)? (:|;) \s*  carreras
    pattern3 = '(%s)\\s*((%s)\\s+((%s)\\s+(%s))*)*(\\s+(%s)\\s*)?(:|;)\\s*(?P<carreras>.*)\\.?' % (A,C,sep,C,bett)
    pattern3 = re.compile(pattern3)

    ### Tercero multiword         3
    #** <Amw> <C>? (o C)? <de,en>? (:|;) <B> sf <B>
    #          Amw \s+ ( C  \s+ ( o  \s+  C )* )*(\s+ bet \s*)? (:|;) \s*  carreras
    pattern35 = '(%s)\\s*((%s)\\s+((%s)\\s+(%s))*)*(\\s+(%s)\\s*)?(:|;)\\s*(?P<carreras>.*)\\.?' % (Amultiword,C,sep,C,bett)
    pattern35 = re.compile(pattern35)

    pattern_dpoint_wide = r'(%s).*:\s*((%s)\s+)*(?P<carreras>.*)' % (A,bett)
    pattern_dpoint_wide = re.compile(pattern_dpoint_wide)

    # TITULO + EN <LUGAR>
    #                A  \s* ( C \s+( o  \s+  C )*)*(\s+ bet \s*)?   carreras              en                            lima
    pattern_lug = r'(%s)\s*((%s)\s+((%s)\s+(%s))*)*(\s+(%s)\s+)?(?P<carreras>[a-z -]*)\s+((en)|(de)|(residente en))\s+(%s)' % (A,C,sep,C,bett,lugares)
    pattern_lug = re.compile(pattern_lug)

    # SANDWICH  <A> <carrera> <A_post>
    #                A  \s* ( C \s+( o  \s+  C )*)*(\s+ bet \s*)?   carreras              en                            lima
    pattern_sand = r'(%s)\s*((%s)\s+((%s)\s+(%s))*)*(\s+(%s)\s+)?(?P<carreras>.*)\s+(%s)' % (A,C,sep,C,bett,A_post)
    pattern_sand = re.compile(pattern_sand)

    # A + (CARRERAS EN PARENTESIS)
    pattern_brak = r'(%s)\s+((%s)\s+((%s)\s+(%s))*)*[(](?P<carreras>.*)[)]?' % (A,C,sep,C)
    pattern_brak = re.compile(pattern_brak)

    ### Cuarto yeahhh! | por siaca      4
    #** <A> <C>? (o C)?: .* <de,en,-> <B> sf <B>
    #            A  \s ( C   \s ( o   \s  C )* )* \s*(:|;).* \b bet \b \s* carreas
    #pattern4 ='(%s)\\s*((%s)\\s+((%s)\\s+(%s))*)*\\s*(:|;).*\\b(%s)\\b\\s*(?P<carreras>.*)\.?' % (A,C,sep,C,bett)

    #** <A> <C>? (o C)?: <B> sf <B>     CORREGIDO / REDUNDANTE CON 3
    #            A  \s+(o\s+)?(bett\s+)* C  \s+ (o      C)*      bett
    pattern4 =r'(%s)\s+(o\s+)?((%s)\s+)*((%s)\s+)((%s)\s+(%s)\s+)*(%s)\s+(?P<carreras>.*)\.?' % (A,bett,C,sep,C,bett)
    pattern4 = re.compile(pattern4)

    ### Cuarto mutliword                5
    #** <Amw> <C>? (o C)?: .* <de,en,-> <B> sf <B>
    #          Amw  \s ( C   \s ( o   \s  C )* )* \s*(:|;).* \b bet \b \s* carreas
    pattern45 ='(%s|(personal))\\s*((%s)\\s+((%s)\\s+(%s))*)*\\s*(:|;).*\\b(%s)\\b\\s*(?P<carreras>.*)\.?' % (Amultiword,C,sep,C,bett)
    pattern45 = re.compile(pattern45)

    ### Quinto:                  6
    #** <B> sf <B>? <A_post> <C>? (o C)? <de,en,->
    #           <frontera>                               carreras<=30char \s+ A  (\s+ C )? bet?
    pattern5 =r'(:|,|;|/|y|o|u|e)\s+(?P<carreras>.{0,30})\s+(%s)(\s+(%s))?(%s)?' % (A_post,C,bett)
    pattern5 = re.compile(pattern5)

    # CARRERA{,20} A_POST
    pattern_ini =r'^((%s)\s+)?(?P<carreras>.{0,20})\s+(%s)' % (bett,A_post)
    pattern_ini = re.compile(pattern_ini)

    ### INGENIEROS
    #  <ing> <de,en> <B>
    pattern_ing = r'^.{,10}\bing[a-z.()]*\s+((%s)\s+)?(?P<carreras>.*)(\s+((en)|(de)|(residente en))\s+(%s))?' % (bett,lugares)
    pattern_ing = re.compile(pattern_ing)

    pattern_ing_lug = r'^.{,5}\bing[a-z.()]*\s+((%s)\s+)?(?P<carreras>.*)\s+((en)|(de)|(residente en))\s+(%s)' % (bett,lugares)
    pattern_ing_lug = re.compile(pattern_ing_lug)

    #  ing <&> tec <B>
    pattern_ing_tec1 = r'^.{,10}\bing[a-z.()]*\s+(y|o|(y/o))\s+tecnic[a-z.()]*\s+((%s)\s+)?(?P<carreras>.*)\.?' % bett
    pattern_ing_tec1 = re.compile(pattern_ing_tec1)
    # tec <&> ing <B>
    pattern_ing_tec2 = r'^.{,10}\btecnic[a-z.()]*\s+(y|o|(y/o))\s+ing[a-z.()]*\s+((%s)\s+)?(?P<carreras>.*)\.?' % bett
    pattern_ing_tec2 = re.compile(pattern_ing_tec2)

    # REQUISITO: -? ing ...
    pattern_req = r'\brequ[a-z]+\s*:?\s*-?\s*ing[a-z().]*\s+(%s)?\s*(?P<carreras>.*)\.?' % bett
    pattern_req = re.compile(pattern_req)

    #PREFERECIA
    pattern_pref = r'\bpreferencia\s*:?\s*ing[a-z().]*\s+(%s)?\s*(?P<carreras>.*)\.?' % bett
    pattern_pref = re.compile(pattern_pref)

    pattern_pref2 = r'^(%s\s+)?(?P<carreras>.*)\s+((de)|(en))\s+preferencia' % bett
    pattern_pref2 = re.compile(pattern_pref2)

    ### Comodin <B> <sf> <B>                                         - :( | not used yet
    patternCom = pattern_sep

    ### Sexto <A> <C>?,? sf <Atec> <C>? <de,con,en>? <B> si <B> sf <B>          8
    #           A  \s+  C*  \s* ,? \s* sep \s+ Atec \s*  C* \s+ bet \s+  carreras
    pattern6 ='(%s)\\s+((%s)\\s*)*,?\\s*(%s)\\s+(%s)\\s+((%s)\\s+)*(%s)\\s+(?P<carreras>.*)' % (A, C,sep,Atec,C,bett)
    pattern6 = re.compile(pattern6)

    ### Sexto multiword
    # <Amw> <C>?,? sf <Atec> <C>? <de,con,en>? <B> si <B> sf <B>          9
    #          Amw  \s+  C*  \s* ,? \s* sep \s+ Atec \s*  C* \s+ bet \s+  carreras
    pattern65 ='(%s)\\s+(%s)*\\s*,?\\s*(%s)\\s+(%s)\\s*(%s)*\\s+(%s)\\s+(?P<carreras>.*)\\.?' % (Amultiword, C,sep,Atec,C,bett)
    pattern65 = re.compile(pattern65)

    ### Setimo <Atec> <C>? ,? sf <A> <C>? <de,con,en>? <B> si <B> sf <B>        10
    #          Atec \s* C*  \s* ,? \s  sep \s*  A  \s*  C*  \s* bet \s* carreras
    pattern7 ='(%s)\\s+((%s)\\s*)*,?\\s*(%s)\\s+(%s)\\s+((%s)\\s+)*(%s)\\s+(?P<carreras>.*)' % (Atec, C,sep,A,C,bett)
    pattern7 = re.compile(pattern7)

    ### Septimo multiword
    # <Atec> <C>? ,? sf <Amw> <C>? <de,con,en>? <B> si <B> sf <B>        11
    #          Atec \s* C*  \s* ,? \s  sep \s* Amw  \s*  C*  \s* bet \s* carreras
    pattern75 ='(%s)\\s*(%s)*\\s*,?\\s*(%s)\\s*(%s)\\s*(%s)*\\s*(%s)\\s*(?P<carreras>.*)\\.?' % (Atec, C,sep,Amultiword,C,bett)
    pattern75 = re.compile(pattern75)


    ### Octavo <Atec> <C>? sf <C>? <de,en> <B> sf <B>                12
    #           Atec \s+( C  \s+  (sep \s+  C)* )*\s* (bet \s+)? (carreras.*)
    pattern8 = '(%s)\\s+((%s)\\s+((%s)\\s+(%s))*)*\\s*((%s)\\s+)?(?P<carreras>.*)\\.?' % (Atec,C,sep,C,bett)
    pattern8 = re.compile(pattern8)

    ### Noveno                                                  13
    #** <Atex> <C>? (o C)?: .* <de,en,-> <B> sf <B>
    #          Atec  \s ( C   \s ( o   \s  C )* )* \s*(:|;).* \b bet \b \s* carreas
    pattern9 ='(%s)\\s*((%s)\\s+((%s)\\s+(%s))*)*\\s*(:|;).*\\b(%s)\\b\\s*(?P<carreras>.*)\\.?' % (Atec,C,sep,C,bett)
    pattern9 = re.compile(pattern9)

    ### Decimo                                                  14s
    #** <Atec> <C>? (o C)? <de,en>? (:|;) <B> sf <B>
    #          Atec  \s+ ( C  \s+ ( o  \s+  C )* )*(\s+ bet \s*)? (:|;) \s*  carreras
    pattern10 ='(%s)\\s*((%s)\\s+((%s)\\s+(%s))*)*(\\s+(%s)\\s*)?(:|;)\\s*(?P<carreras>.*)\\.?' % (Atec,C,sep,C,bett)
    pattern10 = re.compile(pattern10)

    global RAW_PATTERNS
    RAW_PATTERNS = [
#            pattern1,
            pattern2,   #0
            pattern25,  #1
            pattern3,   #2
            pattern35,  #3
            pattern4,   #4
            pattern45,  #5
            pattern5,   #6
            pattern_ing,    #7
            pattern_ing_lug,    #8
            pattern_ing_tec1,   #9
            pattern_ing_tec2,   #10
            pattern_req,        #11
            pattern_pref,        #12
            pattern_pref2,        #13
            pattern_lug,        #14
            pattern_sand,       #15
            pattern_ini,        #16
            pattern_dpoint_wide, #17
            pattern_brak, #18
#            patternCom,
            pattern6,
            #pattern65,
            pattern7,
            #pattern75,
            #pattern8,
            #pattern9,
            #pattern10,
            ]
    return RAW_PATTERNS 

TEC_TH = 18  # umbral para patrones de tecnico : last normal index

comp_par = re.compile(r'[(].*[)]')

def extractParenthesis(text):
    text = text.replace(',,',',').replace(', ,',',').replace(' , ',', ')
    _map = np.zeros(len(text))
    if '(' not in text and ')' not in text:
        return [text]
    pairs = []
    _open = []

    for i in xrange(len(text)):
        if text[i]=='(':
            _open.append(i)
        if text[i]==')':
            if len(_open)>0:
                pairs.append( (_open.pop(),i) )
            else:
                pairs.append((-1,i))

    while(len(_open)):
        pairs=[(_open.pop(),len(text)) ] + pairs

    res = set()
    for i in xrange(len(pairs)):
        u,v = pairs[i]
        chunk = ''.join([c for j,c in enumerate(text[u+1:v]) if _map[u+1+j]==0])
        chunk = ini.sub('',chunk)
        chunk = fin.sub('',chunk)
        _map[max(u,0):min(v+1,len(text))]=1
        res.add(chunk)
    chunk = ''.join([c for i,c in enumerate(text) if _map[i]==0])
    chunk = ini.sub('',chunk)
    chunk = fin.sub('',chunk)
    res = list(res)
    res.append(chunk)
    res = [t for t in res if all([  len(t)>0,
                                    stemAugmented(t) not in stopwords or t=='tecnico'
                                 ] )]

    return res



def careersFromPatterns(text,patterns,index,debug=False):
    """
    :param text: (str) texto de donde extraer carreras
    :param patterns: list of pattern objects, (re.compile)
    :param index: index in patterns, which pattern to use
    :return: Lista de (str), "carreras" extraidas
    """
    _open  = clean_ini.search(text).group('open')
    _close = clean_fin.search(text).group('close')
    text = clean_ini.sub(_open,text)
    text = clean_fin.sub(_close,text)

    pattern = patterns[index]

    careers = []

    match = pattern.search(text)

    if pattern != pattern_sep:
        if not match:
            return []
        careers = match.group('carreras')
        careers = [cc.strip(' ') for cc in pattern_sep.split(careers) if cc and len(cc)>0]
    else:
        careers = [cc.strip(' ') for cc in pattern.split(text) if cc and len(cc)>0]
        if text in careers:
            return careers

    if index > TEC_TH:       # patrones de tecnicos
        careers.append('tecnico')

    careers = [w for w in careers if w not in sep_list]

    ## CASO ESPECIAL : MEDICINA | TECNOLOGIA MEDICA
    temp = []
    for car in careers:
        match = med_pattern.search(car)
        if match:
            temp.append(match.group('prev'))
            temp.append(match.group('post'))
        else:
            temp.append(car)

    careers = list(temp)

    """
    if debug:
        print careers
        pdb.set_trace()
    """

    # Depurar separadores quedados al final / inicio de linea
    temp = []
    idx_agreg = []
    # extraer parentesis
    for i,car in enumerate(careers):
        ext = extractParenthesis(car)
        temp.extend(ext)
        if len(ext)!=1:
            idx_agreg.append(len(temp)-1)
    careers = list(temp)



    # Iterativo, seguir filtrando en cada pseudo_carrera
    res = set()
    for j,maj in enumerate(careers):
        replace = False
        foo = []
        # testear cada extraccion O(pat^2)
        for (i,pat) in enumerate(patterns):
            #probar si aun falta filtrar palabras ajenas a carreras
            temp = careersFromPatterns(maj,patterns,i)
            if len(temp)>0:
                foo.extend(temp)
                replace = True
        if replace:
            res |=set(foo)
        else:
            res.add(maj)

        #if 'mineria' in maj and debug:
        #    print maj
        #    pdb.set_trace()

        # Buscar patrones internos
        if j not in idx_agreg:
            inner = careersFromPatterns(maj,[patternInner],0)
            res |= set(inner)
    

    res = strip_encode(res)
    res = [line.replace('.','') for line in res]
    return res



if __name__=="__main__":
    patrones = compilePatterns()
    carreras = set()
    """
    cuerpo = ['importante empresa contratista del sector de la mineria y construccion, se encuentra en \
    la busqueda de profesional tecnico mecanico de trackles para que se integren a su equipo de trabajo. se \
    ofrece remuneracion superior al mercado, desarrollo profesional en una empresa de minera del norte de envergadura \
    con grato ambiente de trabajo. . requisitos:  tecnico mecanico *egresados (indispensable). * experiencia no menor a 2 \
    anos en temas de gestion de mina y con conocimiento en: -gestion, evaluacion, planeacion y mantenimiento de equipos dumper, \
    scoops, trackless (equipos subterraneos) - conocimientos en motores electronicos, soldaduras y torno.']
    
    cuerpo = ['egresado de computacion e informatica o ingenieria de sistemas ( tecnico o universitario)']

    cuerpo = [' - 01 (uno) ing. egresado ambiental, minas o ingenierias afines, titulado y colegiado. ']

    cuerpo = ['egresados de las carreras administracion de empresas, ingenieria industrial, negocios internacionales, y/o afines ( instituto o universidad)']

    cuerpo = ['ingeniero de seguridad - titulado en ingenieria de minas o civil en junin']
    cuerpo = ['supervisor de seguridad - bachiller en minas y/o civil en lima']

    cuerpo = ['superior universitaria de preferencia carreras ingenieria civil o electromecanica']
    cuerpo = ['tecnico y/o ingenieria de minas, geomecanica.']
    """
    cuerpo = ['importante empresa contratista requiere personal: ingeniero de minas, mecanico y/o electrico. afines para spcc cuajone.']
    cuerpo = ['de preferencia con practicas pre profesionales como minimo 6 meses en el area de capacitacion rrhh, de preferencia en empresas del sector petrolero o minero']
    #cuerpo = ['empresa de servicios requiere ing minas, metalurgicos (bachilleres) con 2 anos de experiencia como supervisor de seguridad en unidad minera para trabajar a regimen 14x7 en campamento minero']
    #cuerpo = ['nuestro cliente, atlas copco, una empresa responsable de la comercializacion de maquinaria, accesorios, herramientas, repuestos y servicio tecnico de post venta para la industria de construccion y mineria, requiere incorporar a su staff de profesionales: supervisor mantenimiento - trackless (dd-sup) perfil: profesional titulado y colegiado en ing mecanica']
    cuerpo = ['responsable del area de operaciones en mina lagunas norte ( mina tajo a cielo abierto), de preferencia ing de minas titulado, colegiado y ']
    #cuerpo = ['ingeniero industrial alimentarias en la libertad']
    #cuerpo = ['se necesita ing civil titulado y colegiado, sin experiencia para trabajo de campo, tanto en lima como en provincia']
    cuerpo = ['estudios universitarios/postgrado/conocimientos:     ing  mecanica, mecatronica, electronico']
    cuerpo = ['estudiantes a partir del viii ciclo de ing industrial, contabilidad, economia o afines']
    
    
    for line in cuerpo:
        for (i,pattern) in enumerate(patrones):
            for car in careersFromPatterns(line,patrones,i,debug=True):
                carreras.add(car)

    carreras = list(carreras)
    print "#################################################################################################"
    for car in carreras:
        print car


    """
    vh = 'vh_prof'
    jobs = readHashVector(vh)
    jobs = sampleVectorHash(data=jobs,samples=50)

    c = 0

    for post in jobs:
        det_pk = post[0]
        desc_pk = post[1]
        job = Details.objects.filter(pk=det_pk)[0]
        desc = job.description_set.filter(pk=desc_pk)[0]

        title = strip_encode([job.title])
        req = strip_encode([w for w in desc.requirements.split('\n')])
        fun = strip_encode([w for w in desc.functions.split('\n')])

        cuerpo = title
        cuerpo.extend(req)
        cuerpo.extend(fun)
        carreras = []

        for line in cuerpo:
            temp = []
            for (i,pattern) in enumerate(patrones):
                temp.extend(careersFromPatterns(line,patrones,i))
            carreras.extend(list(set(temp)))

        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print '\n'.join(cuerpo)
        print carreras
        print "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

    """
