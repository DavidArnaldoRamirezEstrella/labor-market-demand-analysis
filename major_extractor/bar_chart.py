import os, sys
import inflection
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.join(BASE_DIR, 'www')
CRAWLER_DIR = os.path.dirname(os.path.abspath(__file__))
IDENTIFIER_DIR = os.path.join(CRAWLER_DIR, 'Identifiers')
IDENTIFIER_STEM_DIR = os.path.join(CRAWLER_DIR, 'Identifiers_stem')

ESPECIALES = ['\n','\r','\t','\a','\b',' ','-','>','*']

#print IDENTIFIER_DIR

sys.path.append(PROJECT_DIR)
os.environ['DJANGO_SETTINGS_MODULE'] = 'project.dev'
from core.models import Details, Description

Group_Identifiers = []
JobAreas = []
for root, dirs, filenames in os.walk(IDENTIFIER_STEM_DIR):
    for f in filenames:
    	if f[-1]!='~':
    		JobAreas.append(f)
    		Group_Identifiers.append(os.path.join(IDENTIFIER_STEM_DIR, f))
JobAreas.append('Otros')

JobCount = []
for i in range(len(Group_Identifiers)):
	JobCount.append([0,i])
JobCount.append([0,len(Group_Identifiers)])

def especial(car): # retorna verdadero o falso si el caracter pertenece al conjunto
    return car in ESPECIALES

def leer_tags(data): # lee los delimitadores de los archivos tags
    ans = []
    line = data.readline()
    while line:
        ans.append( line )
        line = data.readline()
    return ans

def strip_encode(text,flag_code = True): # devuelve el ascii y ademas elimina los caracteres especiales
    ans = []
    if type(text)==str:
    	texto = text
    	texto = texto.lower()
        while(len(texto)>0 and (especial(texto[-1]) or especial(texto[0]) )):
            for special in ESPECIALES:
            	texto = texto.strip(special)

        return texto
    if flag_code==True:
    	text = [unicodedata.normalize('NFKD', line).encode('ascii','ignore') for line in text]
    for texto in text:
        texto = texto.lower()
        while(len(texto)>0 and (especial(texto[-1]) or especial(texto[0]) )):
            for special in ESPECIALES:
            	texto = texto.strip(special)

        if len(texto)>1:
            ans.append(texto)

    return ans

punctuation = re.compile(r'[-.?!,":;()|0-9]')

def searchIdentifier(text,ident):
	for tag in ident:
		tag = tag.split()
		if len(tag)>1:
			for i in range(0,len(text)-len(tag)+1):
				cont_ind = 0
				for j in range(0,len(tag)):
					ind = text[i+j].find(tag[j])
					if ind!=-1:
						cont_ind+=1
				if cont_ind == len(tag):
					return True
		else:
			for t in text:
				ind = t.find(tag[0])
				if( ind!=-1 ):
					return True
	return False

Identifiers = []
for ident in Group_Identifiers:
	tags = strip_encode(leer_tags(open(ident)), False)
	Identifiers.append(tags)



###################################################################################################
###################################################################################################

job_details = Details.objects.all()
contador_jobs = 0


for job in job_details:
    Ifound = False
    flagDesc = False
    textoInput = punctuation.sub(" ",job.title).split()
    descInput=""
    descriptions = []
    try:
        descriptions = job.description_set.all()
        flagDesc = True
    except:
        pass

    for i in range(len(Identifiers)):
        ident = Identifiers[i]
        if searchIdentifier(textoInput,ident):
            #print "job: " + str(job) + " - " + searchIdentifier(job.title,ident)
            JobCount[i][0] += 1
            Ifound = True
        elif flagDesc:
            for desc in descriptions:
                descInput = punctuation.sub(" ",desc.requirements).split()
                if searchIdentifier(descInput,ident):
                    JobCount[i][0] += 1
                    Ifound = True

    if Ifound==False:
        JobCount[len(Identifiers)][0] += 1
    contador_jobs+=1
    if contador_jobs % 500==0:
        print '-->',contador_jobs

JobCount.sort()
print 'Campo de trabajo, Numero de trabajos'

total = 0.0

for area in JobCount:
    total += area[0]
    print '\'{0}\':\'{1}\''.format(JobAreas[area[1]],str(area[0]))

