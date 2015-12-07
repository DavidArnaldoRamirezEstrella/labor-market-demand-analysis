from utilities import *
import os,sys
import json
import numpy
import re
    
career_counts = {}

###################################################################################################
def dfs_count(node):
    name = node["name"]
    if "ing." in node["name"]:
        name = ' '.join(["ingenieria",node["name"].split(" ")[1] ])
    
    if "children" not in node:
        career_counts[name] = node["size"]
        return node["size"]
    
    total = 0
    for child in node["children"]:
        total = total + dfs_count(child)
    career_counts[name] = total
    return total


def createDoc(results_dir = '',career_tags=[],num_posts=100):
    """ input:  career_tags : (list) filtro de matching en job post
                results_dir: path donde escribir los archivos
                num_posts : numero de posts escogidos aleatoriamente
        output: None
        descripcion: crea un archivo por job post
    """

    # Get career counts
    career_path = os.path.join(CRAWLER_DIR,'carreras.json')
    career_tree = json.loads(readLatin(career_path))
    career_tree = career_tree["children"]
     
    for children in career_tree:
        dfs_count(children)
    
    # Get num_posts random numbers
    total_data = 0
    for tag in career_tags:
        total_data = total_data + career_counts[tag]
        
    id_posts = numpy.random.random_integers(0,total_data,num_posts)
    id_posts.sort()
    idx = 0
    
    # Get identifiers by group
    Group_Identifiers = []
    cont = 0
    career_tags = [stemAugmented(tag) for tag in career_tags]
    
    # Identificar carreras que contengan esos tags
    for root, dirs, filenames in os.walk(IDENTIFIER_STEM_DIR):
        for f in filenames:
            if f[-1]!='~':
                ident = os.path.join(IDENTIFIER_STEM_DIR, f)
                text_input = strip_encode(leer_tags(open(ident)), False)
                
                if searchIdentifier(text_input,career_tags):
                    Group_Identifiers.append(ident)
    
    # Leer identificadores de cada carrera relacionada
    Identifiers = []
    for ident in Group_Identifiers:
        tags = strip_encode(leer_tags(open(ident)), False)
        Identifiers.append(tags)
    
    # Buscar en DB
    job_details = Details.objects.all()
    Corpus = []
    flag_maximum = False
    for job in job_details:
        flagDesc = False
        tituloInput = punctuation.sub(" ",job.title).split()
        descInput=""
        descriptions = []
        
        # Get Descriptions
        try:
            descriptions = job.description_set.all()
            flagDesc = True
        except:
            pass
        
        # Search for identifiers
        for i in range(len(Identifiers)):
            ident = Identifiers[i]
            if searchIdentifier(tituloInput,ident):
                cont = cont + 1
                if cont == id_posts[idx]:
                    idx = idx + 1
                    for desc in descriptions:
                        requirements = [strip_encode(str(line)) for line in desc.requirements.split('\n') if len(line)>0]
                        requirements = [req for req in requirements if len(req)>0]
                        doc = [' '.join(tituloInput)]
                        doc.extend(requirements)
                        
                        Corpus.append('\n'.join(doc))
                    # maximo numero de posts
                    if idx == len(id_posts):
                        flag_maximum = True
                        break
                if flag_maximum:
                    break
            elif flagDesc:
                for desc in descriptions:
                    descInput = punctuation.sub(" ",desc.requirements).split()
                    if searchIdentifier(descInput,ident):
                        cont = cont + 1
                        if cont == id_posts[idx]:
                            idx = idx + 1
                            requirements = [strip_encode(str(line)) for line in desc.requirements.split('\n') if len(line)>0]
                            requirements = [req for req in requirements if len(req)>0]
                            doc = [ ' '.join(tituloInput)]
                            doc.extend(requirements)
                            
                            Corpus.append('\n'.join(doc))
                            # maximo numero de posts
                            if idx == len(id_posts):
                                flag_maximum = True
                                break
            if flag_maximum:
                break
        if flag_maximum:
            break
    
    for (i,job) in enumerate(Corpus):
        open(os.path.join(results_dir,'doc%d' % i),'w').write(job)

if __name__=="__main__":
    posts = 50
    career_tags = ['ingenieria']
    results_dir = os.path.join(UTIL_DIR,'data')
    
    createDoc(results_dir=results_dir,career_tags=career_tags,num_posts = posts)
    
    