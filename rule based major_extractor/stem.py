import os,sys
import unicodedata
import pdb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAWLER_DIR = os.path.dirname(os.path.abspath(__file__))
IDENTIFIER_DIR = os.path.join(CRAWLER_DIR, 'Identifiers')
IDENTIFIER_STEM_DIR = os.path.join(CRAWLER_DIR, 'Identifiers_stem')
UTIL_DIR = os.path.join(CRAWLER_DIR, 'nlp scripts')

AREAS_DIR = os.path.join(CRAWLER_DIR, 'areas')
AREAS_STEM_DIR = os.path.join(CRAWLER_DIR, 'areas_stem')

sys.path.append(UTIL_DIR)

from utilities import stemAugmented

#pdb.set_trace()

def leer(data): 
    temp = unicode(data.read().decode('utf-8'))
    temp = unicodedata.normalize('NFKD', temp).encode('ascii','ignore')

    #ans = [line.lower().strip() for line in temp.split('\n') if len(line)>0]
    #ans = [line.replace('.','') for line in ans if len(line)>0]
    ans = [line.lower() for line in temp.split('\n') if len(line)>0]
    return ans

for root, dirs, filenames in os.walk(IDENTIFIER_DIR):
  for f in filenames:
    if f[-1]!='~':
      log = open(os.path.join(root, f), 'r')
      dest = open(os.path.join(IDENTIFIER_STEM_DIR, f), 'w')
      doc = '\n'.join(list(set([stemAugmented(line,degree=1) for line in leer(log)])))
      dest.write(doc)

for root, dirs, filenames in os.walk(AREAS_DIR):
  for f in filenames:
    if f[-1]!='~':
      log = open(os.path.join(root, f), 'r')
      dest = open(os.path.join(AREAS_STEM_DIR, f), 'w')
      doc = '\n'.join(list(set([stemAugmented(line,degree=1) for line in leer(log)])))
      dest.write(doc)
