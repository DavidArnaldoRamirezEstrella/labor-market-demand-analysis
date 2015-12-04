import os,sys
import unicodedata
import pdb,ipdb

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_DIR = os.path.dirname(path_utils)
CRAWLER_DIR = os.path.join(PROJECT_DIR,'crawler')
IDENTIFIER_DIR = os.path.join(CRAWLER_DIR, 'Identifiers')
IDENTIFIER_STEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'identifiers')

sys.path.append(path_utils)

from utils_new import stemAugmented

#pdb.set_trace()

for root, dirs, filenames in os.walk(IDENTIFIER_DIR):
  for f in filenames:
    if f[-1]!='~':
      dest = open(os.path.join(IDENTIFIER_STEM_DIR, f), 'w')
      for line in open(os.path.join(root, f), 'r'):
        line = line.lower().strip('\n').strip(' ').replace('.','')
        if line!='':
          text = unicodedata.normalize('NFKD', line).encode('ascii','ignore').decode('utf-8')
          ident = ' '.join([stemAugmented(word) for word in text.split(' ')])
          dest.write(ident+'\n')
