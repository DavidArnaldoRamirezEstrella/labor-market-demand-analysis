import os, sys
import random
from nltk import FreqDist
from datetime import datetime, date
import ipdb

utils_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WWW_DIR = os.path.join(os.path.dirname(utils_path), 'www')
SETTINGS_DIR = os.path.join(WWW_DIR, 'project')

sys.path.append(utils_path)
sys.path.append(WWW_DIR)
sys.path.append(SETTINGS_DIR)

"""
os.environ['DJANGO_SETTINGS_MODULE'] = 'project.dev'

from core.models import Details, Description
"""

from utils import tokenizer
from utils_new import *
BASE_DIR = os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )

from pymongo import MongoClient
client = MongoClient()
db = client.JobDB
# Coleccion de trabajos
job_posts = db.core
tokenized = db.core_tokenized

#########################################################################
random.seed(42)		# siempre, justo antes de usar random

#random_idx = random.sample(range(210000),1000)
#random_idx.sort()

random_idx = range(100000)

#########################################################################
word_dict_filtered = uploadObject(os.path.join(utils_path,'word_dict_filtered'))

#base_prueba_dir = os.path.join(os.path.dirname(BASE_DIR),'clustering/jobs_1000')
base_prueba_dir = os.path.join(os.path.dirname(BASE_DIR),'clustering/jobs_enero')
docs_dir = os.path.join(base_prueba_dir,'docs')

title_map_file = open(os.path.join(base_prueba_dir,'title_map.dat'),'w')
doc_title_map = {}

doc_term_freq_file = open(os.path.join(base_prueba_dir,'word_counts.dat'),'w')
vocab_file = open(os.path.join(base_prueba_dir,'vocab.dat'),'w')

#########################################################################


if __name__ == '__main__':
	# get docs
	print("Getting docs...")
	#data = Details.objects.all()
	#data = job_posts.find().batch_size(10000)

	# PRUEBA DE 1000
	#data = tokenized.find().batch_size(10000)

	
	# TODO ENERO
	ene = date(2015,1,1)
	feb = date(2015,2,1)
	data = job_posts.find({'$and':[
		{'date':{'$gte': datetime.combine(ene,datetime.min.time())}},
		{'date':{'$lt' : datetime.combine(feb,datetime.min.time())}}
		]}).batch_size(10000)
	

	count = 0
	count_iter = 0

	print("Iterando...")
	frequency_by_doc = []

	for post in data:
		# get title
		if count_iter==len(random_idx):
			break

		if count == random_idx[count_iter]:
			count_iter+=1
			
			# procesado para core | PRUEBA ENERO
			text = '\n'.join([post.get('title',''),post.get('description','')])
			if text=='':
				print("Wot, texto vacio!!")
			# Tokenize and assign filter tags
			tokens = tokenizer(text)
			
			# procesado para core_tokenized | PRUEBA DE 1000
			#tokens = post['tokens']

			filtered_tokens = filterTokens(tokens,word_dict_filtered)
			# Title map and doc_title mapping
			title = ' '.join(tokens[0])[:50]
			doc_title = 'doc%i' % count_iter
			title_map_file.write(doc_title+'\n')
			doc_title_map[doc_title] = title
			
			## content to display
			content = [' '.join(sent) for sent in filtered_tokens]
			content = '<br>\n'.join(content)
			open(os.path.join(docs_dir,doc_title),'w').write(content)

			## word_counts by doc
			word_frequency = make_term_frequency(filtered_tokens)
			frequency_by_doc.append(word_frequency)

			if count%1000 == 0:
				print("->",count)
		count += 1

	print("Word freqs by doc...")
	# Write doc : word freqs
	word_id = set()
	for doc in frequency_by_doc:
		words = set(doc.keys())
		word_id |= words
	word_id = list(word_id)

	for doc in frequency_by_doc:
		doc_term_freq_file.write(str(len(doc)))
		for word, freq in doc.items():
			doc_term_freq_file.write(" %i:%i" % (word_id.index(word),freq) )
		doc_term_freq_file.write('\n')


	print("Writing vocab...")
	# Write vocabulary
	vocab = '\n'.join(word_id)
	vocab_file.write(vocab)