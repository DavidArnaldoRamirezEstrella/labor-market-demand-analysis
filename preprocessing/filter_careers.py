"""
FILTER ENG-MAJORS ADS AND GENERATE COUNT FILES
GENERATE MAJOR-TOPIC MATRIX
"""
import os, sys
import pymongo
from pymongo import MongoClient
import json
import ipdb

from utils_pipe import *

utils_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(utils_path)
from utils_new import uploadObject

#########################################################################
BASE_DIR = os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
clustering_dir = os.path.join(os.path.dirname(BASE_DIR),'clustering')

clustering_project_dir = os.path.join(BASE_DIR,'dataR')

#data_dir = os.path.join(clustering_project_dir,'enero_ing')
data_dir = os.path.join(clustering_project_dir,'enero_ing_stem')

docs_dir = os.path.join(clustering_dir,'jobs_enero/docs')
ing_dir = os.path.join(clustering_project_dir,'jobs_enero/complete_docs')

if not os.path.exists(data_dir):
	os.makedirs(data_dir)


#########################################################################
client = MongoClient()
db = client.JobDB
db_prueba = db.prueba

#########################################################################
word_dict_filtered = uploadObject(os.path.join(utils_path,'word_dict_filtered'))

temp = json.loads(open('hierarchy/ident_names.json','r').read())
hierarchy  = json.loads(open('hierarchy/carreras.json','r').read())

fileByName = {}
for k,v in temp.items():
	fileByName[v]=k
#########################################################################



if __name__ == '__main__':
	
	files_ingenieria = []
	for level1 in hierarchy["children"]:
		if level1["name"]=='Ingeniería':
			for career in level1["children"]:
				name = career["name"]
				files_ingenieria.append(fileByName[name])

	or_query = [{'carreras':file_name} for file_name in files_ingenieria]

	stem=True

	data = db_prueba.find({'$or':or_query})

	data_names = [p["name"] for p in data]
	for name in data_names:
		path = os.path.join(docs_dir,name)
		text = mallet_text(readDoc(path))
		target = os.path.join(ing_dir,name)
		open(target,'w').write(text)

	"""
	freqdist_ing = make_freqdist(data_names, docs_dir,word_dict_filtered,stem=stem)

	#ipdb.set_trace()

	data = db_prueba.find({'$or':or_query})
	title_map_file = open(os.path.join(data_dir,'title_map.dat'),'w')
	doc_title_map = {}

	doc_term_freq_file = open(os.path.join(data_dir,'word_counts.dat'),'w')
	vocab_file = open(os.path.join(data_dir,'vocab.dat'),'w')

	frequency_by_doc = []
	count = 0
	
	for doc in data:
		doc_name = os.path.join(docs_dir,doc["name"])
		tokens = readDoc(doc_name)

		filtered_tokens = filterTokens(tokens,freqdist_ing)
		# Title map and doc_title mapping
		title = ' '.join(tokens[0])[:50]
		doc_title = doc["name"]
		title_map_file.write(doc_title+'\n')
		doc_title_map[doc_title] = title
		## word_counts by doc
		word_frequency = make_term_frequency(filtered_tokens,freqdist_ing,stem=stem)
		frequency_by_doc.append(word_frequency)
		
		if count%100==0:
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
	"""

	"""
	################################################
	# CAREERS VS TOPICS
	print("Careers vs topics")
	n_docs = db_prueba.count()
	topics = 10
	doc_topic = load_doc_topic_matrix(topics = topics,folder='enero_ing_stem',n_docs=n_docs)
	topicsByCareer = {}

	data = db_prueba.find({'$or':or_query})
	for pointer in data:
		name = pointer['name']
		id = int(name[3:])-1 # doc...
		for car in pointer['carreras']:
			if car not in files_ingenieria:
				continue
			if car not in topicsByCareer:
				topicsByCareer[car] = np.zeros(topics)
			topicsByCareer[car] += doc_topic[id]

	output = open(os.path.join(data_dir,'career_topic_%i.csv' % topics),'w')

	for car,topics in topicsByCareer.items():
		line = car + ' ' + ' '.join([str(top) for top in topics]) + '\n'
		output.write(line)
	"""