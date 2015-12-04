# SYSTEM PIPELINE
import os, sys
import time
import pymongo
from datetime import date, datetime
from pymongo import MongoClient
import pickle

utils_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(utils_path)

from utils import tokenizer
from utils_new import saveObject, uploadObject

from nltk.stem import SnowballStemmer
from nltk.stem.snowball import SpanishStemmer


stemmer = SpanishStemmer()
eng_stemmer = SnowballStemmer("english")

import pdb,ipdb

## MongoDB authentication and connection
client = MongoClient()
# conexion a nueva base de datos
db = client.JobDB
# Coleccion de trabajos
job_posts = db.core_col
tokenized = db.core_col_tokenized

def insert(_dict, key, val = 1):
	if key not in _dict:
		_dict[key] = val
	else:
		_dict[key] += val


if __name__ == '__main__':
	word_dict = {}

	data = job_posts.find().batch_size(10000)
	case = 1
	for post in data:
		id = post['_id']
		job = tokenized.find_one({'$and':[{'_id.description':id['description'] },
								          {'_id.details':id['details'] }
										 ]})
		if job:
			print("Ya esta")
			continue
		
		text = '\n'.join([post.get('title',''),post.get('description','')])
		if text=='':
		  continue
		
		tokens = tokenizer(text)
		new_item = {'_id': id, 'tokens':tokens}

		
		"""
		for sent in tokens:
			for word in sent:
				#if 'movilidad' in word.lower() and len(word) > 9 and word.lower() != 'movilidades':
				#	ipdb.set_trace()
				
				#try:
				#	stem = stemmer.stem(word.lower())
				#except:
				#	stem = eng_stemmer.stem(word.lower())
				#insert(stem_dict,stem)
				insert(word_dict,word.lower())
		"""
				
		if case % 1000 == 0:
			print("->",case)
		
		try:
			tokenized.insert({'_id': post['_id'], 'tokens':tokens})
		except:
			print("REPETIDO!")
		
		case += 1
	
	
	"""
	data = tokenized.find().batch_size(10000)
	case = 1
	for post in data:
		tokens = post['tokens']
		id = post['_id']
		
		if case % 1000 == 0:
			print("->",case)

		for sent in tokens:
			for word in sent:
				if "´practicante" in word:
					ipdb.set_trace()
				
				try:
					stem = stemmer.stem(word.lower())
				except:
					stem = eng_stemmer.stem(word.lower())
				insert(stem_dict,stem)
				insert(word_dict,word.lower())		
				
		case += 1
	"""

	#saveObject(word_dict,'word_dict')
	#saveObject(stem_dict,'stem_dict')

