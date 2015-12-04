# SYSTEM PIPELINE
import os, sys
import pymongo
from pymongo import MongoClient
from datetime import date, datetime

import os,sys
import pdb,ipdb

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(path_utils,'carr_model')

sys.path.append(features_path)
sys.path.append(path_utils)

import ext2 as exfc2
from utils import tokenizer
from utils_new import saveObject, uploadObject
from utils_pipe import *

## MongoDB authentication and connection
client = MongoClient()
db = client.JobDB
core = db.core_col
tokenized = db.core_col_tokenized

def insert(_dict, key, val = 1):
	if key not in _dict:
		_dict[key] = val
	else:
		_dict[key] += val

def readDoc(doc_file):
	res = []
	try:
		for line in open(doc_file):
			line = line.strip('\n').replace('<br>','')
			if line!='':
				sent = line.split(' ')
				res.append(sent)
	except:
		pass
	return res


if __name__ == '__main__':
	model = uploadObject('sp_5_by_sent')

	ini = date(2014,6,1)

	data = core.find(	{'date':{'$gte': datetime.combine(ini,datetime.min.time())} } ,timeout=False).batch_size(1000)

	count = 0
	for dat in data:
		id = dat['_id']
		job = tokenized.find_one({'$and':[{'_id.description':id['description'] },
								          {'_id.details':id['details'] }
										 ]})
		temp = job.get("carreras",[])
		if temp!=[]:
			print("YALA")
			continue
		tokens = job['tokens']
		name_entities = []
		for sent in tokens:
			sequence = makeSequence(sent)
			pred_sequence,_ = model.viterbi_decode_bigram(sequence)
			pred_sequence.sequence_list.seq_list[0].y = pred_sequence.y

			nes = getNameEntities(pred_sequence)
			name_entities.extend(nes)

		carreras = discretizeCareers(name_entities)
		"""
		if carreras!=[]:
			print(name_entities)
			print(carreras)
			ipdb.set_trace()
		else:
		"""
		try:
			tokenized.update({'$and':[{'_id.description':id['description'] },
							          {'_id.details':id['details'] }
							  	     ]},
					  		 {'$set': {'carreras':carreras}} )
		except:
			print("NO INSERTO")
			
		if count%100==0:
			print('-->',count)
		count +=1