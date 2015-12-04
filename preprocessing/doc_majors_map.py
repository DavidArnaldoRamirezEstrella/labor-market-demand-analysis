"""
GENERATE DOC: [MAJORS] csv/tsv for R processing
"""
import os, sys
import pymongo
from pymongo import MongoClient
import json
import ipdb

BASE_DIR = os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
counts_dir = os.path.join(BASE_DIR,'dataR/counts')

ne_counts   = os.path.join(counts_dir,'NE_ene_ing')
full_counts = os.path.join(counts_dir,'full_ene_ing')

#########################################################################
client = MongoClient()
db = client.JobDB
db_prueba = db.prueba

#########################################################################
temp = json.loads(open('hierarchy/ident_names.json','r').read())
hierarchy  = json.loads(open('hierarchy/carreras.json','r').read())

fileByName = {}
for k,v in temp.items():
	fileByName[v]=k

#########################################################################

if __name__ == '__main__':
	major_eng = []
	for level1 in hierarchy["children"]:
		if level1["name"]=='Ingeniería':
			for career in level1["children"]:
				name = career["name"]
				major_eng.append(fileByName[name])

	ipdb.set_trace()
	#curr_count = ne_counts
	curr_count = full_counts

	title_map = [line.strip('\n') for line in open(os.path.join(curr_count,'title_map.dat')) if line.strip('\n')!='']	
	majors_by_doc = open(os.path.join(curr_count,'majors_by_doc.dat'),'w')
	for doc_name in title_map:
		p = db.prueba.find_one({'name':doc_name})
		majors = [a for a in p['carreras'] if a in major_eng]
		majors_by_doc.write(','.join(majors)+'\n')