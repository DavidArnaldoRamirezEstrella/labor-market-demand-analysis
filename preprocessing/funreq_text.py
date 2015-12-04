import os, sys
import pymongo
from pymongo import MongoClient
import json
import ipdb

from utils_pipe import *

utils_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(utils_path)
from utils_new import uploadObject, ChunkSet

#########################################################################
BASE_DIR = os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
clustering_dir = os.path.join(BASE_DIR,'dataR')
models_dir = os.path.join(os.path.dirname(BASE_DIR),'modelos pickle')

source_dir = os.path.join(os.path.dirname(BASE_DIR),'clustering/jobs_enero/docs')
target_dir = os.path.join(clustering_dir,'jobs_enero/funreq_docs')
funmodel_path = os.path.join(models_dir,'fun')
reqmodel_path = os.path.join(models_dir,'req')
carrmodel_path = os.path.join(models_dir,'carr')
jobarea_path = os.path.join(models_dir,'job')

sys.path.append(funmodel_path)
sys.path.append(reqmodel_path)

if not os.path.exists(target_dir):
	os.makedirs(target_dir)


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
#########################################################################
print("Loading models...")
fun_model = uploadObject(os.path.join(funmodel_path,'prod_model'))
req_model = uploadObject(os.path.join(reqmodel_path,'sp_5_by_sent'))
carr_model = uploadObject(os.path.join(carrmodel_path,'prod_model'))
jobarea_model = uploadObject(os.path.join(jobarea_path,'sp_5_by_doc'))

def parallel_funct(docname):
	print(docname)
	doc = readDoc(os.path.join(source_dir,docname))
	sequence = makeSequence_doc(doc)

	fun_pred,_ = fun_model.viterbi_decode_bigram(sequence)
	fun_pred.sequence_list.seq_list[0].y = fun_pred.y

	jobarea_pred,_ = jobarea_model.viterbi_decode_bigram(sequence)
	jobarea_pred.sequence_list.seq_list[0].y = jobarea_pred.y

	sentence_map = np.zeros(len(fun_pred.x),dtype=np.int8)
	k=2
	for sent in doc:
		sequence = makeSequence(sent)
		req_pred,_ = req_model.viterbi_decode_bigram(sequence)
		req_pred.sequence_list.seq_list[0].y = req_pred.y

		req_chunks = ChunkSet(req_pred.sequence_list)
		for chunk in req_chunks.chunk_list:
			seq = req_pred.sequence_list[chunk.sequence_id]
			pos = chunk.pos-2
			fin = min(chunk.pos+chunk.length,len(seq.x)-1)-2 # [pos - fin >
			sentence_map[k+pos]+=1
			sentence_map[k+fin]-=1

		carr_pred,_ = carr_model.viterbi_decode_bigram(sequence)
		carr_pred.sequence_list.seq_list[0].y = carr_pred.y

		carr_chunks = ChunkSet(carr_pred.sequence_list)
		for chunk in carr_chunks.chunk_list:
			seq = carr_pred.sequence_list[chunk.sequence_id]
			pos = chunk.pos-2
			fin = min(chunk.pos+chunk.length,len(seq.x)-1)-2 # [pos - fin >
			sentence_map[k+pos]+=1
			sentence_map[k+fin]-=1

		k+=len(sequence.x)-3+1

	fun_chunks = ChunkSet(fun_pred.sequence_list)
	jobarea_chunks = ChunkSet(jobarea_pred.sequence_list)
	for chunk in fun_chunks.chunk_list:
		seq = fun_pred.sequence_list[chunk.sequence_id]
		sentence_map[chunk.pos]+=1
		sentence_map[chunk.pos+chunk.length]-=1
	for chunk in jobarea_chunks.chunk_list:
		seq = jobarea_pred.sequence_list[chunk.sequence_id]
		sentence_map[chunk.pos]+=1
		sentence_map[chunk.pos+chunk.length]-=1

	if sum(sentence_map==0)<=20:
		# empty
		return None
	#temp = [pos for pos in range(sentence_map.size) if sentence_map[pos]]
	#print(fun_pred)
	#print(temp)
	final_text = ''
	new_doc=[]
	sent = []
	acum = 0
	newline=True
	for i in range(2,len(fun_pred.x)):
		word = fun_pred.sequence_list.x_dict.get_label_name(fun_pred.x[i])
		if word==BR or word==END:
			if not newline:
				final_text += '\n'
				#new_doc.append(sent)
			#sent=[]
			newline=True
			continue
		acum += sentence_map[i]
		if acum>0:
			#sent.append(word)
			if not newline:
				final_text += ' '
			final_text += word
			newline=False
	if len(final_text)>100:
		open(os.path.join(target_dir,docname),'w').write(final_text)

	#ipdb.set_trace()

	return None




if __name__ == '__main__':
	print("Reading ing names...")
	files_ingenieria = []
	for level1 in hierarchy["children"]:
		if level1["name"]=='Ingeniería':
			for career in level1["children"]:
				name = career["name"]
				files_ingenieria.append(fileByName[name])

	or_query = [{'carreras':file_name} for file_name in files_ingenieria]
	data = db_prueba.find({'$or':or_query})

	print("Getting doc names...")
	names = [p["name"] for p in data]

	print("Running main loop...")
	parallelize = True
	
	startTime = datetime.now()
	if parallelize:
		pool = Pool(processes=10)
		res = pool.map(parallel_funct,names)
		pool.close()
		pool.join()
	else:
		count = 0
		for docname in names:
			parallel_funct(docname)
			if count%1==0:
				print("->",count)
			count+=1
			if count>=10:
				break
	print("Execution time: ",datetime.now()-startTime)

	#################################################################################
	"""
	stem=True

	data = db_prueba.find({'$or':or_query})
	freqdist_ing = make_freqdist(data, docs_dir,word_dict_filtered,stem=stem)

	#ipdb.set_trace()

	data = db_prueba.find({'$or':or_query})
	title_map_file = open(os.path.join(data_dir,'title_map.dat'),'w')
	doc_title_map = {}

	doc_term_freq_file = open(os.path.join(data_dir,'word_counts.dat'),'w')
	vocab_file = open(os.path.join(data_dir,'vocab.dat'),'w')

	frequency_by_doc = []

	count = 0
	n_docs = 0
	for root, dirs, filenames in os.walk(target_dir):
		for f in filenames:
			if f[-1]!='~':
				#print("Doc:",f)
				doc = readDoc(os.path.join(target_dir,f))
				n_docs += 1
				name_entities = []
				for sent in doc:
					sequence = makeSequence(sent)
					pred_sequence,_ = model.viterbi_decode_bigram(sequence)
					pred_sequence.sequence_list.seq_list[0].y = pred_sequence.y

					nes = getNameEntities(pred_sequence)
					name_entities.extend(nes)

				carreras = discretizeCareers(name_entities)
				if carreras==[]:
					print(name_entities)
					print(carreras)
					ipdb.set_trace()
				else:
					try:
						db_prueba.insert({'name':f, 'carreras':carreras})
					except:
						print("NO INSERTO")

				if count%100==0:
					print('-->',count)
				count +=1
	"""