import os, sys
import json
import copy
import numpy as np
import random
from multiprocessing import Pool
import ipdb

################################################################################################
utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'nlp scripts')

source_vh_dir = '/home/ronotex/Downloads/vector_hash/ingenierias_mineria'
#source_vh_dir = '/home/ronotex/Downloads/vector_hash/mantenimiento_en_minernia'

#treemap_name = 'carreras_rubro_mina'
#adj_name = 'ing_total_adjmatrix'

treemap_name = 'carreras_mantto_mina'
adj_name = 'mantto_mina_adjmatrix'


class LabelDict(dict):
	def __init__(self, label_names=[]):
		self.names = []
		for name in label_names:
			self.add(name)

	def add(self, name):
		label_id = len(self.names)
		if name in self:
			#warnings.warn('Ignoring duplicated label ' +  name)
			return self[name]
		self[name] = label_id
		self.names.append(name)
		return label_id

	def get_label_name(self, label_id):
		return self.names[label_id]

	def get_label_id(self, name):
		if name not in self:
			return -1
		return self[name]

	def size(self):
		return len(self.names)


################################################################################################
hierarchy = json.loads(open('carreras_ing2.json').read())

# docname : {docname : true name}
nameByFile = json.loads(open('ident_names2.json').read())
fileByName = {}
temp={}
for (file,name) in nameByFile.items():
    temp[file.strip(' ')] = name.strip(' ')
    fileByName[name.strip(' ')] = file.strip(' ')
nameByFile = dict(temp)

################################################################################################

def sorter(T,sizeById, file_dict):
	if "children" not in T:
		_id = file_dict.get_label_id(fileByName[T["name"]])
		try:
			T["size"] = int(sizeById[_id])
		except:
			T["size"] = sizeById[_id]
		return float(T["size"])

	children = T["children"]
	temp = []
	_total = 0
	for child in children:
		subt_sum = sorter(child,sizeById, file_dict)
		_total += subt_sum
		
		temp.append(subt_sum)
	temp = list(zip(temp,range(len(children))))
	temp.sort(reverse=True)
	T["children"] = [children[k[1]] for k in temp]
	return _total

def getSortedLeaves(T, V,file_dict):
	if "children" not in T:
		fn = fileByName[ T["name"] ]
		V.append(file_dict.get_label_id(fn))
		return
	for child in T["children"]:
		getSortedLeaves(child,V,file_dict)

################################################################################################
################################################################################################

if __name__=='__main__':
	vh_dict = LabelDict()
	file_dict = LabelDict()
	graph = np.zeros([30,30])
	vhsByFile = [set() for i in range(30)]
	freq_major = np.zeros([30])
	


	for root,dirs,filenames in os.walk(source_vh_dir):
		for f in filenames:
			if f[-1]!='~':
				#file_name = f[3:] # vh_name
				#if file_name=='all' or file_name=='ing':
				#	continue
				p = f.find('_mineria')
				#p = f.find('_mantto_mineria')
				file_name = f[3:p] # vh_name_mineria
				#file_name = f[14:] # mantto_min_vh_name
				id_file = file_dict.add(file_name)
				for line in open(os.path.join(source_vh_dir,f)):
					line = line.strip('\n')
					if line!='':
						id_vh = vh_dict.add(line)
						freq_major[id_file]+=1
						vhsByFile[id_file].add(id_vh)
	
	count_id_vh = vh_dict.size()
	count_id_file = file_dict.size()
	print(count_id_vh)
	print(count_id_file)
	ipdb.set_trace()

	# node					
	for k in range(count_id_file):
		# posible edges
		outgoing = set()
		for i in range(count_id_file):
			if k!=i:
				temp = vhsByFile[k] & vhsByFile[i]
				graph[k,i] = len(temp)
				outgoing |= temp
		graph[k,k] = freq_major[k] - len(outgoing)

	# GENERATE CARRERAS.JSON

	tot = sorter(hierarchy,freq_major,file_dict)
	open(treemap_name+'.json','w').write(json.dumps(hierarchy,ensure_ascii=False, indent = 2))

	
	per_hierarchy = dict(hierarchy)
	temp = [format(x,'.2f') for x in 100.0*freq_major/count_id_vh]
	
	tot = sorter(per_hierarchy,temp,file_dict)
	open(treemap_name+'_perc.json','w').write(json.dumps(per_hierarchy,ensure_ascii=False, indent = 2))

	# GENERATE ADJMATRIX.JSON
	sorted_ids = []
	getSortedLeaves(hierarchy,sorted_ids,file_dict)
	adjmatrix = []
	for k in sorted_ids:
		if freq_major[k]==0:
			continue
		u = file_dict.get_label_name(k)
		item = dict()
		item["name"] = nameByFile[u]
		item["size"] = int(freq_major[k])
		item["imports"] = []

		for i in sorted_ids:
			if graph[k,i]>0:
				v = file_dict.get_label_name(i)
				imp = dict({'name':nameByFile[v],'weight':int(graph[k,i])})
				item["imports"].append(imp)
		adjmatrix.append(item)

	open(adj_name + '.json','w').write(json.dumps(adjmatrix,ensure_ascii=False, indent = 2))