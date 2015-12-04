import os, sys
import json
import ipdb

from utils_pipe import *

utils_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(utils_path)
from utils_new import uploadObject

#################################################################################
BASE_DIR = os.path.dirname( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
clustering_dir = os.path.join(BASE_DIR,'dataR')

#_dir = 'NE_ene_ing'
#_dir = 'full_ene_ing'
_dir = 'NE_ene_ing_sp_hmm'
target_dir = os.path.join(clustering_dir,'docs/'+_dir)
counts_dir = os.path.join(clustering_dir,'counts/'+_dir)

#titles_path = os.path.join(clustering_dir,'counts/NE_ene_ing/title_map.dat')
titles_path = os.path.join(clustering_dir,'counts/' + _dir + '/title_map.dat')
post_filenames = [line.strip('\n') for line in open(titles_path) if len(line.strip('\n'))>0]


if not os.path.exists(counts_dir):
	os.makedirs(counts_dir)

#################################################################################
word_dict_filtered = uploadObject(os.path.join(utils_path,'word_dict_filtered'))

#################################################################################

if __name__ == '__main__':
	stem=False

	#filenames = [f for r,d,fn in os.walk(target_dir) for f in fn if f[-1]!='~']
	filenames = post_filenames

	freqdist_ing = make_freqdist(filenames,target_dir,word_dict_filtered,stem=stem,THR=3)

	title_map_file = open(os.path.join(counts_dir,'title_map.dat'),'w')
	doc_title_map = {}

	doc_term_freq_file = open(os.path.join(counts_dir,'word_counts.dat'),'w')
	vocab_file = open(os.path.join(counts_dir,'vocab.dat'),'w')

	frequency_by_doc = []

	count = 0
	for f in filenames:
		tokens = readDoc(os.path.join(target_dir,f))
		if len(tokens)==0:
			continue
		filtered_tokens = filterTokens(tokens,freqdist_ing)
		# Title map and doc_title mapping
		title = ' '.join(tokens[0])[:50]
		
		doc_title = f
		title_map_file.write(doc_title+'\n')
		doc_title_map[doc_title] = title
		## word_counts by doc
		word_frequency = make_term_frequency(filtered_tokens,freqdist_ing,stem=stem)
		frequency_by_doc.append(word_frequency)

		if count%500==0:
			print('-->',count)
		count +=1
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