import pickle
import io
import ipdb

def saveObject(obj, name='model'):
    with open(name + '.pickle', 'wb') as fd:
        pickle.dump(obj, fd, protocol=pickle.HIGHEST_PROTOCOL)

folder = 'NE_ene_ing/'

title_doc = {}
for doc in open('counts/'+folder+'title_map.dat','r'):
	doc = doc.strip('\n')
	if doc != '':
		try:
			title = io.open('docs/full_ene_ing/'+doc,'r',encoding='utf-8').readline()
		except:
			ipdb.set_trace()
			title = '--bad encoding--'
		title = title.strip('\n').strip('<br>').replace('<','.').replace('>','.').replace('/','_SLASH_')
		title = title[:40]
		title_doc[doc] = title

saveObject(title_doc,'doc_title_map')
