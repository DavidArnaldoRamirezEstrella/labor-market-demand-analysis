__author__ = 'ronald'
import nltk
import ipdb
from nltk.chunk.util import ChunkScore
# from utils import *

BR = '**'

class MyChunkScore(ChunkScore):
    def __init__(self, dataset, mode = 'BIO'):
        self._correct = set()
        self._guessed = set()
        self._tp = set()
        self._fp = set()
        self._fn = set()
        self._tp_num = 0
        self._fp_num = 0
        self._fn_num = 0
        self._count = 0
        self._tags_correct = 0.0
        self._tags_total = 0.0

        self._chunksets = ''
        if mode == 'BIO':
            self._chunksets = self._chunksets_BIO
        else:
            self._chunksets = self._chunksets_TODO

        self.dataset = dataset


    def score(self, correct, guessed):
        '''
        :param correct: sequence object with Gold data (true)
        :param guessed: sequence object with predicted tags
        :return:
        '''
        self._correct |= self._chunksets(correct, self._count)
        self._guessed |= self._chunksets(guessed, self._count)
        self._count += 1
        self._measuresNeedUpdate = True
        # Keep track of per-tag accuracy (if possible)
        correct_tags = list(zip(correct.x, correct.y))
        guessed_tags = list(zip(guessed.x, guessed.y))

        self._tags_total += len(correct_tags)
        self._tags_correct += sum(1 for i, k in enumerate(correct_tags) if correct_tags[i] == guessed_tags[i])


    def _chunksets_BIO(self, sequence, count):
        '''
        :param sequence: sequence object
        :param count: contador para poner IDs
        :return:list de {(id,pos) , chunk-struct }
        '''
        chunks = []
        pos = 0
        open = False
        n = len(sequence.x)
        for (i, w) in enumerate(sequence.x):
            tag = sequence.sequence_list.y_dict.get_label_name(sequence.y[i])
            if tag == 'B':
                if open and i>0:
                    temp = ( (count, pos), tuple(zip(sequence.x[pos:i], sequence.y[pos:i])) )
                    chunks.append(temp)
                pos = i
                open = True
            elif tag != 'I' and open:
                open = False
                temp = ( (count, pos), tuple(zip(sequence.x[pos:i], sequence.y[pos:i])) )
                chunks.append(temp)
        if open:
            temp = ((count, pos), tuple(zip(sequence.x[pos:n], sequence.y[pos:n])) )
            chunks.append(temp)
        return set(chunks)


    def _chunksets_TODO(self, sequence, count):
        '''
        :param sequence: sequence object
        :param count: contador para poner IDs
        :return:list de {(id,pos) , chunk-struct }
        '''
        chunks = []
        pos = 0
        open = False
        n = len(sequence.x)
        for (i, w) in enumerate(sequence.x):
            tag = sequence.sequence_list.y_dict.get_label_name(sequence.y[i])
            if len(tag)>1:
                ne = tag[2:]
            else:
                ne = tag
            ne = tag[2:]
            prev_ne = ne
            if i>0:
                prev_tag = sequence.sequence_list.y_dict.get_label_name(sequence.y[i-1])
                if len(tag)>1:
                    prev_ne = prev_tag[2:]
            if tag.find('B')==0:
                if open and i>0:
                    temp = ( (count, pos), tuple(zip(sequence.x[pos:i], sequence.y[pos:i])) )
                    chunks.append(temp)
                pos = i
                open = True
            elif tag.find('I') != 0 and open:
                open = False
                #temp = ( (count, pos), tuple(zip(sequence.x[pos:i + 1], sequence.y[pos:i + 1])) )
                temp = ( (count, pos), tuple(zip(sequence.x[pos:i], sequence.y[pos:i])) )
                chunks.append(temp)
        if open:
            temp = ((count, pos), tuple(zip(sequence.x[pos:n], sequence.y[pos:n])) )
            chunks.append(temp)
        return set(chunks)


    def evaluate(self, gold, test):
        '''
        :param gold: seq_list object (true tags)
        :param test: (list) seq_list (predicted tags) !!!!
        :return:
        '''
        for i, correct in enumerate(gold.seq_list):
            guessed = test[i]
            self.score(correct, guessed)


if __name__ == '__main__':
    """
	reader = JobDBCorpus()
	dataset = reader.read_sequence_list(target = 'BIO')
	train_seq,test_seq = reader.train_test_data(test_size=0.1)

	cs = MyChunkScore(dataset)

	cs.evaluate(train_seq,train_seq)
	print(cs)
	"""