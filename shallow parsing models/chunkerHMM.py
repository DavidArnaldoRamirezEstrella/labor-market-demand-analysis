from utils_hmm import *
from nltk.tag.hmm import HiddenMarkovModelTagger
from sklearn.cross_validation import train_test_split
from metrics import *

careers_dir = "careers tagged/"
random_dir = "random/"
BR = '**'

import sequence_lxmls.hmm as hmmc
from sequences.log_domain import *
from sequences.sequence_list import *

import sequences.confusion_matrix as cm
import matplotlib.pyplot as plt
import numpy as np
import ipdb

from sklearn.metrics import classification_report, accuracy_score

def read_sequence_list(target='BIO'):
    """
        :param target: BIO : IBO tagging, Y = B,I,O
                       NE : Y = NE names
        :return: list of sentences
        """
    seq_list = []
    for i in range(100, 401):
        sent_x = []
        sent_y = []
        sent_pos = []
        for line in open(careers_dir + str(i) + '.tsv'):
            line = line.strip('\n')
            if len(line) > 0:
                temp = line.split('\t')
                pos = temp[1]
                x = temp[0]
                if target == 'BIO':
                    y = temp[-1][0]
                else:
                    y = temp[-1]  # temp[-1][2:]
            else:
                x, y, pos = (BR, BR, BR)
            sent_x.append(x)
            sent_y.append(y)
            sent_pos.append(pos)
        if x == BR:
            sent_x.pop()
            sent_y.pop()
            sent_pos.pop()

        seq_list.append(zip(sent_x, sent_y))

    for i in range(1, 401):
        sent_x = []
        sent_y = []
        sent_pos = []
        for line in open(random_dir + str(i) + '.tsv'):
            line = line.strip('\n')
            if len(line) > 0:
                temp = line.split('\t')
                x = temp[0]
                pos = temp[1]

                if target == 'BIO':
                    y = temp[-1][0]
                else:
                    y = temp[-1]  # temp[-1][2:]
            else:
                x, y, pos = (BR, BR, BR)
            sent_x.append(x)
            sent_y.append(y)
            sent_pos.append(pos)
        if x == BR:
            sent_x.pop()
            sent_y.pop()
            sent_pos.pop()

        seq_list.append(zip(sent_x, sent_y))

    ret = []
    for sent in seq_list:
        tt = []
        for word in sent:
            tt.append(word)
        ret += [tt]

    return ret


def train_test_data(data, size_p=0.1):
    tn, tt = train_test_split(data, test_size=size_p)
    return tn, tt


def train_and_test(train_data, test_data, estimator):
    """
    :train : training dataset
    :test : testing dataset
    :est  : NLTK Prob object
    """
    tag_set = list(set([tag for sent in train_data for (word, tag) in sent]))
    symbols = list(set([word for sent in train_data for (word, tag) in sent]))

    # trainer HMM
    trainer = nltk.HiddenMarkovModelTrainer(tag_set, symbols)
    hmm_model = trainer.train_supervised(train, estimator=estimator)
    print(hmm_model.tag(test_data[10]))
    res = 100 * hmm_model.evaluate(test_data)

    return res


"""
    Usando Libreria NLTK
"""

'''
# estimadores prob
###
mle = lambda fd, bins: MLEProbDist(fd)
laplace = LaplaceProbDist
#ele = ELEProbDist
witten = WittenBellProbDist
gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)
###

corpus = read_sequence_list(target='ALL')
train, validation = train_test_data(corpus, size_p=0.2)
req_diff = set()
for sent in corpus:
    #print(par)
    for par in sent:
        if par[1][2:] == 'REQ':
            req_diff.add(par)

print(len(req_diff))
print(req_diff)

print("Entrenando HMM default...")
hmm = HiddenMarkovModelTagger.train(train)

sentence = validation[100]
prediction = hmm.tag(sentence)

print('prediction\n', prediction)
print('sentence\n', sentence)

print("Evaluando HMM default...")
print('HMM: %.4f %%' % (hmm.evaluate(validation) * 100))

#print("Diferentes estimadores:")
print('Turing: %.4f %%' % train_and_test(train, validation, gt))
#print("Laplace:"), train_and_test(train, validation, laplace)
print("MLE: %.4f %%" % train_and_test(train, validation, mle))

'''
"""
    Usando Libreria de LxMLS
"""





# ipdb.set_trace()

def pick_best_smoothing_f1(hmm, dataset, train, test, smooth_values):
    max_smooth = 0
    max_acc = 0
    score_train = MyChunkScore(dataset, mode='TODO')
    score_test = MyChunkScore(dataset, mode='TODO')
    for i in smooth_values:
        hmm.train_supervised(train, smoothing=i)
        viterbi_pred_train = hmm.viterbi_decode_corpus_bigram(train)
        eval_viterbi_train = hmm.evaluate_corpus(train, viterbi_pred_train)
        score_train.evaluate(train, viterbi_pred_train)
        print ("Smoothing %f --  Train Set Accuracy: Viterbi Decode: %.3f" % (i, eval_viterbi_train))
        print(score_train)

        viterbi_pred_test = hmm.viterbi_decode_corpus_bigram(test)
        eval_viterbi_test = hmm.evaluate_corpus(test, viterbi_pred_test)
        score_test.evaluate(test, viterbi_pred_test)
        print ("Smoothing %f --  Train Set Accuracy: Viterbi Decode: %.3f" % (i, eval_viterbi_test))
        print(score_test)

        if score_test.f_measure() > max_acc:
            max_acc = score_test.f_measure()
            max_smooth = i

        print("")
    return max_smooth

#mode = 'by_sent'
mode = 'by_doc'
train, test, val = getData_HMM(test=0.1, val=0.1, tags=['JOB','AREA'], mode=mode, filter_empty=True)
hmm = hmmc.HMM(train.x_dict, train.y_dict)

# best_smooth = hmm.pick_best_smoothing_bigram(train, val, [10, 1, 0.1, 0.01, 0.001, 0])
# values = np.arange(0,4.2,0.1)
# best_smooth = pick_best_smoothing_f1(hmm, dataset, train, test, values)
# print("Best Smooth was: %.2f" % best_smooth)

#smooths = [1]
smooths = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
for smooth in smooths:
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("SMOOTH:", smooth)
    hmm.train_supervised(train, smoothing=smooth)
    # hmm.print_transition_matrix()
    pred_train = hmm.viterbi_decode_corpus(train)
    pred_test = hmm.viterbi_decode_corpus(test)
    pred_val = hmm.viterbi_decode_corpus(val)

    temp = [(v,k) for k,v in train.y_dict.items() if k in ['B','I','O']]
    temp.sort()
    names_train = [k for v,k in temp]

    temp = [(v,k) for k,v in val.y_dict.items() if k in ['B','I','O']]
    temp.sort()
    names_val = [k for v,k in temp]

    temp = [(v,k) for k,v in test.y_dict.items() if k in ['B','I','O']]
    temp.sort()
    names_test = [k for v,k in temp]

    Y_train = join_data_tags_bio(train.seq_list)
    Y_val   = join_data_tags_bio(val.seq_list)
    Y_test  = join_data_tags_bio(test.seq_list)

    Y_train_pred = join_data_tags_bio(pred_train)
    Y_val_pred   = join_data_tags_bio(pred_val)
    Y_test_pred  = join_data_tags_bio(pred_test)

    ipdb.set_trace()

    print("Metrics: Training data")
    print(classification_report(Y_train, Y_train_pred, target_names=names_train))
    print("Accuracy: ",accuracy_score(Y_train,Y_train_pred))

    print("Metrics: Validation data")
    print(classification_report(Y_val  , Y_val_pred  , target_names=names_val))
    print("Accuracy: ",accuracy_score(Y_val,Y_val_pred))

    print("Metrics: Testing data")
    print(classification_report(Y_test  , Y_test_pred  , target_names=names_test))
    print("Accuracy: ",accuracy_score(Y_test,Y_test_pred))

#saveObject(hmm, 'HMM_FUN_1_smooth')

"""
def plot_error_vs_data(hmm, train):
    X = np.arange(10, 101, 5) / 100.0
    J_train = np.zeros(X.size)
    J_val = np.zeros(X.size)

    for i, train_size in enumerate(X):
        new_train = reader.trimTrain(train, train_size)
        print("Training dataset size: ", train_size * 0.6, new_train.size())
        hmm.train_supervised(new_train, smoothing=0.1)
        pred_train = hmm.viterbi_decode_corpus_bigram(new_train)
        pred_val = hmm.viterbi_decode_corpus_bigram(val)
        J_train[i] = 100 * (1.0 - hmm.evaluate_corpus(new_train, pred_train))
        J_val[i] = 100 * (1.0 - hmm.evaluate_corpus(val, pred_val))

    plt.plot(X, J_train, 'r', X, J_val, 'b')
    plt.show()


def plot_error_vs_smoothing(hmm, train):
    hmm.graph_error_smoothing(train, val, 0, 3.1, 0.1)

"""