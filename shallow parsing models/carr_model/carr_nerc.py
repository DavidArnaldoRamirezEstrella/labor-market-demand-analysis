__author__ = 'ronald'

import matplotlib.pyplot as plt
import numpy as np
import ipdb, pdb

from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm, grid_search

from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.lda import LDA
from scipy import sparse

from prueba_nec import plot2D, plot3D

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_utils)

from utils_new import *
import features_nerc as idf


import sequences.confusion_matrix as cm

if __name__ == '__main__':
    print("Reading data...")
    mode = 'by_sent'
    #mode = 'by_doc'
    #train,test,val = getData(tags = ['CARR'], mode=mode,filter_empty=False)
    train,test,val = getData(test=0.1, val=0.1, tags = ['CARR'], mode=mode)

    print("Building features...")
    idf = exfc_nec.ExtendedFeatures(train,mode=mode)
    idf.build_features()

    print("Standarizing dataset...")
    X_train,Y_train = getStandart_NERC(train, idf)
    X_val  ,Y_val   = getStandart_NERC(val  , idf)
    X_test ,Y_test  = getStandart_NERC(test , idf)

    # normalize
    X_train = normalize(X_train, copy = False)
    X_val   = normalize(X_val, copy=False)
    X_test  = normalize(X_test, copy=False)

    ###############################################################################

    print("Dimesionality reduction SVD...")
    svd = SVD(n_components=1000)
    svd.fit(X_train)

    X_train = svd.transform(X_train)
    X_val = svd.transform(X_val)
    X_test = svd.transform(X_test)

    """
    print("Dimesionality reduction LDA...")
    lda = LDA(n_components = 100)
    lda.fit(X_train, Y_train)
    X_train = lda.transform(X_train)
    X_val   = lda.transform(X_val)
    X_test = lda.transform(X_test)
    """

    
    ###############################################################################
    C_vals     = [0.1, 1, 10, 100, 1000,10000,100000]
    gamma_vals = [0.1, 0.01, 0.001, 0.0001]
    
    ###############################################################################
    print("Classes quantities")
    n_train = len(Y_train)
    n_val = len(Y_val)
    n_test = len(Y_test)

    print("  Training")
    print("  words:",n_train)
    for label in label_names:
        label_id = train.y_dict.get_label_id(label)
        print("  %s : %.3f" % (label, sum(Y_train == label_id)/n_train))
    print("\n---------------------------------")

    print("  Testing")
    print("  words:",n_test)
    for label in label_names:
        label_id = test.y_dict.get_label_id(label)
        print("  %s : %.3f" % (label, sum(Y_test == label_id)/n_test))
    print("\n---------------------------------")
    
    
    print("  Validation")
    print("  words:",n_val)
    for label in label_names:
        label_id = val.y_dict.get_label_id(label)
        print("  %s : %.3f" % (label, sum(Y_val == label_id)/n_val))
    print("\n---------------------------------")

    best_param = []
    best_acc = 0

    ###############################################################################
    print("Validation metrics...")
    for c in C_vals:
        for g in gamma_vals:
            svmc = svm.SVC(kernel = 'rbf', C=c, gamma=g)
            svmc.fit(X_train, Y_train)
            
            ST_id = val.y_dict.get_label_id(START_TAG)
            BR_id = val.y_dict.get_label_id(BR)
            BR_x_id = val.x_dict.get_label_id(BR)
            
            pred_val = []
            for sequence in val.seq_list:
                for pos in range(2,len(sequence.x)-1):
                    x = sequence.x[pos]
                    if x == BR_x_id:
                        continue

                    y_1,y_2 = '',''
                    if pos == 2:
                        y_1,y_2 = ST_id,ST_id
                    elif pos == 3:
                        y_1 = pred_val[-1]
                        y_2 = ST_id
                    else:
                        y_1 = pred_val[-1]
                        y_2 = pred_val[-2]

                    y = sequence.y[pos]
                    features = idf.get_features(sequence, pos, y_1, y_2)

                    X_m = np.zeros((1,idf.get_num_features()))
                    X_m[0,features] = 1

                    X_m = svd.transform(X_m)
                    #X_m = lda.transform(X_m)

                    y_m = svmc.predict(X_m)[0]
                    pred_val.append(y_m)
            acc = sum(Y_val == pred_val)/n_val
            if acc > best_acc:
                best_param = [c,g]
                best_acc = acc
            print("Metrics Validation data...")
            print("Parameters: C: %i | gamma: %f" % (c,g))
            print(classification_report(Y_val, pred_val, target_names=label_names))
            print("Accuracy total: %.3f" % acc)
            print("##################################")


    print("Best accuracy: ",best_acc)
    print("Best param: ",best_param)

    
    """
    ### PRUEBA EN TEST DATA
    print("Entrenando...")
    svmc = svm.SVC(kernel = 'rbf', C=100, gamma=0.0001)
    svmc.fit(X_train, Y_train)
    
    ST_id = val.y_dict.get_label_id(START_TAG)
    BR_id = val.y_dict.get_label_id(BR)
    BR_x_id = val.x_dict.get_label_id(BR)
    
    print("Testeando...")
    pred_test = []
    for sequence in test.seq_list:
        for pos in range(2,len(sequence.x)-1):
            x = sequence.x[pos]
            if x == BR_x_id:
                continue

            y_1,y_2 = '',''
            if pos == 2:
                y_1,y_2 = ST_id,ST_id
            elif pos == 3:
                y_1 = pred_test[-1]
                y_2 = ST_id
            else:
                y_1 = pred_test[-1]
                y_2 = pred_test[-2]

            y = sequence.y[pos]
            features = idf.get_features(sequence, pos, y_1, y_2)

            X_m = np.zeros((1,idf.get_num_features()))
            X_m[0,features] = 1

            X_m = svd.transform(X_m)
            #X_m = lda.transform(X_m)

            y_m = svmc.predict(X_m)[0]
            pred_test.append(y_m)
    acc = sum(Y_test == pred_test)/n_test
    print("Metrics Testing data...")
    print(classification_report(Y_test, pred_test, target_names=label_names))
    print("Accuracy total: %.3f" % acc)


    pred_seq_test = toSeqList(test,pred_test)

    print("Confussion Matrix: sapeee!")
    n_states = len(data.y_dict)
    confusion_matrix = cm.build_confusion_matrix(test.seq_list, pred_seq_test, n_states)

    cm.plot_confusion_bar_graph(confusion_matrix, data.y_dict, range(n_states), 'Confusion matrix')
    plt.show()
    """

