import os,sys

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_utils)

from utils_new import *
import sequences.structured_perceptron as spc
import id_feature_bigram_job as idf
import ext2_sp as exfc2

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    print("Reading data...")
    mode = 'by_sent'
    #mode = 'by_doc'
    train,test,val = getData(test=0.1, val=0.1, tags = ['JOB','AREA'], mode=mode,filter_empty=True)

    print("Building features...")
    feature_mapper = idf.IDFeatures(train,mode=mode)
    #feature_mapper = exfc2.ExtendedFeatures(train,mode=mode)
    feature_mapper.build_features()

    #pdb.set_trace()
    
    print("Init Struc Perceptron")
    epochs = 5
    learning_rate = 1
    reg = 0    # extended feat
    #reg = 0.001
    sp = spc.StructuredPerceptron(  train.x_dict, train.y_dict, feature_mapper,
                                    num_epochs=epochs, learning_rate=learning_rate,
                                    reg_param=reg)
    print("Training...")
    sp.train_supervised_bigram(train)

    print("Predicting...")
    print("::Training...")
    pred_train = sp.viterbi_decode_corpus_bigram(train)
    print("::Valalidation...")
    pred_val   = sp.viterbi_decode_corpus_bigram(val)
    print("::Testing...")
    pred_test  = sp.viterbi_decode_corpus_bigram(test)

    print("Evaluating...")
    eval_train = sp.evaluate_corpus(train, pred_train)
    eval_val   = sp.evaluate_corpus(val, pred_val, DEBUG=False)
    eval_test  = sp.evaluate_corpus(test, pred_test)
    print("Structured Perceptron - Features Accuracy Train: %.4f | Val: %.4f | Test: %.4f" % (eval_train, eval_val, eval_test) )

    print()
    print("Metrics: Training data")
    cs_train = MyChunkScore(train)
    cs_train.evaluate(train,pred_train)
    print(cs_train)

    print()

    print("Metrics: Validation data")
    cs_val = MyChunkScore(val)
    cs_val.evaluate(val,pred_val)
    print(cs_val)

    print()

    print("Metrics: Testing data")
    cs_test = MyChunkScore(test)
    cs_test.evaluate(test,pred_test)
    print(cs_test)
    

    ###################################################################################################################
    print("==========================================================")
    print("Confussion Matrix: sapeee!") # DICT HAS TO BE FROM THE SAME PART (TRAIN, VAL OR TEST)
    conf_matrix = cm.build_confusion_matrix(val.seq_list, pred_val, sp.get_num_states())
    for id,_dict in conf_matrix.items():
        name = val.y_dict.get_label_name(id)
        print("::",name)
        for k,v in _dict.items():
            name_in = val.y_dict.get_label_name(k)
            print("  %s: %i" % (name_in,v))

    #cm.plot_confusion_bar_graph(conf_matrix, val.y_dict, range(sp.get_num_states()), 'Confusion matrix')

    ###################################################################################################################
    model = 'sp_%i_%s' % (epochs,mode)
    print("Saving model with name:",model)
    saveObject(sp,model)
    
    ###################################################################################################################

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
    Y_test   = join_data_tags_bio(test.seq_list)

    Y_train_pred = join_data_tags_bio(pred_train)
    Y_val_pred   = join_data_tags_bio(pred_val)
    Y_test_pred   = join_data_tags_bio(pred_test)

    print("Metrics: Training data")
    print(classification_report(Y_train, Y_train_pred, target_names=names_train))
    print("Accuracy: ",accuracy_score(Y_train,Y_train_pred))

    print("Metrics: Validation data")
    print(classification_report(Y_val  , Y_val_pred  , target_names=names_val))
    print("Accuracy: ",accuracy_score(Y_val,Y_val_pred))

    print("Metrics: Testing data")
    print(classification_report(Y_test  , Y_test_pred  , target_names=names_test))
    print("Accuracy: ",accuracy_score(Y_test,Y_test_pred))


    print("Debugging!!!")
    ipdb.set_trace()

    print("Sapee!!!")        