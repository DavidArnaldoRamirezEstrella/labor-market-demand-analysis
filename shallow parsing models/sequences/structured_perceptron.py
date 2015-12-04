import sys,os
import numpy as np
import sequences.discriminative_sequence_classifier as dsc
import ipdb

from multiprocessing import Pool
from functools import partial
from datetime import datetime

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_utils)

from utils_new import NPROCESSORS,PARALLELIZE

class StructuredPerceptron(dsc.DiscriminativeSequenceClassifier):
    ''' Implements a first order CRF'''

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 num_epochs = 10, learning_rate = 1.0, reg_param = 0, averaged = True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.regularization_param = reg_param
        self.params_per_epoch = []

        print("Epochs: ", num_epochs)
        print("Learning rate: ",learning_rate)
        print("Reg param: ", reg_param)
        print("=======================================")


    def train_supervised_bigram(self, dataset):
        startTime = datetime.now()

        self.y_dict = dataset.y_dict
        self.x_dict = dataset.x_dict
        self.decoder.init_dicts(dataset.x_dict,dataset.y_dict)
        
        # BIGRAMAS
        self.parameters = np.zeros(self.feature_mapper.get_num_features())

        num_examples = dataset.size()
        acc = 0
        for epoch in range(self.num_epochs):
             num_labels_total = 0
             num_mistakes_total = 0
             for i in range(num_examples):
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update_bigram(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
             self.params_per_epoch.append(self.parameters.copy())
             acc = 1.0 - float(num_mistakes_total)/float(num_labels_total)
             print("Epoch: %i Accuracy: %f" %(epoch, acc))
        #print("-- Last epoch accuracy: %f" % acc )
        self.trained = True

        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w = new_w / len(self.params_per_epoch)
            self.parameters = new_w
        print("---------------------------------------")
        print("Execution time: ",datetime.now()-startTime)



    def train_supervised(self, dataset):
        self.y_dict = dataset.y_dict
        self.x_dict = dataset.x_dict
        self.decoder.init_dicts(dataset.x_dict,dataset.y_dict)

        # TRIGRAMAS
        self.parameters = np.zeros(self.feature_mapper.get_num_features())

        num_examples = dataset.size()
        acc = 0
        for epoch in range(self.num_epochs):
             num_labels_total = 0
             num_mistakes_total = 0
             for i in range(num_examples):
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
             self.params_per_epoch.append(self.parameters.copy())
             acc = 1.0 - float(num_mistakes_total)/float(num_labels_total)
             print("Epoch: %i Accuracy: %f" %(epoch, acc))
        #print("-- Last epoch accuracy: %f" % acc )
        self.trained = True

        if(self.averaged == True):
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w = new_w / len(self.params_per_epoch)
            self.parameters = new_w


    def perceptron_update(self, sequence, reg_parameter=0):
        reg = self.regularization_param

        num_labels = 0
        num_mistakes = 0
        true_features = []
        hat_features = []

        predicted_sequence, _ = self.viterbi_decode(sequence)
        y_hat = predicted_sequence.y

        param_temp = self.parameters

        for pos in range(2,len(sequence.x)):
            y_t_true = sequence.y[pos]
            y_t_hat = y_hat[pos]

            # Update emission features.
            num_labels += 1
            if y_t_true != y_t_hat:
                num_mistakes += 1
                true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true)
                #true_features.extend(true_emission_features)
                self.parameters[true_emission_features] += self.learning_rate

                hat_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_hat)
                #hat_features.extend(hat_emission_features)
                self.parameters[hat_emission_features] -= self.learning_rate

            ## update trigram features
            y_1_true = sequence.y[pos-1]
            y_2_true = sequence.y[pos-2]
            y_1_hat = y_hat[pos-1]
            y_2_hat = y_hat[pos-2]
            ## If true trigram != predicted trigram update trigram features
            if( (y_t_true,y_1_true,y_2_true) != (y_t_hat,y_1_hat,y_2_hat) ):
                true_transition_features = self.feature_mapper.get_transition_features(sequence, pos-2, y_t_true, y_1_true, y_2_true)
                #true_features.extend(true_transition_features)
                self.parameters[true_transition_features] += self.learning_rate

                hat_transition_features = self.feature_mapper.get_transition_features(sequence, pos-2, y_t_hat, y_1_hat,y_2_hat)
                #hat_features.extend(hat_transition_features)
                self.parameters[hat_transition_features] -= self.learning_rate

        # PARAMETERS UPDATE

        #self.parameters[true_features] += self.learning_rate
        #self.parameters[hat_features] -= self.learning_rate
        self.parameters -= self.learning_rate * reg * param_temp

        return num_labels, num_mistakes


    def perceptron_update_bigram(self, sequence):
        num_labels = 0
        num_mistakes = 0
        true_features = []
        hat_features = []
        reg = self.regularization_param

        predicted_sequence, _ = self.viterbi_decode_bigram(sequence)
        
        if PARALLELIZE:
            num_labels = len(sequence.x)-2
            num_mistakes = sum([1 for i in range(2,num_labels+2) if sequence.y[i]!=predicted_sequence.y[i]])
            
            pool = Pool(processes=NPROCESSORS)
            params = []
            for pos in range(2,len(sequence.x)):
                params.append([ sequence,
                                predicted_sequence,
                                pos,
                                self.learning_rate])

            update_list = pool.map(self.feature_mapper.get_features_parallel, params)
            for phi in update_list:
                self.parameters += phi
            
            pool.close()
            pool.join()
        else:
            phi_y = np.zeros(self.feature_mapper.get_num_features())
            phi_z = np.zeros(self.feature_mapper.get_num_features())

            y_hat = predicted_sequence.y
            for pos in range(2,len(sequence.x)):
                y_t_true = sequence.y[pos]
                y_t_hat = y_hat[pos]
                prev_y_t_true = sequence.y[pos-1]
                prev_y_t_hat = y_hat[pos-1]

                # Update emission features.
                num_labels += 1
                if y_t_true != y_t_hat:
                    num_mistakes += 1
                    true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true)
                    #self.parameters[true_emission_features] += self.learning_rate
                    phi_y[true_emission_features] += self.learning_rate

                    hat_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_hat)
                    #self.parameters[hat_emission_features] -= self.learning_rate
                    phi_z[hat_emission_features] += self.learning_rate

                if(y_t_true != y_t_hat or prev_y_t_true != prev_y_t_hat):
                    true_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_true, prev_y_t_true)
                    #self.parameters[true_transition_features] += self.learning_rate
                    phi_y[true_transition_features] += self.learning_rate

                    hat_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_hat, prev_y_t_hat)
                    #self.parameters[hat_transition_features] -= self.learning_rate
                    phi_z[hat_transition_features] += self.learning_rate
        
            # PARAMETERS UPDATE
            self.parameters = (1.0 - reg*self.learning_rate)*self.parameters + phi_y - phi_z

        return num_labels, num_mistakes



    def save_model(self,dir):
        fn = open(dir+"parameters.txt",'w')
        for p_id,p in enumerate(self.parameters):
            fn.write("%i\t%f\n"%(p_id,p))
        fn.close()

    def load_model(self,dir):
        fn = open(dir+"parameters.txt",'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
