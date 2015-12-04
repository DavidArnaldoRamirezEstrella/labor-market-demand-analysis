import sys,os
import numpy as np
import sequences.sequence_classifier as sc
import ipdb
from multiprocessing import Pool

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_utils)

from utils_new import NPROCESSORS,PARALLELIZE

class DiscriminativeSequenceClassifier(sc.SequenceClassifier):

    def __init__(self, observation_labels, state_labels, feature_mapper):
        sc.SequenceClassifier.__init__(self, observation_labels, state_labels)

        # Set feature mapper and initialize parameters.
        self.feature_mapper = feature_mapper
        self.parameters = np.zeros(self.feature_mapper.get_num_features())

    ################################
    ##  Build the node and edge potentials
    ## node - f(t,y_t,X)*w
    ## edge - f(t,y_t,y_(t-1),X)*w
    ## Only supports binary features representation
    ## If we have an HMM with 4 positions and transitions
    ## a - b - c - d
    ## the edge potentials have at position:
    ## 0 a - b
    ## 1 b - c
    ################################
    def compute_scores(self, sequence):
        num_states = self.get_num_states()
        length = len(sequence.x)
        emission_scores = np.zeros([length, num_states])
        initial_scores = np.zeros([num_states,num_states])
        transition_scores = np.zeros([length-2, num_states, num_states, num_states])
        final_scores = np.zeros([num_states,num_states])

        # Intermediate position.
        for pos in range(2,length):
            for tag_id in range(num_states):
                 emission_features = self.feature_mapper.get_emission_features(sequence, pos, tag_id)
                 score = 0.0
                 for feat_id in emission_features:
                     score += self.parameters[feat_id]
                 emission_scores[pos, tag_id] = score

            for y in range(num_states):
                for y_1 in range(num_states):
                    for y_2 in range(num_states):
                        transition_features = self.feature_mapper.get_transition_features(sequence, pos-2, y,y_1,y_2)
                        score = 0.0
                        for feat_id in transition_features:
                            score += self.parameters[feat_id]
                        transition_scores[pos-2, y, y_1, y_2] = score

        return initial_scores, transition_scores, final_scores, emission_scores


    def compute_scores_bigram(self, sequence):
        num_states = self.get_num_states()
        length = len(sequence.x)
        emission_scores = np.zeros([length, num_states])
        initial_scores = np.zeros(num_states)
        transition_scores = np.zeros([length-1, num_states, num_states])
        final_scores = np.zeros(num_states)

        if PARALLELIZE:
            ##############################################
            ## PARALLELIZATION
            pool = Pool(processes=NPROCESSORS)
            # Intermediate position.
            parallel_params = []
            for pos in range(2,length):
                parallel_params.append([ sequence,
                                         pos,
                                         self.parameters,
                                         num_states,
                                        ])
            scores = pool.map(self.feature_mapper.get_scores_parallel,parallel_params)
            pool.close()
            pool.join()

            for pos in range(2,length):
                emiss = scores[pos-2][0]
                trans = scores[pos-2][1]
                emission_scores[pos] = emiss
                transition_scores[pos-1] = trans
        else:
            for pos in range(2,length):
                for tag_id in range(num_states):
                    #############################################
                    emission_features = self.feature_mapper.get_emission_features(sequence, pos, tag_id)
                    score = 0.0
                    for feat_id in emission_features:
                        score += self.parameters[feat_id]
                    emission_scores[pos, tag_id] = score

                    for prev_tag_id in range(num_states):
                        #############################################
                        transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, tag_id, prev_tag_id)
                        score = 0.0
                        for feat_id in transition_features:
                            score += self.parameters[feat_id]
                        transition_scores[pos-1, tag_id, prev_tag_id] = score
        
        return initial_scores, transition_scores, final_scores, emission_scores


    def compute_scores_backup(self, sequence):
        num_states = self.get_num_states()
        length = len(sequence.x)
        emission_scores = np.zeros([length, num_states])
        initial_scores = np.zeros(num_states)
        transition_scores = np.zeros([length-1, num_states, num_states])
        final_scores = np.zeros(num_states)

        # Initial position.
        for tag_id in range(num_states):
             initial_features = self.feature_mapper.get_initial_features(sequence, tag_id)
             score = 0.0
             for feat_id in initial_features:
                 score += self.parameters[feat_id]
             initial_scores[tag_id] = score

        # Intermediate position.
        for pos in range(2,length-1):
            for tag_id in range(num_states):
                 emission_features = self.feature_mapper.get_emission_features(sequence, pos, tag_id)
                 score = 0.0
                 for feat_id in emission_features:
                     score += self.parameters[feat_id]
                 emission_scores[pos, tag_id] = score
            if pos > 2: 
                for tag_id in range(num_states):
                    for prev_tag_id in range(num_states):
                        transition_features = self.feature_mapper.get_transition_features(sequence, pos, tag_id, prev_tag_id)
                        score = 0.0
                        for feat_id in transition_features:
                            score += self.parameters[feat_id]
                        transition_scores[pos-1, tag_id, prev_tag_id] = score

        # Final position.
        for prev_tag_id in range(num_states):
             final_features = self.feature_mapper.get_final_features(sequence, prev_tag_id)
             score = 0.0
             for feat_id in final_features:
                 score += self.parameters[feat_id]
             final_scores[prev_tag_id] = score

        return initial_scores, transition_scores, final_scores, emission_scores

