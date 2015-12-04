from sequences.label_dictionary import *
import ipdb,pdb
from nltk.stem.snowball import SpanishStemmer
stemmer = SpanishStemmer()

#################
### Replicates the same features as the HMM
### One for word/tag and tag/tag pair
#################
BR = '**'
RARE = "<RARE>"

class IDFeatures:
    '''
        Base class to extract features from a particular dataset.

        feature_dic --> Dictionary of all existing features maps feature_name (string) --> feature_id (int) 
        feture_names --> List of feature names. Each position is the feature_id and contains the feature name
        nr_feats --> Total number of features
        feature_list --> For each sentence in the corpus contains a pair of node feature and edge features
        dataset --> The original dataset for which the features were extracted

        Caches (for speedup):
        initial_state_feature_cache -->
        node_feature_cache -->
        edge_feature_cache -->
        final_state_feature_cache -->
    '''

    def __init__(self, dataset):
        '''dataset is a sequence list.'''
        self.feature_dict = LabelDictionary()
        self.feature_list = []

        self.add_features = False
        self.dataset = dataset

        #Speed up
        self.node_feature_cache = {}
        self.initial_state_feature_cache = {}
        self.final_state_feature_cache = {}
        self.edge_feature_cache = {}


    def get_num_features(self):
        return len(self.feature_dict)

    def build_features(self):
        '''
        Generic function to build features for a given dataset.
        Iterates through all sentences in the dataset and extracts its features,
        saving the node/edge features in feature list.
        '''

        self.add_features = True
        for sequence in self.dataset.seq_list:
           initial_features, transition_features, final_features, emission_features = \
               self.get_sequence_features(sequence)
           self.feature_list.append([initial_features, transition_features, final_features, emission_features])
        self.add_features = False
        #ipdb.set_trace()

    def get_sequence_features(self, sequence):
        '''
        Returns the features for a given sequence.
        For a sequence of size N returns:
        Node_feature a list of size N. Each entry contains the node potentials for that position.
        Edge_features a list of size N+1.
        - Entry 0 contains the initial features
        - Entry N contains the final features
        - Entry i contains entries mapping the transition from i-1 to i.
        '''
        emission_features = []
        initial_features = []
        transition_features = []
        final_features = []

        ## Take care of middle positions
        for pos in range(2,len(sequence.y)):
            features = []
            features = self.add_emission_features(sequence, pos, sequence.y[pos], features)
            emission_features.append(features)

            tag = sequence.y[pos]
            y_1 = sequence.y[pos-1]
            y_2 = sequence.y[pos-2]

            features = []
            features = self.add_transition_features(sequence, pos-2, tag, y_1, y_2, features)
            transition_features.append(features)

        ## Take care of final position | Yn-1, Yn ,STOP
        #features = []
        #features = self.add_final_features(sequence, sequence.y[-1],sequence.y[-2], features)
        #final_features.append(features)

        return initial_features, transition_features, final_features, emission_features

    #f(t,y_t,X)
    # Add the word identity and if position is
    # the first also adds the tag position
    def get_emission_features(self, sequence, pos, y):
        all_feat = []
        x = sequence.x[pos]
        x_name = sequence.sequence_list.x_dict.get_label_name(x)
        y_name = sequence.sequence_list.y_dict.get_label_name(y)
        stem = stemmer.stem(x_name.lower())
        if stem not in self.dataset.stem_vocabulary:
            x_name = RARE
        if(x_name not in self.node_feature_cache):
            self.node_feature_cache[x_name] = {}
        if(y_name not in self.node_feature_cache[x_name]):
            node_idx = []
            node_idx = self.add_emission_features(sequence, pos, y, node_idx)
            self.node_feature_cache[x_name][y_name] = node_idx
        idx = self.node_feature_cache[x_name][y_name]
        all_feat = idx[:]
        return all_feat

    #f(t,y_t,y_(t-1),X)
    ##Speed up of code
    def get_transition_features(self, sequence, pos, y, y_1, y_2):
        assert(pos >= 0 and pos < len(sequence.x)), pdb.set_trace()
        y_name = sequence.sequence_list.y_dict.get_label_name(y)
        y_name_1 = sequence.sequence_list.y_dict.get_label_name(y_1)
        y_name_2 = sequence.sequence_list.y_dict.get_label_name(y_2)
        if( (y_name,y_name_1,y_name_2) not in self.edge_feature_cache):
            edge_idx = []
            edge_idx = self.add_transition_features(sequence, pos, y, y_1,y_2, edge_idx)
            self.edge_feature_cache[(y_name,y_name_1,y_name_2)] = edge_idx
        return self.edge_feature_cache[(y_name,y_name_1,y_name_2)]


    def get_initial_features(self, sequence, y, y_1):
        # NO SE USA ACTUALMENTE
        if (y,y_1) not in self.initial_state_feature_cache:
           edge_idx = []
           edge_idx =  self.add_initial_features(sequence, y, y_1, edge_idx)
           self.initial_state_feature_cache[(y,y_1)] = edge_idx
        if len(self.initial_state_feature_cache[(y,y_1)]) > 500:
            ipdb.set_trace()

        return self.initial_state_feature_cache[(y,y_1)]


    def get_final_features(self, sequence, y, y_1):
        # NO SE USA ACTUALMENTE
        if((y,y_1) not in self.final_state_feature_cache):
            edge_idx = []
            edge_idx = self.add_final_features(sequence, y, y_1, edge_idx)
            self.final_state_feature_cache[(y,y_1)] = edge_idx
        return self.final_state_feature_cache[(y,y_1)]

    def get_context_features(self, sequence, pos_current, features):
        return features

    # NO SE USA
    def add_initial_features(self, sequence, y, y_1, features):
        # Get label name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        y_1_name = self.dataset.y_dict.get_label_name(y_1)
        # Generate feature name.
        feat_name = "init_tag:%s::%s" % (y_1_name,y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if(feat_id != -1):
            features.append(feat_id)
        return features

    # NO SE USA
    def add_final_features(self, sequence, y,y_1, features):
        # Get label name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        y_1_name = self.dataset.y_dict.get_label_name(y_1)
        # Generate feature name.
        feat_name = "final_tag:%s::%s" % (y_1_name,y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if(feat_id != -1):
            features.append(feat_id)
        return features

    def add_emission_features(self, sequence, pos, y, features):
        '''Add word-tag pair feature.'''
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get word name from ID.
        x_name = sequence.sequence_list.x_dict.get_label_name(x)
        stem = stemmer.stem(x_name)
        if stem not in self.dataset.stem_vocabulary:
            x_name = RARE
        # Generate feature name.
        feat_name = "id:%s::%s"%(x_name,y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)
        return features

    def add_transition_features(self, sequence, pos, y, y_1, y_2, features):
        """ Adds a feature to the edge feature list.
        Creates a unique id if its the first time the feature is visited
        or returns the existing id otherwise
        """
        assert pos < len(sequence.x)-1, pdb.set_trace()

        # Get label name from ID.
        y_name = sequence.sequence_list.y_dict.get_label_name(y)
        # Get previous label name from ID.
        y_1_name = sequence.sequence_list.y_dict.get_label_name(y_1)
        y_2_name = sequence.sequence_list.y_dict.get_label_name(y_2)
        # Generate feature name.

        if self.add_features and any([y_name == 'I' and y_1_name != 'I' and y_1_name != 'B',
                                      y_1_name == 'I' and y_2_name != 'I' and y_2_name != 'B'
            ]):
            print("FEATURE ERROR")
            ipdb.set_trace()

        feat_name = "prevs_tag:%s::%s::%s"%(y_2_name,y_1_name,y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if(feat_id != -1):
            features.append(feat_id)

        features = self.get_context_features(sequence,pos,features)

        return features

    def add_feature(self, feat_name):
        """
        Builds a dictionary of feature name to feature id
        If we are at test time and we don't have the feature
        we return -1.
        """
        # Check if feature exists and if so, return the feature ID. 
        if(feat_name in self.feature_dict):
            return self.feature_dict[feat_name]
        # If 'add_features' is True, add the feature to the feature 
        # dictionary and return the feature ID. Otherwise return -1.
        if not self.add_features:
            return -1
        return self.feature_dict.add(feat_name)
