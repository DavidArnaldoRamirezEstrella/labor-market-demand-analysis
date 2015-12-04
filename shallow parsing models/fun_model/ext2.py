import os, sys
import re, string
import pdb, ipdb
import unicodedata

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_utils)

from utils_new import uploadObject, assignFilterTag, filter_names, stemAugmented, NPROCESSORS
from fun_model.id_feature_bigram import *
from sequences.label_dictionary import *

from multiprocessing import Pool
from datetime import datetime


WINDOW = 3


#######################
#### Feature Class
### Extracts features from a labeled corpus (only supported features are extracted
#######################

###############################################################################################

def unwrapp_seq(arg, **kwargs):
    return ExtendedFeatures.get_sequence_features(*arg,**kwargs)

##############################################################################################

class ExtendedFeatures(IDFeatures):

    ###################################################################################################################
    def build_features(self):
        startTime = datetime.now()
        self.add_features = True

        # Reading word-cluster lexicon
        mx_len = 0
        for line in open(self.wordcluster_source):
            if line:
                #word,bitstream,freq = line.split('\t')
                bitstream,word,freq = line.split('\t')
                self.word_clusters[word] = bitstream
                mx_len = max(mx_len, len(bitstream))

        # MULTIPROCESSING LIKE A BOSS
        """        
        pool = Pool(processes=NPROCESSORS)
        features = pool.map(unwrapp_seq,zip([self]*len(self.dataset.seq_list),self.dataset.seq_list))
        pool.close()
        pool.join()

        unique_features = set()
        for _set in features:
            unique_features |= _set
        """
        
        count = 0
        unique_features = set()
        for seq in self.dataset.seq_list:
            unique_features |= self.get_sequence_features(seq)
            if count%1000==0:
                print('->',count)
            count += 1
        

        for feat in unique_features:
            self.insert_feature(feat)
        
        self.add_features = False
        
        print("Features:",self.get_num_features())
        print("Max len bitstream on brown:",mx_len)
        print("Window: ",WINDOW)
        print("---------------------------------------")
        print("Execution time: ",datetime.now()-startTime)
        print("=======================================")
        #ipdb.set_trace()

    ###################################################################################################################
    def get_features(self,sequence, pos_current, y_1, y_2,features=[]):
        length = len(sequence.x)
        y_prev_name = sequence.sequence_list.y_dict.get_label_name(y_1)

        """
        # INNER_LIST PRIOR
        if y_prev_name[0]=='B' or y_prev_name[0]=='I':
            features = self.get_inner_list_features(sequence,pos_current,include_stem=True,features=features)
        """

        features = self.get_context_features(sequence,pos_current,features)
        features = self.get_transition_features(sequence, pos_current, y_1, y_2, features)

        return features


    # CONTEXTUAL FEATURES
    def get_context_features(self, sequence, pos_current,features=[]):
        length = len(sequence.x)
        window_range = range(max(0, pos_current-WINDOW), min(pos_current+WINDOW+1, length))

        for pos in window_range:
            word,low_word,stem = self.get_label_names(sequence,pos)

            #############################################################################
            # WINDOW WORD + POSITION + EMISSION ORIGINAL
            feat_name = "id::%i::%s" % (pos-pos_current, low_word)
            features = self.insert_feature(feat_name, features)

            #############################################################################
            # WINDOW STEM + POSITION + EMISSION ORIGINAL
            feat_name = "stem::%i::%s" % (pos-pos_current, stem)
            features = self.insert_feature(feat_name, features)

            """
            # STEM CONTEXT UNIGRAM
            if (pos-pos_current) in [-1,0,1]:
                feat_name = "context_UNI_STEM::%s" % (stem)
                features = self.insert_feature(feat_name, features)
            """
            #############################################################################
            # WINDOW ORTOGRAPHIC FEATURES
            # emission + ort + POSITION
            if (pos-pos_current) in [-1,0,1]:
                features = self.get_ortographic_features(word, pos=pos-pos_current,features=features)

            #############################################################################
            # SUFFIX AND PREFFIX + POSITION 
            if (pos-pos_current) in [-1,0,1]:
                features = self.get_suffpref_features(low_word,pos=pos-pos_current,features=features)

            #############################################################################
            # REDUCED WINDOW WORD-CLUSTER
            """
            if (pos-pos_current) in [-1,0,1]:
                features = self.get_word_cluster_features(low_word, pos=pos-pos_current,features=features)
            """
        
            #############################################################################
            # Y BIGRAM CONTEXT
            """
            if pos < length-1:
                word1,low_word1,stem1 = self.get_label_names(sequence,pos+1) # OJO es POS+1

                feat_name = "context_BI_WORD::%s__%s" % (low_word,low_word1)
                features = self.insert_feature(feat_name, features)
                
                feat_name = "context_BI_STEM::%s__%s" % (stem,stem1)
                features = self.insert_feature(feat_name, features)

                ########################################
                # Y_BIGRAM STEM + POSITION
                
                if (pos-pos_current) in [-1,0,1]:
                    ########################################
                    # Y BIGRAM CONTEXT ORT : Y-> ORTi ORTi+1
                    ort  = self.get_ortography(word)
                    ort1 = self.get_ortography(word1)
                    rare_ort = True
                    for u in ort:
                        for v in ort1:
                            feat_name = "context_BI_ORT::%s__%s" % (u,v)
                            features = self.insert_feature(feat_name, features)
                    
                ########################################
            """

        return features






    def get_transition_features(self, sequence, pos, y_1, y_2, features):
        """                                      y_2 y_1 [pos]
        Adds a feature to the edge feature list.
        Creates a unique id if its the first time the feature is visited
        or returns the existing id otherwise
        """
        assert pos < len(sequence.x), pdb.set_trace()

        # Get label name from ID.
        y_name = sequence.sequence_list.y_dict.get_label_name(y_1)
        y_prev_name = sequence.sequence_list.y_dict.get_label_name(y_2)

        if self.add_features and any([y_name == '**' and y_prev_name == '**',
                                      y_name == 'B' and y_prev_name == '_STOP_',
            ]):
            print("FEATURE ERROR")
            ipdb.set_trace()

        
        feat_name = "trans_bi::%s::%s"%(y_prev_name,y_name)
        features = self.insert_feature(feat_name, features)
        

        ###########################################################################################
        # TRANSITION_2 + WORD_2, + POS_2
        word ,low_word ,stem  = self.get_label_names(sequence,pos)
        word1,low_word1,stem1 = self.get_label_names(sequence,pos-1)
        word2,low_word2,stem2 = self.get_label_names(sequence,pos-2)

        feat_name = "TRANS_WORD::%s::%s_%s::%s" % (y_name,y_prev_name,low_word1,low_word2)
        features = self.insert_feature(feat_name, features)


        feat_name = "TRANS_STEM::%s::%s_%s::%s" % (y_name,y_prev_name,stem1,stem2)
        features = self.insert_feature(feat_name, features)


        #################################################################
        # RESTRINGIENDO CONTEXT TRAINING TO B & I & FIRST O
        #if any([y_name[0]=='B',
                #y_name[0]=='I',
        #        ]):

        ###############################
        ## Y,Y -> ORT,ORT
        rare_ort = True
        ort  = self.get_ortography(word)
        ort1 = self.get_ortography(word1)
        ort2 = self.get_ortography(word2)
        
        for u in ort1:
            for v in ort2:
                feat_name = "TRANS_ORT::%s:%s__%s:%s" % (y_name,y_prev_name,u,v)
                features = self.insert_feature(feat_name, features)

        ###############################
        ## Y,Y -> CLUST,CLUST
        """ NO CAMBIA EN NADA !! WTF
        g1 = self.get_word_clusters(word)
        g2 = self.get_word_clusters(word1)
        
        for u in g1:
            for v in g2:
                feat_name = "TRANS_CLUSTER::%s:%s__%s:%s" % (y_name,y_prev_name,u,v)
                features = self.insert_feature(feat_name, features)
        """
        #############################################################################
        
        return features
