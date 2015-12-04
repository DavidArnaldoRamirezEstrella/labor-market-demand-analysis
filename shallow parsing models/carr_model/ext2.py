import os, sys
import re, string
import pdb, ipdb
import unicodedata

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_utils)

from utils_new import uploadObject, assignFilterTag, filter_names, stemAugmented, NPROCESSORS
from carr_model.id_feature_bigram import *
from sequences.label_dictionary import *

from multiprocessing import Pool
from functools import partial
from datetime import datetime


WINDOW = 4


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
        
        ## Build trigger_word dictionary
        for sequence in self.dataset.seq_list:
            for pos in range(2,len(sequence.x)-1):      # NO START END
                self.update_tw(sequence,pos)
        
        # Filter unfrequent ones
        #ipdb.set_trace()

        self.inner_trigger_words = self.filter_unfrequent(self.inner_trigger_words, self.FILTER_WORDS)
        #self.outer_trigger_words = self.filter_unfrequent(self.outer_trigger_words,0.01, filter_common=False)
        self.include_gazzeter()

        #self.inner_trigger_pos = self.filter_unfrequent(self.inner_trigger_pos, self.FILTER_POS,filter_common=False) # CAMBIO!!
        #self.outer_trigger_pos = self.filter_unfrequent(self.outer_trigger_pos, self.FILTER_POS)
        #ipdb.set_trace()

        # Reading word-cluster lexicon
        mx_len = 0
        for line in open(self.wordcluster_source):
            if line:
                #word,bitstream,freq = line.split('\t')
                bitstream,word,freq = line.split('\t')
                self.word_clusters[word] = bitstream
                mx_len = max(mx_len, len(bitstream))

        # MULTIPROCESSING LIKE A BOSS
        pool = Pool(processes=NPROCESSORS)
        features = pool.map(unwrapp_seq,zip([self]*len(self.dataset.seq_list),self.dataset.seq_list))
        pool.close()
        pool.join()

        unique_features = set()
        for _set in features:
            unique_features |= _set

        for feat in unique_features:
            self.insert_feature(feat)
        
        """
        for sequence in self.dataset.seq_list:
            initial_features, transition_features, final_features, emission_features = \
                    self.get_sequence_features(sequence)
        """
        self.add_features = False
        
        print("Features:",self.get_num_features())
        print("Max len bitstream on brown:",mx_len)
        print("Window: ",WINDOW)
        print("---------------------------------------")
        print("Execution time: ",datetime.now()-startTime)
        print("=======================================")
        #ipdb.set_trace()


    ###################################################################################################################
    def add_emission_features(self, sequence, pos_current, y, features):
        ########    SOLO PALABRA ACTUAL : EMISSION ORIGINAL
        word ,low_word ,stem  = self.get_label_names(sequence,pos_current)
        word1,low_word1,stem1 = self.get_label_names(sequence,pos_current-1)
        
        y_name = sequence.sequence_list.y_dict.get_label_name(y)


        #############################################################################
        # CURR_WORD TRIGGER FEATURES
        if y_name[0]=='B' or y_name[0]=='I':
            ## TRIGGER BAG OF WORDS FEATURES
            features = self.get_trigger_features(low_word, y_name, prefix='innerTW', _dict=self.inner_trigger_words,
                                                 features=features)

        # INNER_LIST PRIOR
            features = self.get_inner_list_features(sequence,pos_current,y_name,include_stem=True,features=features)
        #features = self.get_inner_list_features(sequence,pos_current,y_name,include_stem=False,features=features)
        
        """
        # OUTTER TW PRIOR
        ### LO BAJA
        if y_name[0]!='B' and y_name[0]!='I':
            features = self.get_trigger_features(low_word, y_name, prefix='outterTW', _dict=self.outer_trigger_words,
                                              features=features)
        """
            # only inside NE
        features = self.get_context_features(sequence,pos_current,y, window=WINDOW, features=features)
        
        return features






    # CONTEXTUAL FEATURES
    def get_context_features(self, sequence, pos_current, y, window=WINDOW,features=[]):
        ## INFO  GLOBAL DE Y, Y_prev    
        length = len(sequence.y)
        y_name = sequence.sequence_list.y_dict.get_label_name(y)
        word_0,low_word_0,stem_0 = self.get_label_names(sequence,pos_current)

        """
        if y_name[0]=='B':
            window_range = range(max(0, pos_current-WINDOW),pos_current+1)
        elif y_name[0]=='O':
            window_range = range(pos_current,min(pos_current+WINDOW + 1, length))
        else:
            window_range = range(max(0, pos_current-WINDOW), min(pos_current+WINDOW + 1, length))
        """
        window_range = range(max(0, pos_current-WINDOW), min(pos_current+WINDOW+1, length))

        for pos in window_range:
            word,low_word,stem = self.get_label_names(sequence,pos)
            y_curr_name = sequence.sequence_list.y_dict.get_label_name(sequence.y[pos])

            #############################################################################
            # WINDOW WORD + POSITION + EMISSION ORIGINAL
            feat_name = "id::%i::%s::%s" % (pos-pos_current, low_word, y_name)
            features = self.insert_feature(feat_name, features)

            #############################################################################
            # WINDOW STEM + POSITION + EMISSION ORIGINAL
            feat_name = "stem::%i::%s::%s" % (pos-pos_current, stem, y_name)
            features = self.insert_feature(feat_name, features)

            # STEM CONTEXT UNIGRAM
            if (pos-pos_current) in [-1,0,1]:
                feat_name = "context_UNI_STEM::%s__%s" % (y_name, stem)
                features = self.insert_feature(feat_name, features)

            #############################################################################
            # OUTTER TRIGGER WORDS - BAG
            """
            # LO BAJA
            if y_name[0]=='B' or y_name[0]=='I':
                features = self.get_trigger_features(low_word, y_name, prefix='context_outterTW', _dict=self.outer_trigger_words,
                                                     features=features)
            """

            #############################################################################
            # WINDOW ORTOGRAPHIC FEATURES            
            # emission + ort + POSITION
            if (pos-pos_current) in [-1,0,1]:
                features = self.get_ortographic_features(word,y_name=y_name, pos=pos - pos_current,features=features)

            #############################################################################
            # SUFFIX AND PREFFIX + POSITION 
            if (pos-pos_current) in [-1,0,1]:
                features = self.get_suffpref_features(low_word,y_name,pos=pos-pos_current,features=features)

            #############################################################################
            # REDUCED WINDOW WORD-CLUSTER
            if (pos-pos_current) in [-1,0,1]:
                features = self.get_word_cluster_features(low_word,y_name, pos=pos-pos_current,features=features)
        
            #############################################################################
            # Y BIGRAM CONTEXT
            if pos < length-1:
                word1,low_word1,stem1 = self.get_label_names(sequence,pos+1) # OJO es POS+1

                feat_name = "context_BI_WORD::%s__%s::%s" % (y_name,low_word,low_word1)
                features = self.insert_feature(feat_name, features)
                
                feat_name = "context_BI_STEM::%s__%s::%s" % (y_name,stem,stem1)
                features = self.insert_feature(feat_name, features)

                ########################################
                # Y_BIGRAM STEM + POSITION
                """# f
                feat_name = "context_BI_STEM_POS::%s__%i::%s::%s" % (y_name,pos-pos_current,stem,stem1)
                features = self.insert_feature(feat_name, features)
                """
                
                if (pos-pos_current) in [-1,0,1]:
                    ########################################
                    # Y BIGRAM CONTEXT ORT : Y-> ORTi ORTi+1
                    ort  = self.get_ortography(word)
                    ort1 = self.get_ortography(word1)
                    rare_ort = True
                    for u in ort:
                        for v in ort1:
                            feat_name = "context_BI_ORT::%s__%s:%s" % (y_name,u,v)
                            features = self.insert_feature(feat_name, features)
                    
                ########################################

            #############################################################################
            # Y TRIGRAM CONTEXT
            """
            if pos < length-2:
                word1,low_word1,stem1 = self.get_label_names(sequence,pos+1) # OJO es POS+1
                word2,low_word2,stem2 = self.get_label_names(sequence,pos+2) # OJO es POS+2

                ### NOT ENOUGH DATA
                #feat_name = "context_BI_WORD::%s__%s::%s" % (y_name,low_word,low_word1)
                #features = self.insert_feature(feat_name, features)
                
                #feat_name = "context_TRI_STEM::%s__%s::%s::%s" % (y_name,stem,stem1,stem2)
                #features = self.insert_feature(feat_name, features)

                if (pos-pos_current) == -1:
                    ########################################
                    # Y BIGRAM CONTEXT ORT : Y-> ORTi ORTi+1 ORTi+2    || BAJA TODO
                    
                    ort  = self.get_ortography(word)
                    ort1 = self.get_ortography(word1)
                    ort2 = self.get_ortography(word2)
                    rare_ort = True
                    for u in ort:
                        for v in ort1:
                            for w in ort2:
                                feat_name = "context_TRI_ORT::%s__%s:%s:%s" % (y_name,u,v,w)
                                features = self.insert_feature(feat_name, features)
                    
                    feat_name = "context_TRI_STEM::%s__%s::%s::%s" % (y_name,stem,stem1,stem2)  # BAJA I BASTANTE
                    features = self.insert_feature(feat_name, features)
            """

        return features


    




    def add_transition_features(self, sequence, pos, y, y_prev, features):
        """                            pos -> y_prev
        Adds a feature to the edge feature list.
        Creates a unique id if its the first time the feature is visited
        or returns the existing id otherwise
        """
        assert pos < len(sequence.x), pdb.set_trace()

        # Get label name from ID.
        y_name = sequence.sequence_list.y_dict.get_label_name(y)
        # Get previous label name from ID.
        y_prev_name = sequence.sequence_list.y_dict.get_label_name(y_prev)
        # Generate feature name.
        feat_name = "prev_tag::%s::%s"%(y_prev_name,y_name)

        if self.add_features and any([y_name == '**' and y_prev_name == '**',
                                      y_name == 'B' and y_prev_name == '_STOP_',
            ]):
            print("FEATURE ERROR")
            ipdb.set_trace()

        features = self.insert_feature(feat_name, features)
        
        ###########################################################################################
        # TRIGRAM TRANSITION
        """
        y_prev_2_name = sequence.sequence_list.y_dict.get_label_name(sequence.y[pos-1])
        feat_name = "prev_prev_tag::%s::%s::%s"%(y_prev_2_name,y_prev_name,y_name)
        features = self.insert_feature(feat_name, features)
        """

        ###########################################################################################
        # TRANSITION_2 + WORD_2, + POS_2
        word ,low_word ,stem  = self.get_label_names(sequence,pos+1)
        word1,low_word1,stem1 = self.get_label_names(sequence,pos)
        word2,low_word2,stem2 = self.get_label_names(sequence,pos-1)

        feat_name = "TRANS_WORD::%s::%s_%s::%s" % (y_name,y_prev_name,low_word,low_word1)
        features = self.insert_feature(feat_name, features)


        feat_name = "TRANS_STEM::%s::%s_%s::%s" % (y_name,y_prev_name,stem,stem1)
        features = self.insert_feature(feat_name, features)


        #################################################################
        # RESTRINGIENDO CONTEXT TRAINING TO B & I & FIRST O
        if any([y_name[0]=='B',
                y_name[0]=='I',
                y_name[0]=='O' and y_prev_name[0]=='I',
                y_name[0]=='O' and y_prev_name[0]=='B' ]):

            ###############################
            ## Y,Y -> ORT,ORT
            rare_ort = True
            ort  = self.get_ortography(word)
            ort1 = self.get_ortography(word1)
            
            for u in ort:
                for v in ort1:
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
