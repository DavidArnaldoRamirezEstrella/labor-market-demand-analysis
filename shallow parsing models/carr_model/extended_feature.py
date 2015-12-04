
import os, sys
import re, string
import pdb, ipdb
import unicodedata

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_utils)

from utils_new import uploadObject, assignFilterTag, filter_names, stemAugmented
from carr_model.id_feature_bigram import *
from sequences.label_dictionary import *


##########################################################################################################
MX_LEN_BITSTREAM = 51       # MAX_LEN en Liand 5000 es 32
cluster_prefixes = range(8,MX_LEN_BITSTREAM,4)

WINDOW = 2

#######################
#### Feature Class
### Extracts features from a labeled corpus (only supported features are extracted
#######################
class ExtendedFeatures(IDFeatures):

    ###################################################################################################################
    def build_features(self):
        self.add_features = True
        
        ## Build trigger_word dictionary
        for sequence in self.dataset.seq_list:
            for pos in range(2,len(sequence.x)-1):      # NO START END
                self.update_tw(sequence,pos)
        
        # Filter unfrequent ones
        #ipdb.set_trace()
        self.outer_trigger_words = self.filter_unfrequent(self.outer_trigger_words,self.FILTER_WORDS)
        self.inner_trigger_words = self.filter_unfrequent(self.inner_trigger_words, self.FILTER_WORDS)
        self.include_gazzeter()

        self.inner_trigger_pos = self.filter_unfrequent(self.inner_trigger_pos, self.FILTER_POS,filter_common=False) # CAMBIO!!
        self.outer_trigger_pos = self.filter_unfrequent(self.outer_trigger_pos, self.FILTER_POS)
        #ipdb.set_trace()

        # Reading word-cluster lexicon
        mx_len = 0
        for line in open(self.wordcluster_source):
            if line:
                #word,bitstream,freq = line.split('\t')
                bitstream,word,freq = line.split('\t')
                self.word_clusters[word] = bitstream
                mx_len = max(mx_len, len(bitstream))
        cont = 0
        for sequence in self.dataset.seq_list:
            initial_features, transition_features, final_features, emission_features = \
                    self.get_sequence_features(sequence)
            #if cont % 100 == 0:
            #    print('sample->',cont)
            #cont += 1
        self.add_features = False
        
        print("Features:",self.get_num_features())
        print("Max len bitstream on brown:",mx_len)
        print("WINDOW:",WINDOW)
        print("=======================================")
        #ipdb.set_trace()





    ###################################################################################################################
    def add_emission_features(self, sequence, pos_current, y, y_prev, features):
        ########    SOLO PALABRA ACTUAL : EMISSION ORIGINAL
        word ,low_word ,pos_tag  ,stem  = self.get_label_names(sequence,pos_current)
        word1,low_word1,pos_1_tag,stem1 = self.get_label_names(sequence,pos_current-1)
        
        y_name = sequence.sequence_list.y_dict.get_label_name(y)
        y_1_name = sequence.sequence_list.y_dict.get_label_name(y_prev)  # SEQUENCE AHORA ES PREDICTED_SEQUENCE


        #############################################################################
        # CURR_WORD TRIGGER FEATURES
        if y_name[0]=='B' or y_name[0]=='I':
            ## TRIGGER BAG OF WORDS FEATURES
            features = self.get_trigger_features(low_word, y_name, prefix='innerTW', _dict=self.inner_trigger_words,
                                                 features=features)

            ## TRIGGER BAG OF POS FEATURES
            features = self.get_trigger_features(pos_tag, y_name, prefix='innerTP', pos_tag=True, _dict=self.inner_trigger_pos,
                                                 features=features)

        # RESTRINGIENDO CONTEXT TRAINING TO B & I
        if any([y_name[0]=='B',
                y_name[0]=='I',
                y_name[0]=='O' and y_1_name[0]=='I',
                y_name[0]=='O' and y_1_name[0]=='B' ]):
            features = self.get_context_features(sequence,pos_current,y,features)
        else:
            #############################################################################
            #### COMPENSATE MISSING EMISSION FEATURES

            # WINDOW WORD + POSITION + EMISSION ORIGINAL
            feat_name = "id::%i::%s::%s" % (0, low_word, y_name)
            features = self.insert_feature(feat_name, features)

            #############################################################################
            # WINDOW POS_tag + POSITION + EMISSION ORIGINAL
            feat_name = "pos::%i::%s::%s" % (0, pos_tag, y_name)
            features = self.insert_feature(feat_name, features)
            
            #############################################################################
            # WINDOW STEM + POSITION + EMISSION ORIGINAL
            feat_name = "stem::%i::%s::%s" % (0, stem, y_name)
            features = self.insert_feature(feat_name, features)

            #############################################################################
            # WINDOW ORTOGRAPHIC FEATURES
            # emission + ort + POSITION
            features = self.get_ortographic_features(word,y_name=y_name, pos=0,features=features)

            #############################################################################
            # SUFFIX AND PREFFIX + POSITION + FEATURES
            
            #features = self.get_suffpref_features(low_word,y_name,pos=0,features=features)
            
            #############################################################################
            # REDUCED WINDOW WORD-CLUSTER FEATURES
            #features = self.get_word_cluster_features(low_word,y_name, pos=0,features=features)

        return features








    def get_context_features(self, sequence, pos_current, y, features):
        # CONTEXTUAL FEATURES
        length = len(sequence.y)
        y_name = sequence.sequence_list.y_dict.get_label_name(y)
        y_prev_name = sequence.sequence_list.y_dict.get_label_name(sequence.y[pos_current-1])  # SEQUENCE AHORA ES PREDICTED_SEQUENCE
        
        word_0,low_word_0,pos_tag_0,stem_0 = self.get_label_names(sequence,pos_current)

        for pos in range(max(0, pos_current-WINDOW), min(pos_current+(WINDOW-1) + 1, length)):
            word,low_word,pos_tag,stem = self.get_label_names(sequence,pos)
            y_curr_name = sequence.sequence_list.y_dict.get_label_name(sequence.y[pos])

            #############################################################################
            # WINDOW WORD + POSITION + EMISSION ORIGINAL
            feat_name = "id::%i::%s::%s" % (pos-pos_current, low_word, y_name)
            features = self.insert_feature(feat_name, features)

            #############################################################################
            # WINDOW POS_tag + POSITION + EMISSION ORIGINAL
            if (pos-pos_current) in [-1,0,1]:
                feat_name = "pos::%i::%s::%s" % (pos-pos_current, pos_tag, y_name)
                features = self.insert_feature(feat_name, features)
            
            #############################################################################
            # WINDOW STEM + POSITION + EMISSION ORIGINAL
            feat_name = "stem::%i::%s::%s" % (pos-pos_current, stem, y_name)
            features = self.insert_feature(feat_name, features)

            #############################################################################
            # WINDOW ORTOGRAPHIC FEATURES
            #features = self.get_ortographic_features(word,pos=pos - pos_current,features=features)
            
            # emission + ort + POSITION
            features = self.get_ortographic_features(word,y_name=y_name, pos=pos - pos_current,features=features)


            #############################################################################
            # SUFFIX AND PREFFIX + POSITION + FEATURES
            features = self.get_suffpref_features(low_word,y_name,pos=pos-pos_current,features=features)
            
            #############################################################################
            # REDUCED WINDOW WORD-CLUSTER FEATURES
            #if (pos-pos_current) in [-1,0,1]:
            features = self.get_word_cluster_features(low_word,y_name, pos=pos-pos_current,features=features)

            # BAG OF WORD-CLUSTERS
            #features = self.get_word_cluster_features(low_word,y_name,features=features)


            #############################################################################
            #############################################################################
            ### EMISSION
            ## BAG OF WORDS AS EXTERN CONTEXT
            feat_name = "context_id::%s::%s" % (low_word,y_name)
            features = self.insert_feature(feat_name, features)

            
            # BAG OF POS_tag AS EXTERN CONTEXT
            feat_name = "context::pos:%s::%s" % (pos_tag,y_name)
            features = self.insert_feature(feat_name, features)
            

            # BAG OF STEM AS EXTERN CONTEXT
            feat_name = "context_stem::%s::%s" % (stem,y_name)
            features = self.insert_feature(feat_name, features)
            
            # CONTEXT SUFFIX AND PREFFIX BAG FEATURES
            features = self.get_suffpref_features(low_word,y_name,features=features)
            
            
            #############################################################################
            ## BIGRAMAS EN CONTEXTO DE Y_NAME
            ## Y_t + Xpos + Xpos+1
            #if all([pos < length-1,
            #        any([   y_name[0]=='B' and pos<=pos_current,
            #                y_name[0]=='O' and pos>=pos_current-1,
            #                y_name[0]=='I',
            #        ]),
            #    ]):
            if pos < length-1:
                word1,low_word1,pos_1_tag,stem1 = self.get_label_names(sequence,pos+1) # OJO es POS+1

                feat_name = "context_BI_WORD::%s__%s::%s" % (y_name,low_word,low_word1)
                features = self.insert_feature(feat_name, features)

                #if (pos-pos_current) in [-1,0,1]:
                feat_name = "context_BI_POS::%s__%s::%s" % (y_name,pos_tag,pos_1_tag)
                features = self.insert_feature(feat_name, features)
                
                feat_name = "context_BI_STEM::%s__%s::%s" % (y_name,stem,stem1)
                features = self.insert_feature(feat_name, features)

                ########################################
                # Y BIGRAM CONTEXT ORT : Y-> ORTi ORTi+1
                if (pos-pos_current) in [-1,0,1]:
                    #if y_name[0] == 'B' and word==START:
                    #    ipdb.set_trace()
                    rare_ort = True
                    ort = []
                    ort1 = []
                    if word in filter_names:
                        ort = [word]
                    else:
                        for i, pat in enumerate(ORT):
                            if pat.search(word):
                                rare_ort = False
                                ort.append(DEBUG_ORT[i])
                        if rare_ort:
                            ort = ["OTHER_ORT"]
                    
                    rare_ort = True
                    if word1 in filter_names:
                        ort1 = [word1]
                    else:
                        for i, pat in enumerate(ORT):
                            if pat.search(word1):
                                rare_ort = False
                                ort1.append(DEBUG_ORT[i])
                        if rare_ort:
                            ort1 = ["OTHER_ORT"]
                    for u in ort:
                        for v in ort1:
                            feat_name = "context_BI_ORT::%s__%s:%s" % (y_name,u,v)
                            features = self.insert_feature(feat_name, features)

                ########################################
                # Y BIGRAM CONTEXT ORT : Y-> ORTi ORTi+1
            
            #############################################################################
                """
                ## TRIGRAMAS EN CONTEXTO DE Y_NAME
                ## Y_t + Xpos-1 + Xpos + Xpos+1
                word1,low_word1,pos_1_tag,stem1 = self.get_label_names(sequence,pos+1) # OJO es POS+1

                feat_name = "context_BI_WORD::%s__%s::%s" % (y_name,low_word,low_word1)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

                #if (pos-pos_current) in [-1,0,1]:
                feat_name = "context_BI_POS::%s__%s::%s" % (y_name,pos_tag,pos_1_tag)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                
                feat_name = "context_BI_STEM::%s__%s::%s" % (y_name,stem,stem1)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)
                """

            #############################################################################
            ## CONTEXT TRIGGER FEATURES
            # OUTTER TRIGGER
            if any([y_name[0]=='B' and pos < pos_current,
                    y_name[0]=='O' and y_prev_name[0]=='I' and pos >= pos_current,
                    y_name[0]=='O' and y_prev_name[0]=='B' and pos >= pos_current ]):
                ## TRIGGER BAG OF WORDS FEATURES
                features = self.get_trigger_features(low_word, y_name, prefix='outerTW', _dict=self.outer_trigger_words,
                                                     features=features)

                
            #############################################################################
        #############################################################################
        # INNER TRIGGER  <- 0 ->  COHESION
        COHESION_WIND = 2       # solo 4 feature con 3 :( - not enough data
        for pos in range(max(0, pos_current-COHESION_WIND), min(pos_current+ COHESION_WIND + 1, length)):
            word,low_word,pos_tag,stem = self.get_label_names(sequence,pos)
            y_curr_name = sequence.sequence_list.y_dict.get_label_name(sequence.y[pos])
            
            if low_word_0 not in filter_names and low_word not in filter_names:
                word_0_ascii = unicodedata.normalize('NFKD', low_word_0).encode('ascii','ignore').decode('unicode_escape')
                word_0_ascii = stemAugmented(word_0_ascii)

                ##    SOLO HACER COHESION EN I-I | I-B
                if all([    any([y_name[0]=='I' and y_curr_name[0]=='B' ,#and pos<pos_current,
                                 y_name[0]=='I' and y_curr_name[0]=='I',]),
                            y_name[0] == 'I',
                            word_0_ascii in self.inner_trigger_words['B']       ]): # B o I da igual
                    #features = self.get_trigger_features(low_word, y_name, prefix='insideNE', _dict=self.inner_trigger_words,
                    #                                     features=features)
                    features = self.get_trigger_features(low_word, y_name, prefix='innerTW', _dict=self.inner_trigger_words,
                                                         pos=pos-pos_current, features=features)
            #############################################################################

        return features





    def add_transition_features(self, sequence, pos, y, y_prev, features):
        """ Adds a feature to the edge feature list.
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
        # TRANSITION_2 + WORD_2, + POS_2
        word ,low_word ,pos_tag  ,stem  = self.get_label_names(sequence,pos)
        word1,low_word1,pos_1_tag,stem1 = self.get_label_names(sequence,pos-1)

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
            feat_name = "TRANS_POS::%s::%s_%s::%s" % (y_name,y_prev_name,pos_tag,pos_1_tag)
            features = self.insert_feature(feat_name, features)

        #############################################################################
        ###############################
        ## Y,Y -> ORT,ORT
        # WINDOW ORTOGRAPHIC FEATURES
            rare_ort = True
            ort = []
            ort1 = []
            if word in filter_names:
                ort = [word]
            else:
                for i, pat in enumerate(ORT):
                    if pat.search(word):
                        rare_ort = False
                        ort.append(DEBUG_ORT[i])
                if rare_ort:
                    ort = ["OTHER_ORT"]
            
            rare_ort = True
            if word1 in filter_names:
                ort1 = [word1]
            else:
                for i, pat in enumerate(ORT):
                    if pat.search(word1):
                        rare_ort = False
                        ort1.append(DEBUG_ORT[i])
                if rare_ort:
                    ort1 = ["OTHER_ORT"]
            for u in ort:
                for v in ort1:
                    feat_name = "TRANS_ORT::%s:%s__%s:%s" % (y_name,y_prev_name,u,v)
                    features = self.insert_feature(feat_name, features)
        #############################################################################
        

        return features


