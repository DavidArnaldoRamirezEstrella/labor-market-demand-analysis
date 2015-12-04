import os, sys
import re, string
import pdb, ipdb
import unicodedata
import numpy as np

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_utils)

from sequences.label_dictionary import *
from utils_new import uploadObject, assignFilterTag, filter_names, stemAugmented


##########################################################################################################
path_wordcluster = os.path.join(path_utils,'word_clustering')

#wordcluster_source = os.path.join(path_wordcluster,'output_casi')
MX_LEN_BITSTREAM = 51       # MAX_LEN en Liand 5000 es 32
cluster_prefixes = range(8,MX_LEN_BITSTREAM,4)
##########################################################################################################

MAX_SUFFIX = 4
MAX_PREFIX = 3

LONG_WORD_THR = 20
PUNCTUATION = string.punctuation + 'º¡¿ª°'

START = '_START_'
END = '_END_'
START_TAG = '<START>'
END_TAG = '<STOP>'
BR = '**'
RARE = "<RARE>"
NOUN = 'nc'

NO_LABELS = [
    START,
    END,
    START_TAG,
    END_TAG,
    BR,
    RARE,
]
filter_names = list(set(filter_names + NO_LABELS))

##########################################################################################################
CAPS_SPECIAL = 'ÁÉÍÓÚÄËÏÖÜÂÊÎÔÛÀÈÌÒÙÑ'
MIN_SPECIAL = 'áéíóúäëïöüâêîôûñ'

##########################################################################################################
ORT = [
    re.compile(r'^[A-Z%s]+$'        % CAPS_SPECIAL, re.UNICODE),                         # ALL CAPS
    re.compile(r'^[A-Z%s][a-z%s]+'  % (CAPS_SPECIAL,MIN_SPECIAL), re.UNICODE),           # CAPITALIZED
    re.compile(r'^[A-Z%s][a-z%s]{1,2}\.?$'  % (CAPS_SPECIAL,MIN_SPECIAL), re.UNICODE),   # POSSIBLE ACRONYM
    re.compile(r'[A-Z%s]'                   % CAPS_SPECIAL, re.UNICODE),                 # HAS ANY UPPERCASE CHAR
    re.compile(r'^[a-z%s]+$'        % MIN_SPECIAL, re.UNICODE),                          # ALL LOWERCASE
    re.compile(r'[a-z%s]'           % MIN_SPECIAL, re.UNICODE),                          # ANY LOWERCASE CHAR
    re.compile(r'^[a-zA-Z%s%s]$'    % (CAPS_SPECIAL,MIN_SPECIAL), re.UNICODE),           # SINGLE CHAR
    re.compile(r'^[0-9]$', re.UNICODE),                         # SINGLE DIGIT
    #re.compile(r'^.{%i,}$' % LONG_WORD_THR, re.UNICODE),        # LONG WORD >= THRESHOLD

    #re.compile(r'^9[0-9]{8}([%s]9[0-9]{8})*$' % PUNCTUATION, re.UNICODE),  # MOBILE PHONE NUMBER
    #re.compile(r'^[0-9]{7}([%s][0-9]{7})*$' % PUNCTUATION, re.UNICODE),  # OFFICE PHONE NUMBER
    #re.compile(r'^\d+$', re.UNICODE),                           # ALL NUMBERS
    #re.compile(r'^\d+([%s]+\d*)+$' % PUNCTUATION, re.UNICODE),  # NUMBER PUNCT NUMBER PUNCT?

    #re.compile(r'^(([sS$]/\.?)|\$)[0-9a-z%s]+([.,]\d+)?([.,]\d+)?([.,]\d+)?([.,]\d+)?$' % MIN_SPECIAL, re.UNICODE),    # IS MONEY

    #re.compile(r'^\d{1,2}(:\d{1,2})?[ªaApPáÁàÀ]\.?[mM]\.?$', re.UNICODE),      # HOUR 9am, 9:00am, 9no, ...
    #re.compile(r'^[ªaApPáÁàÀ]\.?[mM]\.?$', re.UNICODE),      # HOUR am pm a.m p.m
    #re.compile(r'\d{1,2}:\d{1,2}([-/]\d{1,2}:\d{1,2})*', re.UNICODE),      # RANGO HORARIO U HORA SIMPLE

    re.compile(r'\d', re.UNICODE),  # HAS DIGIT
    re.compile(r'^\w+$', re.UNICODE),  # ALPHA-NUMERIC
    re.compile(r'^[a-zA-Z%s%s]+[%s]+[a-zA-Z%s%s]+([%s]+[a-zA-Z%s%s]+)?$' % 
                            (CAPS_SPECIAL,MIN_SPECIAL,
                            PUNCTUATION,
                            CAPS_SPECIAL,MIN_SPECIAL,
                            PUNCTUATION,
                            CAPS_SPECIAL,MIN_SPECIAL), re.UNICODE),  # alpha + PUNCT + alpha
        
    #re.compile(r'^[xX]+([%s][xX]+)?$' % PUNCTUATION, re.UNICODE),  # ONLY X's - for emails, websites
    #re.compile(r'(www)|(https?)|(gmail)|(WWW)|(HTTPS?)|(GMAIL)|(^com$)|(^COM$)|(php)|(PHP)', re.UNICODE),  # urls

    re.compile(r'-+', re.UNICODE),  # CONTAINS HYPENS
    re.compile(r'^[%s]+$' % PUNCTUATION, re.UNICODE),  # IS PUNCTUATION
    re.compile(r'[%s]' % PUNCTUATION, re.UNICODE),  # HAS PUNCTUATION
]

DEBUG_ORT = [
    'ALL_CAPS',
    'CAPITALIZED',
    'ACRONYM',
    'ANY_UPPERCASE',
    'ALL_LOWERCASE',
    'ANY_LOWERCASE',
    'SINGLE CHAR',
    'SINGLE DIGIT',
    #'LONG WORD',
    #'MOB_NUMBER',
    #'OFFICE_NUMBER',    
    #'ALL_NUMBERS',
    #'NUMBER_PUNCT',
    #'MONEY',
    #'HOUR',
    #'HOUR',
    #'HOUR',
    'HAS_DIGIT',
    'ALPHA_NUMERIC',
    'ALPHA_PUNCT_ALPHA',
    #'URL',
    #'URL',
    'HAS_HYPENS',
    'PUNCTUATION',
    'HAS_PUNCTUATION',

    'OTHERS',
]

NUMBER = [
    re.compile(r'^\d+$'),  # ALL NUMBERS
    re.compile(r'^\d+([.,]\d+)?([.,]\d+)?$'),  # NUMBER W/ COMMA OR PERIOD
    re.compile(r'^((S/\.?)|\$)\s*\d+([.,]\d+)?([.,]\d+)?$'),  # IS MONEY
    re.compile(r'^\d+:\d{2}$'),  # HOUR 10:00
]

LIST_PUNCTUATION = '-,/;\|oO()'
LIST_SEPARATOR = re.compile(r'^[%s]$' % LIST_PUNCTUATION, re.UNICODE)

EXTERNAL_GAZZETER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),'external_gazetters')


######################################################################################################
### Replicates the same features as the HMM
### One for word/tag and tag/tag pair
#################

RARE = "<RARE>"

class IDFeatures:

    def __init__(self, dataset, mode='by_sent'):
        '''dataset is a sequence list.'''
        self.feature_dict = LabelDictionary()
        self.add_features = False
        self.dataset = dataset
        self.feature_templates = {}

        #Speed up
        self.node_feature_cache = {}
        self.initial_state_feature_cache = {}
        self.final_state_feature_cache = {}
        self.edge_feature_cache = {}

        self.wordcluster_source = os.path.join(path_wordcluster,'output_liang_' + mode +'_5000')
        self.word_clusters = {}

        self.word_reference = uploadObject(os.path.join(path_utils,'word_dict_filtered')) # word dict from training data
        self.word_reference[BR] = 1
        
        self.inner_trigger_words= {'I':{}}
        self.outer_trigger_words= {}

        self.outer_trigger_pos= {}
        self.inner_trigger_pos = {}

        if mode=='by_sent':
            self.FILTER_WORDS = 0.001
            #self.FILTER_WORDS = 0.005
            self.FILTER_POS = 0.001
        else:
            self.FILTER_WORDS = 0.01
            self.FILTER_POS = 0.001


    def get_num_features(self):
        return len(self.feature_dict)


    def get_y_name(self,sequence,pos):
        return sequence.sequence_list.y_dict.get_label_name(sequence.y[pos])


    def get_label_names(self,sequence,pos, escape_ascii=False):
        '''
            escape_ascii: solo afecta al stem, word se emite tal cual es
        '''
        x = sequence.x[pos]
        pos_id = sequence.pos[pos]

        word    = sequence.sequence_list.x_dict.get_label_name(x)
        pos_tag = sequence.sequence_list.pos_dict.get_label_name(pos_id)
        
        if self.dataset.pos_dict.get_label_id(pos_tag) == -1:
            pos_tag = NOUN
        
        low_word = ''
        stem = ''
        if word in filter_names:
            low_word = stem = word
        else:
            low_word = word.lower()
            if escape_ascii:
                word_ascii = unicodedata.normalize('NFKD', low_word).encode('ascii','ignore').decode('unicode_escape')
                stem = stemAugmented(word_ascii)
            else:
                stem = stemAugmented(low_word)

        return (word,low_word,stem)


    def get_sequence_features(self, sequence):
        ## Take care of middle positions
        features = []
        for pos in range(2,len(sequence.y)):
            y_1 = sequence.y[pos-1]
            y_2 = sequence.y[pos-2]
            features = self.get_features(sequence, pos, y_1, y_2,features)
        features = set(features)
        
        return features



    def get_ortographic_features(self, word, pos=None, features = []):
        name_pattern = 'ort::'
        if pos != None:
            name_pattern += str(pos) + "::"

        if word in filter_names:
            return features
        orts = self.get_ortography(word)
        for ort_name in orts:
            feat_name = name_pattern  +  ort_name
            features = self.insert_feature(feat_name,features)

        return features

    def get_ortography(self,word):
        if word in filter_names:
            return [word]
    
        rare_ort = True
        RES = []
        for i, pat in enumerate(ORT):
            if pat.search(word):
                rare_ort = False
                RES.append(DEBUG_ORT[i])
        if rare_ort:
            RES.append("OTHER_ORT")
        return RES



    def get_trigger_features(self, word, y_name, prefix, pos_tag=False, _dict={}, pos=None, features=[]):
        name_pattern = prefix + '::'
        if pos!=None:
            name_pattern += str(pos) + ':'

        if word not in filter_names:
            word = unicodedata.normalize('NFKD', word.lower()).encode('ascii','ignore').decode('unicode_escape')
            # confiando q y_name esta en _dict
            if not pos_tag:
                word = stemAugmented(word)

        if word in _dict[y_name]:
            feat_name = name_pattern + y_name
            features = self.insert_feature(feat_name, features)
        return features
    

    def get_word_cluster_features(self,lower_word, pos=None, features=[]):
        name_pattern = "cluster"
        if pos != None:
            name_pattern += '_' + str(pos)
        name_pattern += '::'

        clusters = self.get_word_clusters(lower_word)
        for clust in clusters:
            feat_name = name_pattern + "pref_%i::%s" % (len(clust),clust)
            features = self.insert_feature(feat_name,features)
        return features

    def get_word_clusters(self,lower_word):
        if lower_word in [BR,START,END]:
            return []
        if lower_word not in filter_names and lower_word not in self.word_clusters:
            lower_word = assignFilterTag(lower_word)
        # LEL hay casos de assignFIlter que no se ven en training
        if lower_word not in self.word_clusters:
            return []
        clusters_pref = []
        bitstream = self.word_clusters[lower_word]
        temp_cluster_prefixes = list(cluster_prefixes)
        #if len(bitstream) < temp_cluster_prefixes[0]:
        #    temp_cluster_prefixes = [4,6] + temp_cluster_prefixes

        for pref in temp_cluster_prefixes:
            if pref < len(bitstream):
                clusters_pref.append(bitstream[:pref])
            else:
                break
        return clusters_pref

    def get_suffpref_features(self,low_word,pos=None,features=[]):
        if low_word in filter_names:
            return features

        ##Suffixes
        max_suffix = MAX_SUFFIX
        for i in range(max_suffix):
            if (len(low_word) > i + 1):
                suffix = low_word[-(i + 1):]
                # Generate feature name.
                if pos!=None:
                    feat_name = "suffix::%i::%s" % (pos, suffix)
                else:
                    feat_name = "context_suffix::%s" % suffix
                features = self.insert_feature(feat_name, features)
        ##Prefixes
        max_prefix = MAX_PREFIX
        for i in range(max_prefix):
            if (len(low_word) > i + 1):
                prefix = low_word[:i + 1]
                # Generate feature name.
                if pos!=None:
                    feat_name = "prefix::%i::%s" % (pos, prefix)
                else:
                    feat_name = "context_prefix::%s" % prefix
                features = self.insert_feature(feat_name, features)
        return features


    def get_suffixes(self,low_word):
        res = []
        if low_word in filter_names:
            return res
        ##Suffixes
        max_suffix = MAX_SUFFIX
        for i in range(max_suffix):
            if (len(low_word) > i + 1):
                suffix = low_word[-(i + 1):]
                res.append(suffix)
        return res
        
    def get_prefixes(self,low_word):
        res = []
        if low_word in filter_names:
            return res
        ##Suffixes
        max_prefix = MAX_PREFIX
        for i in range(max_prefix):
            if (len(low_word) > i + 1):
                prefix = low_word[:i + 1]
                res.append(prefix)
        return res


    def get_inner_list_features(self,sequence,pos,include_stem=False,features=[]):
        """
        stem esta en formato ascii para hacer match siempre con InnerTriggerWords
        FALTA CORREGIR PARA Q NO INCLUYA Y_NAME
        """
        length = len(sequence.y)
        w_prev,_,_  = self.get_label_names(sequence,pos-1)
        w,_,stem  = self.get_label_names(sequence,pos, escape_ascii=True)
        w_next = ''
        if pos < length-1:
            w_next,_,_  = self.get_label_names(sequence,pos+1)
        m_prev = LIST_SEPARATOR.search(w_prev)
        m_curr = LIST_SEPARATOR.search(w)
        m_next = LIST_SEPARATOR.search(w_next)

        if not m_curr and m_prev and m_next:
            feat_name=''
            if include_stem:
                feat_name = "inner_list::%s__%s" % (y_name,stem)
            else:
                feat_name = "inner_list::%s" % y_name
            features = self.insert_feature(feat_name,features)
        return features



    def add_feature(self, feat_name):
        """
        Builds a dictionary of feature name to feature id
        If we are at test time and we don't have the feature
        we return -1.
        """
        if self.add_features:
            idx = feat_name.find('::')
            pref = feat_name[:idx]
            suff = feat_name[idx+2:]
            if pref not in self.feature_templates:
                self.feature_templates[pref] = set()
            self.feature_templates[pref].add(suff)

        # Check if feature exists and if so, return the feature ID.
        if(feat_name in self.feature_dict):
            return self.feature_dict[feat_name]
        # If 'add_features' is True, add the feature to the feature
        # dictionary and return the feature ID. Otherwise return -1.
        if not self.add_features:
            return -1
        return self.feature_dict.add(feat_name)

    def insert_feature(self, feat_name, features=[]):
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            if self.add_features:
                features.append(feat_name)
            else:
                features.append(feat_id)
        return features


    def get_features_parallel(self,params):
        sequence,predicted_sequence,pos,learning_rate = params
        y_hat = predicted_sequence.y

        y_t_true = sequence.y[pos]
        y_t_hat = y_hat[pos]
        prev_y_t_true = sequence.y[pos-1]
        prev_y_t_hat = y_hat[pos-1]

        phi = np.zeros(self.get_num_features())

        if y_t_true != y_t_hat:
            true_emission_features = self.get_emission_features(sequence, pos, y_t_true)
            phi[true_emission_features] += learning_rate

            hat_emission_features = self.get_emission_features(sequence, pos, y_t_hat)
            phi[hat_emission_features] -= learning_rate

        if(y_t_true != y_t_hat or prev_y_t_true != prev_y_t_hat):
            true_transition_features = self.get_transition_features(sequence, pos-1, y_t_true, prev_y_t_true)
            phi[true_transition_features] += learning_rate

            hat_transition_features = self.get_transition_features(sequence, pos-1, y_t_hat, prev_y_t_hat)
            phi[hat_transition_features] -= learning_rate
        return phi


    def get_scores_parallel(self,params):
        sequence,pos,parameters,num_states = params

        length = len(sequence.x)
        emission_scores = np.zeros(num_states)
        transition_scores = np.zeros([num_states, num_states])

        for tag_id in range(num_states):
            #############################################
            emission_features = self.get_emission_features(sequence, pos, tag_id)
            score = 0.0
            for feat_id in emission_features:
                score += parameters[feat_id]
            emission_scores[tag_id] = score

            for prev_tag_id in range(num_states):
                #############################################
                transition_features = self.get_transition_features(sequence, pos-1, tag_id, prev_tag_id)
                score = 0.0
                for feat_id in transition_features:
                    score += parameters[feat_id]
                transition_scores[tag_id, prev_tag_id] = score

        return emission_scores,transition_scores