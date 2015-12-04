#from sequences.id_feature import IDFeatures
from sequences.id_feature_bigram import IDFeatures
from sequences.label_dictionary import *

import os, sys
import re, string
import pdb, ipdb
import unicodedata

path_utils = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(path_utils)

from utils_new import uploadObject, assignFilterTag, filter_names, stemAugmented

##########################################################################################################
path_wordcluster = os.path.join(path_utils,'word_clustering')

#wordcluster_source = os.path.join(path_wordcluster,'output_casi')
MX_LEN_BITSTREAM = 51
cluster_prefixes = range(8,MX_LEN_BITSTREAM,4)
##########################################################################################################
WINDOW = 8

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

EXTERNAL_GAZZETER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),'external_gazetters')
#######################
#### Feature Class
### Extracts features from a labeled corpus (only supported features are extracted
#######################
class ExtendedFeatures(IDFeatures):
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

        self.outer_trigger_words= {}
        self.inner_trigger_words= {}

        self.outer_trigger_pos= {}
        self.inner_trigger_pos = {}

        if mode=='by_sent':
            self.FILTER_WORDS = 0.005
            self.FILTER_POS = 0.001
        else:
            self.FILTER_WORDS = 0.005
            self.FILTER_POS = 0.001

    def filter_common(self, _dict):
        '''
        :param _dict:  { NE1 : [], NE2 : [], ...}
        :return:
        '''
        res = {}
        tokens = []
        for tw in _dict.values():
            tokens.extend(tw)
        common = set(tokens)
        for tw in _dict.values():
            common &= set(tw)
        for ne,tw in _dict.items():
            res[ne] = list(set(tw) - common)
        # Otra por si alguno se queda vacio
        common = set(tokens)
        for tw in res.values():
            if tw == []:
                continue
            common &= set(tw)
        new_res = {}
        for ne,tw in res.items():
            new_res[ne] = list(set(tw) - common)
        return new_res

    def filter_unfrequent(self,_dict, threshold):
        res = {}
        for ne,tw_dict in _dict.items():
            mn = min([val for key,val in tw_dict.items()])
            mx = max([val for key,val in tw_dict.items()])
            rg = (mx - mn)
            tw = []
            if rg == 0:
                tw = tw_dict.keys()
            else:
                tw = [trigger for trigger,value in tw_dict.items() if 1.0*(value-mn)/rg >= threshold]
            res[ne] = tw
        res = self.filter_common(res)
        return res

    def include_gazzeter(self):
        careers_gazzeter = open(os.path.join(EXTERNAL_GAZZETER_DIR,'carreras'),'r')

        for carrera in careers_gazzeter:
            carrera = unicodedata.normalize('NFKD', carrera.lower().strip('\n')).encode('ascii','ignore').decode('unicode_escape')
            self.inner_trigger_words['B'].append(carrera)
            self.inner_trigger_words['I'].append(carrera)
        self.inner_trigger_words['B'] = list(set(self.inner_trigger_words['B']))
        self.inner_trigger_words['I'] = list(set(self.inner_trigger_words['I']))

    def update_tw(self, sequence, pos_current):
        '''
        if B: update inner and extern context
        elif I: update inner
        elif O: only update -inner,extern- if it's first O after I
        '''
        length = len(sequence.x)
        y_name = sequence.sequence_list.y_dict.get_label_name(sequence.y[pos_current])
        y_1_name = sequence.sequence_list.y_dict.get_label_name(sequence.y[pos_current-1])

        TW_WINDOW = 5
        extremos = range(max(0, pos_current-TW_WINDOW), min(pos_current+TW_WINDOW + 1, length))

        ## outer TRIGGER WORD & POS
        if any(['B'==y_name[0],
                'O'==y_name[0] and 'I'==y_1_name[0],
                'O'==y_name[0] and 'B'==y_1_name[0]]):
            for pos in extremos:
                if y_name[0] == 'O' and pos < pos_current:
                    continue
                if y_name[0] == 'B' and pos >= pos_current:
                    continue

                x = sequence.x[pos]
                word = sequence.sequence_list.x_dict.get_label_name(x).lower()
                word = unicodedata.normalize('NFKD', word).encode('ascii','ignore').decode('unicode_escape')
                stem = stemAugmented(word)
                if stem not in filter_names and stem not in self.dataset.stem_vocabulary:
                    word = assignFilterTag(word)
                
                pos_id = sequence.pos[pos]
                pos_tag = sequence.sequence_list.pos_dict.get_label_name(pos_id)
                if self.dataset.pos_dict.get_label_id(pos_tag) == -1:
                    pos_tag = NOUN

                if any([pos_tag[0] =='s',   # PREPOS
                        pos_tag[0] =='c',   # CONJ
                        pos_tag[0] =='d',   # DETERM
                        ]):
                    continue

                if y_name not in self.outer_trigger_words:
                    self.outer_trigger_words[y_name] = {}
                if y_name not in self.outer_trigger_pos:
                    self.outer_trigger_pos[y_name] = {}

                # TRIGGER WORD
                if word not in self.outer_trigger_words[y_name]:
                    self.outer_trigger_words[y_name][word] = 0
                self.outer_trigger_words[y_name][word] += 1
                # TRIGGER POS
                if pos_tag not in self.outer_trigger_pos[y_name]:
                    self.outer_trigger_pos[y_name][pos_tag] = 0
                self.outer_trigger_pos[y_name][pos_tag] += 1

        ## INNER TRIGGER WORD & POS
        if y_name[0] != 'O' and y_name!=BR:
            x = sequence.x[pos_current]
            word = sequence.sequence_list.x_dict.get_label_name(x).lower()
            word = unicodedata.normalize('NFKD', word).encode('ascii','ignore').decode('unicode_escape')

            stem = stemAugmented(word)
            if stem not in self.dataset.stem_vocabulary:
                word = assignFilterTag(word)

            pos_id = sequence.pos[pos_current]
            pos_tag = sequence.sequence_list.pos_dict.get_label_name(pos_id)
            if self.dataset.pos_dict.get_label_id(pos_tag) == -1:
                pos_tag = NOUN

            if all([pos_tag[0] !='s',   # PREPOS
                    pos_tag[0] !='c',   # CONJ
                    pos_tag[0] !='d',   # DETERM
                    ]):
                if y_name not in self.inner_trigger_words:
                    self.inner_trigger_words[y_name] = {}
                if y_name not in self.inner_trigger_pos:
                    self.inner_trigger_pos[y_name] = {}
                # TRIGGER WORD
                if y_name not in self.inner_trigger_words[y_name]:
                    self.inner_trigger_words[y_name][word] = 0
                self.inner_trigger_words[y_name][word] += 1
                # TRIGGER POS
                if pos_tag not in self.inner_trigger_pos[y_name]:
                    self.inner_trigger_pos[y_name][pos_tag] = 0
                self.inner_trigger_pos[y_name][pos_tag] += 1

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

        self.inner_trigger_pos = self.filter_unfrequent(self.inner_trigger_pos, self.FILTER_POS)
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

        #ipdb.set_trace()
        cont = 0
        for sequence in self.dataset.seq_list:
            initial_features, transition_features, final_features, emission_features = \
                    self.get_sequence_features(sequence)
            #if cont % 100 == 0:
            #    print('sample->',cont)
            #cont += 1
        self.add_features = False
        
        print("Features:",self.get_num_features())
        print("=======================================")
        #ipdb.set_trace()

    def get_label_names(self,sequence,pos):
        x = sequence.x[pos]
        y = sequence.y[pos]
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
            stem = stemAugmented(low_word)

        return (word,low_word,pos_tag,stem)

    ###################################################################################################################






    def add_emission_features(self, sequence, pos_current, y, features):
        ########    SOLO PALABRA ACTUAL : EMISSION ORIGINAL
        # Get word name from ID.
        word ,low_word ,pos_tag  ,stem  = self.get_label_names(sequence,pos_current)
        word1,low_word1,pos_1_tag,stem1 = self.get_label_names(sequence,pos_current-1)
        y_name = sequence.sequence_list.y_dict.get_label_name(y)

        """
        #############################################################################
        # SUFFIX AND PREFFIX FEATURES
        if low_word not in filter_names:
            ##Suffixes
            max_suffix = 4
            for i in range(max_suffix):
                if (len(low_word) > i + 1):
                    suffix = low_word[-(i + 1):]
                    # Generate feature name.
                    feat_name = "suffix::%s::%s" % (suffix, y_name)
                    # Get feature ID from name.
                    feat_id = self.add_feature(feat_name)
                    # Append feature.
                    if feat_id != -1:
                        features.append(feat_id)
            ##Prefixes
            max_prefix = 3
            for i in range(max_prefix):
                if (len(low_word) > i + 1):
                    prefix = low_word[:i + 1]
                    # Generate feature name.
                    feat_name = "prefix::%s::%s" % (prefix, y_name)
                    # Get feature ID from name.
                    feat_id = self.add_feature(feat_name)
                    # Append feature.
                    if feat_id != -1:
                        features.append(feat_id)
        """

        #############################################################################
        # CURR_WORD TRIGGER FEATURES
        if y_name[0]=='B' or y_name[0]=='I':
            ## TRIGGER BAG OF WORDS FEATURES
            features = self.get_trigger_features(low_word, y_name, prefix='innerTW', _dict=self.inner_trigger_words,
                                                 features=features)

            ## TRIGGER BAG OF POS FEATURES
            features = self.get_trigger_features(pos_tag, y_name, prefix='innerTP', _dict=self.inner_trigger_pos,
                                                 features=features)
        

        features = self.get_context_features(sequence,pos_current,y,features)

        return features








    def get_context_features(self, sequence, pos_current, y, features):
        # CONTEXTUAL FEATURES
        length = len(sequence.y)
        y_name = sequence.sequence_list.y_dict.get_label_name(y)
        y_1_name = sequence.sequence_list.y_dict.get_label_name(sequence.y[pos_current-1])  # SEQUENCE AHORA ES PREDICTED_SEQUENCE

        for pos in range(max(0, pos_current-WINDOW), min(pos_current+WINDOW + 1, length)):
            word,low_word,pos_tag,stem, = self.get_label_names(sequence,pos)

            #############################################################################
            # WINDOW WORD FORM + POSITION
            """
            feat_name = "id::%i::%s" % (pos-pos_current, low_word)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
            """

            # WINDOW WORD + EMISSION ORIGINAL
            feat_name = "id::%i::%s::%s" % (pos-pos_current, low_word, y_name)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

            #############################################################################
            # WINDOW POS_tag + POSITION
            """
            feat_name = "pos::%i::%s" % (pos-pos_current, pos_tag)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
            """

            # WINDOW POS_tag + EMISSION ORIGINAL
            feat_name = "pos::%i::%s::%s" % (pos-pos_current, pos_tag, y_name)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

            #############################################################################
            # WINDOW STEM + POSITION
            """
            feat_name = "stem_id::%i::%s" % (pos - pos_current, stem)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
            """

            # WINDOW STEM + EMISSION ORIGINAL
            feat_name = "stem::%i::%s::%s" % (pos-pos_current, stem, y_name)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

            #############################################################################
            ### NO EMISSION
            """
            ## BAG OF WORDS AS EXTERN CONTEXT
            feat_name = "context::id:%s" % (low_word)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

            # BAG OF POS_tag AS EXTERN CONTEXT
            feat_name = "context::pos:%s" % pos_tag
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

            # BAG OF STEM AS EXTERN CONTEXT
            feat_name = "context::stem:%s" % stem
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)
            """

            #############################################################################
            ### EMISSION
            ## BAG OF WORDS AS EXTERN CONTEXT
            feat_name = "context::id:%s::%s" % (low_word,y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

            # BAG OF POS_tag AS EXTERN CONTEXT
            feat_name = "context::pos:%s::%s" % (pos_tag,y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

            # BAG OF STEM AS EXTERN CONTEXT
            feat_name = "context::stem:%s::%s" % (stem,y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

            # CONTEXT SUFFIX AND PREFFIX BAG FEATURES
            if low_word not in filter_names and (pos-pos_current) in [-1,0,1]:
                ##Suffixes
                max_suffix = 4
                for i in range(max_suffix):
                    if (len(low_word) > i + 1):
                        suffix = low_word[-(i + 1):]
                        # Generate feature name.
                        feat_name = "context::suffix::%s::%s" % (suffix, y_name)
                        # Get feature ID from name.
                        feat_id = self.add_feature(feat_name)
                        # Append feature.
                        if feat_id != -1:
                            features.append(feat_id)
                ##Prefixes
                max_prefix = 3
                for i in range(max_prefix):
                    if (len(low_word) > i + 1):
                        prefix = low_word[:i + 1]
                        # Generate feature name.
                        feat_name = "context::prefix::%s::%s" % (prefix, y_name)
                        # Get feature ID from name.
                        feat_id = self.add_feature(feat_name)
                        # Append feature.
                        if feat_id != -1:
                            features.append(feat_id)

            #############################################################################
            # WINDOW ORTOGRAPHIC FEATURES
            #features = self.get_ortographic_features(word,pos=pos - pos_current,features=features)
            
            # emission + ort + POSITION
            features = self.get_ortographic_features(word,y_name=y_name, pos=pos - pos_current,features=features)

            #############################################################################
            # REDUCED WINDOW WORD-CLUSTER FEATURES
            #if (pos-pos_current) in [-1,0,1]:
            features = self.get_word_cluster_features(low_word,pos-pos_current,y_name,features)

            #############################################################################
            # SUFFIX AND PREFFIX FEATURES | NO EMISSION
            """
            if low_word not in filter_names:
                ##Suffixes
                max_suffix = 4
                for i in range(max_suffix):
                    if (len(low_word) > i + 1):
                        suffix = low_word[-(i + 1):]
                        # Generate feature name.
                        feat_name = "suffix::%i::%s" % (pos - pos_current, suffix)
                        # Get feature ID from name.
                        feat_id = self.add_feature(feat_name)
                        # Append feature.
                        if feat_id != -1:
                            features.append(feat_id)
                ##Prefixes
                max_prefix = 3
                for i in range(max_prefix):
                    if (len(low_word) > i + 1):
                        prefix = low_word[:i + 1]
                        # Generate feature name.
                        feat_name = "prefix::%i::%s" % (pos - pos_current, prefix)
                        # Get feature ID from name.
                        feat_id = self.add_feature(feat_name)
                        # Append feature.
                        if feat_id != -1:
                            features.append(feat_id)
            """

            #############################################################################
            # SUFFIX AND PREFFIX FEATURES
            if low_word not in filter_names:
                ##Suffixes
                max_suffix = 4
                for i in range(max_suffix):
                    if (len(low_word) > i + 1):
                        suffix = low_word[-(i + 1):]
                        # Generate feature name.
                        feat_name = "suffix::%i::%s::%s" % (pos - pos_current, suffix, y_name)
                        # Get feature ID from name.
                        feat_id = self.add_feature(feat_name)
                        # Append feature.
                        if feat_id != -1:
                            features.append(feat_id)
                ##Prefixes
                max_prefix = 3
                for i in range(max_prefix):
                    if (len(low_word) > i + 1):
                        prefix = low_word[:i + 1]
                        # Generate feature name.
                        feat_name = "prefix::%i::%s::%s" % (pos - pos_current, prefix, y_name)
                        # Get feature ID from name.
                        feat_id = self.add_feature(feat_name)
                        # Append feature.
                        if feat_id != -1:
                            features.append(feat_id)
            
            #############################################################################
            ## Y_t + Xpos + Xpos+1
            if pos < length-1:
                word1,low_word1,pos_1_tag,stem1 = self.get_label_names(sequence,pos+1) # OJO es POS+1

                feat_name = "Y_WORD2::%s_%s::%s" % (y_name,low_word,low_word1)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

                feat_name = "Y_POS2::%s_%s::%s" % (y_name,pos_tag,pos_1_tag)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

                feat_name = "Y_STEM2::%s_%s::%s" % (y_name,stem,stem1)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

                #####
                # Y -> ORTi ORTi+1
            #############################################################################
            ## CONTEXT TRIGGER FEATURES
            # OUTTER TRIGGER
            if any([y_name[0]=='B' and pos < pos_current,
                    y_name[0]=='O' and y_1_name[0]=='I' and pos >= pos_current,
                    y_name[0]=='O' and y_1_name[0]=='B' and pos >= pos_current ]):
                ## TRIGGER BAG OF WORDS FEATURES
                features = self.get_trigger_features(low_word, y_name, prefix='outerTW', _dict=self.outer_trigger_words,
                                                     features=features)

                ## TRIGGER BAG OF POS FEATURES
                features = self.get_trigger_features(pos_tag, y_name, prefix='outerTP', _dict=self.outer_trigger_pos,
                                                     features=features)

                """ 
                ## TRIGGER POSITION OF WORDS FEATURES   #NO APORTA!!
                features = self.get_trigger_features(low_word, y_name, prefix='outerTW', _dict=self.outer_trigger_words,
                                                     pos=pos-pos_current, features=features)

                ## TRIGGER POSITION OF POS FEATURES
                features = self.get_trigger_features(pos_tag, y_name, prefix='outerTP', _dict=self.outer_trigger_pos,
                                                     pos=pos-pos_current, features=features)
                """
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

        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if(feat_id != -1):
            features.append(feat_id)

        
        #################################################################
        # TRANSITION_2 + WORD_2, + POS_2
        word ,low_word ,pos_tag  ,stem  = self.get_label_names(sequence,pos)
        word1,low_word1,pos_1_tag,stem1 = self.get_label_names(sequence,pos-1)

        feat_name = "TRANS_WORD::%s::%s_%s::%s" % (y_name,y_prev_name,low_word,low_word1)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        feat_name = "TRANS_POS::%s::%s_%s::%s" % (y_name,y_prev_name,pos_tag,pos_1_tag)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        feat_name = "TRANS_STEM::%s::%s_%s::%s" % (y_name,y_prev_name,stem,stem1)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        #############################################################################
        """
        ###############################
        ## Y,Y -> ORT,ORT
        # WINDOW ORTOGRAPHIC FEATURES
        rare_ort = True
        feat_ort_1 = ""
        if word1 in [START,BR,END,RARE]:
            feat_ort_1 = word1
        else:
            for i, pat in enumerate(ORT):
                if pat.search(word1):
                    rare_ort = False
                    feat_ort_1 = DEBUG_ORT[i]
        if rare_ort:
            feat_ort_1 = "OTHER_ORT"
        feat_name = "TRANS_ORT:%s::%s_%s::%s" % (y_name,y_1_name,feat_ort,feat_ort_1)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)
        """

        return features


    
    def get_ortographic_features(self, word, y_name='', pos = -1, features = []):
        name_pattern = 'ort::'
        if pos != -1:
            name_pattern += "::" + str(pos) + "::"
        if y_name != '' and pos==-1:
            name_pattern += "::" + y_name + "::"
        if y_name != '' and pos!=-1:
            name_pattern += y_name + "::"

        rare_ort = True
        if word in filter_names:
            feat_name = name_pattern + word
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
            rare_ort = False
        else:
            for i, pat in enumerate(ORT):
                if pat.search(word):
                    rare_ort = False
                    feat_name = name_pattern  +  DEBUG_ORT[i]
                    feat_id = self.add_feature(feat_name)
                    if feat_id != -1:
                        features.append(feat_id)
        if rare_ort:
            feat_name = name_pattern + "OTHER_ORT"
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)
        return features


    def get_word_cluster_features(self,lower_word, pos, y_name, features):
        if lower_word in [BR,START,END]:
            return features

        if lower_word not in filter_names and lower_word not in self.word_clusters:
            lower_word = assignFilterTag(lower_word)
        if lower_word not in self.word_clusters:
            lower_word = RARE
        bitstream = self.word_clusters[lower_word]

        for pref in cluster_prefixes:
            if pref < len(bitstream):
                feat_name = "cluster::pref_%i:%i:%s::%s" % (pref,pos,y_name,bitstream[:pref])
                feat_id = self.add_feature(feat_name)
                if feat_id != -1:
                    features.append(feat_id)
            else:
                break
        return features

    def get_trigger_features(self, word, y_name, prefix, _dict, pos=-1, features=[]):
        name_pattern = prefix + '::'
        if pos != -1:
            name_pattern += str(pos) + ':'

        # confiando q y_name esta en _dict
        word_ascii = unicodedata.normalize('NFKD', word).encode('ascii','ignore').decode('unicode_escape')
        if word_ascii in _dict[y_name]:
            feat_name = name_pattern + y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)
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
