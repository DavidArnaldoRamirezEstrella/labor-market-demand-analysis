from utils_new import *

#############################################################################################################################
def makeCorpus(data,idx,mode='by_sent',START_END_TAGS=True, stem_vocab=[], pos_vocab=[], filter_empty=False, extended_filter = True):
    # El output sale ya filtrado para cualquier corpus
    corpus = Corpus()
    if mode=='by_doc':
        corpus.ne_dict.add(BR)
    if START_END_TAGS:
        corpus.word_dict.add(START)
        corpus.word_dict.add(END)
        corpus.ne_dict.add(START_TAG)
        corpus.ne_dict.add(END_TAG)
        corpus.pos_dict.add(START_TAG)
        corpus.pos_dict.add(END_TAG)
        
    seq_list = []
    file_ids = []
    br_pos_list = []
    name_folder = ''
    corpus.stem_vocabulary = stem_vocab

    for id in idx:
        doc = data[id]
        if id < 400:
            name_folder='car_tag_' + str(id+1)
        else:
            name_folder='random_' + str(id-400+1)

        if mode=='by_sent':
            empty_sample = True
            if filter_empty:
                for sent in doc:
                    out = False
                    for tup in sent:
                        x,pos,y = tup
                        if y != 'O':    # si oracion no solo tiene O
                            empty_sample = False
                            out = True
                            break
                    if out:
                        break

            for i,sent in enumerate(doc):
                sent_x = []
                sent_y = []
                sent_pos = []
                #empty_sample = True
                if START_END_TAGS:
                    sent_x   = [START    , START]
                    sent_y   = [START_TAG, START_TAG]
                    sent_pos = [START_TAG, START_TAG]
                for tup in sent:
                    x,pos,y = tup
                    if extended_filter:
                        x = permanentFilter(x)
                    stem = stemAugmented(x.lower())
                    if x not in filter_names and stem_vocab!=[] and stem not in stem_vocab:
                        x = assignFilterTag(x)
                    if pos_vocab!=[] and pos not in pos_vocab:
                        pos = NOUN
                    #if y != 'O':    # si oracion no solo tiene O
                    #    empty_sample = False
                    if x not in corpus.word_dict:
                        corpus.word_dict.add(x)
                    if y not in corpus.ne_dict:
                        corpus.ne_dict.add(y)
                    if pos not in corpus.pos_dict:
                        corpus.pos_dict.add(pos)
                    sent_x.append(x)
                    sent_y.append(y)
                    sent_pos.append(pos)
                if START_END_TAGS:
                    sent_x.append(END)
                    sent_y.append(END_TAG)
                    sent_pos.append(END_TAG)
                if any([not empty_sample and filter_empty,
                        not filter_empty]):
                    seq_list.append([sent_x,sent_y,sent_pos])
                    file_ids.append(name_folder)
                    br_pos_list.append(i)
        else:
            sent_x = []
            sent_y = []
            sent_pos = []
            br_positions = []
            if START_END_TAGS:
                sent_x   = [START    , START]
                sent_y   = [START_TAG, START_TAG]
                sent_pos = [START_TAG, START_TAG]
                br_positions.append(1)  # segundo START como BR
            empty_sample = True
            for i,tup in enumerate(doc):
                x,pos,y = tup
                if x != BR:
                    if extended_filter:
                        x = permanentFilter(x)
                    stem = stemAugmented(x.lower())
                    if x not in filter_names and stem_vocab!=[] and stem not in stem_vocab:
                        x = assignFilterTag(x)
                else:
                    br_positions.append(i+2) # desfase por START,START
                if pos_vocab!=[] and pos not in pos_vocab:
                    pos = NOUN
                if y != 'O' and y != BR:    # si oracion no solo tiene O & BR
                    empty_sample = False
                if x not in corpus.word_dict:
                    corpus.word_dict.add(x)
                if y not in corpus.ne_dict:
                    corpus.ne_dict.add(y)
                if pos not in corpus.pos_dict:
                    corpus.pos_dict.add(pos)
                sent_x.append(x)
                sent_y.append(y)
                sent_pos.append(pos)
            if sent_x[-1] == BR:
                sent_x.pop()
                sent_y.pop()
                sent_pos.pop()
            if START_END_TAGS:
                sent_x.append(END)
                sent_y.append(END_TAG)
                sent_pos.append(END_TAG)
                br_positions.append(len(sent_x)-1)
            seq_list.append([sent_x,sent_y,sent_pos])
            file_ids.append(name_folder)
            br_pos_list.append(br_positions)

    sequence_list = SequenceList(corpus.word_dict, corpus.pos_dict, corpus.ne_dict, corpus.stem_vocabulary)

    for i,(x,y,pos) in enumerate(seq_list):
        sequence_list.add_sequence(x, y, pos,file_ids[i],br_pos_list[i])
    return sequence_list



def reader_HMM(tags=TAGS,target='BIO', mode='by_doc', simplify_POS=True):
    '''
       param: tags: list of NE to consider
       param: target:
               - BIO: only I,B,O tags considered
               - TODO: complete NE tag considered
       param: mode: - by_doc: whole doc considered as one sentence
                    - by_sent: doc is split in sentences
       param: simplify_POS: apply simplifying rules to POS
       param: path: folder where to find the data
       return: list of tuples (word, pos, ne)
    '''
    data = []
    for i in range(1, 401):
        doc = []
        sent = []
        for line in open(careers_dir + str(i) + '.tsv'):
            line = line.strip('\n').strip(' ')
            x = ''
            y = ''
            pos = ''
            if len(line)>0:
                temp = line.split('\t')
                pos = simplify2ep(temp[1])
                x = temp[0]
                if len(temp) != 3:
                    ipdb.set_trace()

                if temp[-1][2:] in tags:
                    if target == 'BIO':
                        y = temp[-1][0]
                    else:
                        y = temp[-1]
                else:
                    y = 'O'
                sent.append(tuple([x,pos,y]))
            else:
                if len(sent)>0:
                    doc.append(sent)
                sent = []
        if mode == 'by_doc':
            temp = []
            for sent in doc:
                temp.extend(sent)
                temp.append((BR,BR,BR))
            temp.pop()
            doc = list(temp)
        data.append(doc)

    for i in range(1, 401):
        doc = []
        sent = []
        for line in open(random_dir + str(i) + '.tsv'):
            line = line.strip('\n').strip(' ')
            x = ''
            y = ''
            pos = ''
            if len(line)>0:
                temp = line.split('\t')
                pos = simplify2ep(temp[1])
                x = temp[0]
                if len(temp) != 3:
                    ipdb.set_trace()

                if temp[-1][2:] in tags:
                    if target == 'BIO':
                        y = temp[-1][0]
                    else:
                        y = temp[-1]
                else:
                    y = 'O'
                sent.append(tuple([x,pos,y]))
            else:
                if len(sent)>0:
                    doc.append(sent)
                sent = []
        if mode == 'by_doc':
            temp = []
            for sent in doc:
                temp.extend(sent)
                temp.append((BR,BR,BR))
            temp.pop()
            doc = list(temp)
        data.append(doc)
    return data
    


def getData_HMM(test=0.1, val=0.1, mode='by_sent', tags=TAGS, target='BIO', START_END_TAGS=True, extended_filter=True, filter_empty=True):
    print(":: Reading parameters")
    print("Mode  : ",mode)
    print("Target: ",target)
    print("Extended filter tags: ",extended_filter)
    print("Filter empty samples: ",filter_empty)
    print("NEs: ",tags)
    print("=======================================")
    startTime = datetime.now()

    idx = range(800)
    val_size = 0
    if val+test != 0.0:
        val_size = val/(val+test)

    #train_idx,temp = train_test_split(idx ,test_size = test+val, random_state=RANDOM_STATE)
    train_idx,temp = custom_train_test(test_size = test+val)
    try:
        test_idx,val_idx = train_test_split(temp,test_size = val_size, random_state=RANDOM_STATE)
    except:
        test_idx = val_idx = []

    raw_data = reader_HMM(tags=tags, mode=mode, target=target)

    stem_vocab,pos_vocab = make_word_counts(raw_data,train_idx,mode=mode, extended_filter=extended_filter)


    temp = makeCorpus(data=raw_data, idx=range(800), mode=mode, START_END_TAGS=START_END_TAGS,
                        stem_vocab=stem_vocab, pos_vocab=pos_vocab, filter_empty=filter_empty, extended_filter=extended_filter)
    
    train = SequenceList(temp.x_dict, temp.pos_dict, temp.y_dict)
    test  = SequenceList(temp.x_dict, temp.pos_dict, temp.y_dict)
    val    = SequenceList(temp.x_dict, temp.pos_dict, temp.y_dict)
    train.seq_list = [temp.seq_list[i] for i in range(800) if i in train_idx]
    test.seq_list  = [temp.seq_list[i] for i in range(800) if i in test_idx]
    val.seq_list   = [temp.seq_list[i] for i in range(800) if i in val_idx]

    print("Dataset analysis:")
    print(":: Training set")
    print("Size training set: ",len(train.seq_list))
    print_tag_proportion(train)
    print("---------------------------------------")
    print(":: Testing set")
    print("Size testing set: ",len(test.seq_list))
    print_tag_proportion(test)
    print("---------------------------------------")
    print(":: Validation set")
    print("Size validation set: ",len(val.seq_list))
    print_tag_proportion(val)
    print("---------------------------------------")
    print("Execution time: ",datetime.now()-startTime)
    print("=======================================")

    return train,test,val
