import sys
import numpy as np
from sequences.my_math_utils import *
import matplotlib as mpl
import matplotlib.cm as cm

import operator
import pdb,ipdb


## Colour for each pos tag

START = '_START_'
END = '_END_'
START_TAG = '_START_TAG_'
END_TAG = '_STOP_'

"""
tag_colors = {
    'ADV':'Fuchsia',
    'NOUN':'ForestGreen',
    'ADP':'Blue',
    'PRON':'DarkGreen',
    'DET':'Khaki',
    '.':'Black',
    'PRT':'LightGrey',
    'NUM':'GreenYellow',
    'X':'DarkGray',
    'CONJ':'Indigo',
    'ADJ':'DarkSeaGreen',
    'VERB':'Red'}
"""
tag_colors = [
    'Blue',
    'GreenYellow',
    'Red',
	'Fuchsia',
    'Khaki',
    'Indigo',
    'ForestGreen',
    'DarkGreen',
    'Black',
    'LightGrey',
    'DarkGray',
    'DarkSeaGreen',

]

def build_confusion_matrix(truth_seq, prediction_seq, nr_states):
    import matplotlib.pyplot as plt
    matrix = {}
    for i in range(nr_states):
        matrix[i] = {}

    for i,seq in enumerate(truth_seq):
        pred = prediction_seq[i]
        for i,y_hat in enumerate(pred.y):
            y_truth = seq.y[i]
            if(y_hat not in matrix):
                matrix[y_hat] = {}
            if(y_truth not in matrix[y_hat]):
                matrix[y_hat][y_truth] = 0
            matrix[y_hat][y_truth] += 1
    return matrix




## Builds the one to many mapping
"""
def get_best_assignment(conf_matrix):
    best_tags = {}
    for i,(cluster,cluster_dist) in enumerate(conf_matrix.items()):
        value_aux = dict_max(cluster_dist)
        if value_aux != [] and value_aux != 0:
            value,tag = value_aux
            best_tags[cluster] = tag
        else:
            best_tags[cluster] = i
    return best_tags
"""
def get_best_assignment(conf_matrix):
    best_tags = {}
    for i,(cluster,cluster_dist) in enumerate(conf_matrix.items()):
        best_tags[cluster] = cluster
    return best_tags


#####################
### Splits the confusion matrix per best tags:
### What clusters have best tag nouns, verbs etc
#####################
def split_matrix_by_best_tag(conf_mat,best_tags):
    matrix_per_tag ={}
    for cluster,best_tag in best_tags.items():
        if(best_tag not in matrix_per_tag):
            matrix_per_tag[best_tag] = {}
        matrix_per_tag[best_tag][cluster]=conf_mat[cluster]
    return matrix_per_tag

#####################################
#### Get's the average purity per tag
#####################################
def get_average_purity_per_tag(conf_mat,best_tags):
    matrix_per_tag =split_matrix_by_best_tag(conf_mat,best_tags)
    purity_per_tag = {}
    for tag,matrix in matrix_per_tag.items():
       values = get_clusters_purity(matrix_per_tag[tag]).values()
       purity_per_tag[tag] = sum(values)/len(values)
    return sort_dic_by_value(purity_per_tag,reverse=True)


#############################
### Returns the purity of each cluster
#############################
def get_clusters_purity(conf_matrix):
    purity = {}
    for i,(cluster,cluster_dist) in enumerate(conf_matrix.items()):
        value,tag = dict_max(cluster_dist)
        total = sum(cluster_dist.values())
        purity[cluster] = 100.0*value/total
    return purity


#####
### Returns a list of clusters sorted by their purity
#####
def sort_conf_matrix_by_purity(conf_matrix):
    return sort_dic_by_value(get_clusters_purity(conf_matrix),reverse=True)


def plot_confusion_bar_graph(matrix,pos_list,clusters,title):
    import matplotlib.pyplot as plt
    norm = mpl.colors.Normalize(vmin=-1, vmax=len(clusters))
    cmap = cm.rainbow
    mapper = m = cm.ScalarMappable(norm=norm, cmap=cmap)

    ## Get the mapping
    mapping = get_best_assignment(matrix)
    ## Figure details
    fig_aux = plt.figure()
    fig = fig_aux.add_subplot(1, 1, 1)
    xlocations = np.array(range(len(clusters)))
    rects = {}
    i = 0

    # print matrix
    for cluster in clusters:
        # Tags for each cluster
        cluster_tags = matrix[cluster]
        # Sort the cluster tags by their number of occurences
        sorted_tags = sort_dic_by_value(cluster_tags,reverse=True)
        bottom = 0
        for tag,value in sorted_tags:
            if(tag not in rects):
                rects[tag] = {}
            tag_name = pos_list.get_label_name(tag)
            tag_name = tag_name.upper()
            #aux = fig.bar(xlocations[i],value,bottom=bottom,linewidth=0,color=tag_colors[k],edgecolor=tag_colors[k])
            aux = fig.bar(xlocations[i],value,bottom=bottom,linewidth=0,color=tag_colors[tag],edgecolor=tag_colors[tag])
            rects[tag][0] = aux
            bottom += value
        i += 1
    fig.set_xticks(xlocations+0.4)
    best_tags_names = []
    for i in mapping:
        tag = mapping[i]
        tag_name = pos_list.get_label_name(tag)
        best_tags_names.append(tag_name)
    #print best_tags_names
    fig.set_xticklabels(best_tags_names)
    temp = [(pos_list.get_label_name(k),k) for k in clusters]
    pos_list2 = sorted(temp,key=lambda x: x[1])
    color_list = [x[0] for x in pos_list2]

    fig.legend(map(lambda t: rects[t][0], clusters), color_list,mode="expand",ncol=8)
    #fig.autoscale()
    plt.show()
