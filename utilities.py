__author__ = 'Christopher Sweet'
"""
Preprocessing script to handle data prior to feeding into network

"""
import os
import json
import time
import csv

from collections import Counter
from itertools import combinations, permutations
from pprint import pprint
from copy import deepcopy

import numpy as np
import scipy.ndimage as ndimage

FILELIST = [
]

#Define Embedding Sizes for features
# -1 means no embedding, -2 means it's a computed feature
FEATURE_EMBEDDING_SIZE = [
    ('alert.signature', -1),
    # ('alert.category', -1),
    # ('dest_ip', -1),
    ('src_ip', -1),
    ('dest_port', -1),
    # ('src_port', -1),
    ('timestamp', -1),
    # ('host', 5),
    # ('time_delta', -2)
]


FEATURES = [feat[0] for feat in FEATURE_EMBEDDING_SIZE]

WIN_SIZE = 24
STRIDE = 6
H_DIM = 128
LATENT_CODE = 4

RESULTS_PATH = os.path.join(os.path.split(__file__)[0], 'results')


def one_hot(arr, arrlen):
    '''
    Creates a one hot encoded representation of the input array
    :param arr: Vector to one hot encode
    :return one_hot_arr: One hot encoded array
    '''
    one_hot_arr = np.zeros((arr.size, arrlen))
    one_hot_arr[np.arange(arr.size), arr] = 1
    return one_hot_arr

def to_tuple(a):
    '''
    Converts nested list, arrays, etc into tuples
    '''
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a

def save_json(fname:str, data:list):
    '''
    Save the data to a *.json file with name fname
    :param fname: Filename to save data to
    :param data: Data, likely generated, to save to file
    '''
    alert_path = os.path.join(RESULTS_PATH, 'generated', 'alerts')
    if not os.path.exists(alert_path):
        os.makedirs(alert_path)
    data = sort_by_timestamps(data)
    with open(os.path.join(alert_path,fname+'.json'), 'w') as f:
        for alert in data:
            alert_dict = {}
            alert_dict['result'] = {}
            for i, feature in enumerate(alert):
                alert_dict['result'][FEATURE_EMBEDDING_SIZE[i][0]] = feature
            f.write(json.dumps(alert_dict)+"\n")

def save_conditional_probability_table(fname:str, data:dict, weights:dict):
    '''
    Saves the the csv file given a dictionary which is formatted into a table.
    :param fname: Filename to save data to
    :param data: Conditional probabilities to save to csv
    :param weights: Weights to conditional probability input
    :notes: Should be feeding in raw probabilities in nested dictionaries and zero pad to make rectangular
    '''
    x_labels = []
    y_labels = []
    occurences = []
    probs = []
    # Find matrix dimensions
    for conditioning_values in data:
        x_labels.append(conditioning_values)
        y_labels.extend(data[conditioning_values])
        if conditioning_values != "overall_total":
            occurences.append(weights[conditioning_values])
            probs.append(weights[conditioning_values]/weights["overall_total"])
    y_labels = set(y_labels)
    y_labels = list(y_labels)
    y_labels.insert(0, 'Num Occurences')
    y_labels.insert(0, 'Probability')
    y_labels.insert(0, 'Combinations')
    # Subtract out three to counteract the additional columns
    arr = np.zeros((len(data), len(y_labels)-3))
    # Go lower than the minimum by one to make a column for labels
    for conditioning_values in data:
        for target_value in data[conditioning_values]:
            # Could be more efficient without index operator but can't ensure that order is always maintained
            # Subtract out one to counteract the additional of a title for the input column
            arr[x_labels.index(conditioning_values), y_labels.index(target_value)-3] = data[conditioning_values][target_value]
    with open(os.path.join(RESULTS_PATH, fname), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(y_labels)
        for i, row in enumerate(arr):
            label = str(x_labels[i]).replace(',', '+')
            row = row.tolist()
            csvfile.write(label+','+str(probs[i])+','+str(occurences[i])+','+str(row)[1:-1]+'\n')
            # csvfile.write(label)
            # writer.writerow(row)

def sort_by_timestamps(data:list):
    '''
    Takes the data output from the GAN and sorts it based off timestamp index
    :param data: Multidimensional numpy array to sort by
    :return data: Data sorted by timestamp
    '''
    time_index = [item[0] for item in FEATURE_EMBEDDING_SIZE].index('timestamp')
    times = []
    #TODO: In order to sort by timestamp we'll need a reference to the timestamps starting each bin
    for alert in data:
        times.append(float(time.mktime(time.strptime(alert[time_index], '%Y-%m-%dT%H:%M:%S.%f+0000'))))
    sort = [times.index(s) for s in sorted(times)]
    sorted_data = []
    for s in sort:
        sorted_data.append(data[s])
    return sorted_data

def get_feature_combinations(features:list):
    '''
    Get's all feature overlap regions for feature pairs
    :param features: Data as numpy array where each row is a list of alert features
    :return combination_hierarchy: Dictionary of all unique feature combinations with varying overlap
    :notes: (e.g.)
        Given an Alert a = {0,1,2,...,5} and overlap size of 2
        combination_indices = [(1,2), (1,3), (1,4), ...] for all unique combinations
        and the feature hierarchy would store it as fh[overlap_size][comb_index] = feature_combos
    '''
    overlap_size = len(features[0])
    num_features = [i for i in range(overlap_size)]
    combination_hierarchy = {i+1:{} for i in range(overlap_size)}
    while overlap_size > 0:
        feature_combos = list(combinations(num_features, overlap_size))
        for comb in feature_combos:
            combination_hierarchy[overlap_size][comb] = features[:,comb]
        overlap_size -= 1
    return combination_hierarchy

def get_feature_permutations(features:list):
    '''
    Get's all feature overlap regions for feature pairs
    :param features: Data as numpy array where each row is a list of alert features
    :return combination_hierarchy: Dictionary of all unique feature permutations with varying overlap
    :notes: (e.g.)
        Given an Alert a = {0,1,2,...,5} and overlap size of 2
        combination_indices = [(1,2), (1,3), (1,4), ...] for all unique permutations
        and the feature hierarchy would store it as fh[overlap_size][comb_index] = feature_combos
    '''
    overlap_size = len(features[0])
    num_features = [i for i in range(overlap_size)]
    combination_hierarchy = {i+1:{} for i in range(overlap_size)}
    while overlap_size > 0:
        feature_combos = list(permutations(num_features, overlap_size))
        for comb in feature_combos:
            combination_hierarchy[overlap_size][comb] = features[:,comb]
        overlap_size -= 1
    return combination_hierarchy

def parse_port_categories():
    '''
    Function from Eric to parse ports into a set of categories defined by IANA
    '''
    port_categories = dict()
    allp = []
    count = 0
    with open("port_categories.txt","r") as file:
        for line in file.readlines():
            ports=[]
            data = line.split('[')
            service = data[0].replace('\t','')
            data[1] = data[1].replace(' ','')
            data[1] = data[1].replace('[','')
            data[1] = data[1].replace(']','')
            data[1] = data[1].replace(',',' ')
            data[1] = data[1].replace('\'','')
            data[1] = data[1].replace('\n','')
            data[1] = data[1].split()
            for s1 in data[1]:
                if "-" in s1:
                    v = s1.split('-')
                    for n in range(int(v[0]),int(v[1])+1):
                        if (n in allp) and ('Unassigned' in service):
                            continue
                        ports.append(n)
                        allp.append(n)
                else:
                    ports.append(int(s1))
                    allp.append(int(s1))
            count += len(ports)
            port_categories[service] = ports
    return port_categories

def simplify_ports(ports):
    '''
    Simplifies the port categories based off common collections of port uses
    :param ports: Port IDs to group
    :returns: Remaped Port Usages
    :notes: (e.g) https may commonly use ports (8008, 8080, 80)
    '''
    port_categories = parse_port_categories()
    for i,port in enumerate(ports):
        port = int(port)
        for category,collection in port_categories.items():
            if port in collection:
                ports[i] = category
                break
    return ports
