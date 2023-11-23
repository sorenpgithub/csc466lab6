import pandas as pd
import numpy as np
from numpy import linalg as LA
import sys
import random
import csv
import os
import re
import datetime
from sklearn.metrics import confusion_matrix


def get_pred(neighbors, gt):
    return gt.iloc[neighbors]["author"].mode() #narrows down to only given documents, then looks at author column and returns parity
    #assumes matrix is in same order 


def generate_cm(preds, gt):
    authors = gt["author"].unique()
    pred_c = pd.Categorical(preds, categories=authors)
    actu_c = pd.Categorical(gt["author"], categories=authors)
    cm = pd.crosstab(pred_c, actu_c, dropna=False) #dataframe, row represents predicted
    return cm

def cosine_sim(v1,v2):
    return np.dot(v1, v2)/(LA.norm(v1)*LA.norm(v2))



def okapi_sim(v1,v2):
    pass

def calc_dist_mat(mat, met):
    rows = mat.shape[0]
    dist = np.zeros((rows,rows))
    for i in range(rows):
        for j in range(i+1,rows):
            dist[i,j] = calc_dist(mat[i], mat[j], met)
    dist = dist + dist.T
    return dist


def calc_dist(v1, v2, cos):
    if cos:
        return cosine_sim(v1,v2)
    else:
        return okapi_sim(v1,v2)


def split_txt(input_string):
    #fancy regex, \b are boundries, [a-zA-z] just means word must have a letter,
    # second chunk after + means take any letters after apostrophes,
    #drops all other punc, should be quick
    words = re.findall(r"\b(?:[a-zA-Z]+(?:'[a-zA-Z]+)?)\b", input_string) #stolen from stackoverflow
    return words



def write_truth(path, list_in): #assumes path is path name, list_in is nested list of ground truths
    filename, author, size = zip(*list_in)
    df = pd.DataFrame({"filename":filename, "author":author, "size":size})
    df.to_csv(path, index=False)


"""
write vocab files
"""
def write_vocab(vocab, stem, stop): #going to be of the form {word:[f1, .., fn]} where each f_i is how many times that word appears in document i
    df = pd.DataFrame(vocab)
    out_name = "vectors"
    if stem:
        out_name += "_stem"
    if stop:
        out_name += "_stop"

    out_name += ".csv"

    df.to_csv(out_name, index=False) #will overwrite if already exists


"""
writes output csv to output directory, name is current time
should work for series
"""
def write_output(df, program_name):
    curr_time = datetime.now()
    name = curr_time.strftime("%d_%H-%M-%S")
    path = f"{program_name}outputs/{name}"
    df.to_csv(path, index=True)


def parse_stop(path):
    with open(path, 'r') as file:
        text = file.read()
        word_set = set(text.split())
    return word_set

"""
want vec to be numpy array such that each row is document and each column is word
"""
def parse_vec(path):
    pass


"""

"""
def parse_gt(path):
    pass

"prediction csv will be of the form such that each row represents the prediction in order of the gt document"
def parse_pred(path):
    pass


# Check if the file exists in cwd
def file_exists(name):
    curr_dir = os.getcwd()
    path = os.path.join(curr_dir, name)

    return os.path.exists(path)
        