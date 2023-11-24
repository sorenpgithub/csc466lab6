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


def calc_dist_mat(mat, cos, gt):
    rows = mat.shape[0]
    dist = np.zeros((rows,rows))
    if cos: #cosine symmalirty is symmetric
        weight_vectors(mat) #need to apply tf idf if using cosine similarity
        #should modify in place
        for i in range(rows):
            for j in range(i+1,rows):
                dist[i,j] = cosine_sim(mat[i], mat[j])
        dist = dist + dist.T

    else: #okapi is not symmetric so need to calculate everything
        #each column [i] will represent okapi similarity of all other documnets to document i. essentially row i is the query
        numdocs = mat.shape[0]
        numwords = mat.shape[1]
        dl = list(gt["size"])
        avdl = gt["size"].mean()
        k1 = 1.5
        b = 0.75
        k2 = 300
        docf = calc_df(mat) #numpy array of document frequencies for word i

        for q in range(rows):
            for j in range(rows):
                sum = 0
                for i in range(numwords): #mat.shape is number of columns in matrix, aka number of words
                    docf[i] 
                    x = np.log((numdocs - docf[i] + 0.5) / (docf[i] + 0.5))
                    y =  ((k1 + 1) * mat[j,i])  /  (k1 * (1 - b + (b * (dl[j]/avdl))))  #this is so hard to read but just check documentation
                    z = ((k2 + 1)*mat[q,i])/(k2 + mat[q,i])
                    #note that mat[j,i] is frequency of word i on document j, similar for q
                    sum += x * y * z
                
                dist[q,j] = sum

    return dist

"""
just calculate frequency of each word
"""
def calc_df(mat):
    numcols = mat.shape[1]
    temp = np.zeros(numcols)

    for i in range(numcols):
        temp[i] = np.count_nonzero(mat[:,i])  #counts nonzero values in column i
    
    return temp

"""
Implement TF-IDF
"""
def weight_vectors(vectors):
    n = vectors.shape[0] #number of documents
    numwords = vectors.shape[1]
    docf = calc_df(vectors)
    for j in range(n):
        norm_fac = np.max(vectors[j]) #should be row
        for i in range(numwords):
            vectors[j,i] = (vectors[j,i] / norm_fac) * np.log2(n / docf[i])



def split_txt(input_string):
    #fancy regex, \b are boundries, [a-zA-z] just means word must have a letter,
    # second chunk after + means take any letters after apostrophes,
    #drops all other punc, should be quick
    words = re.findall(r"\b(?:[a-zA-Z]+(?:'[a-zA-Z]+)?)\b", input_string) #stolen from stackoverflow
    return words



def write_truth(path, list_in): #assumes path is path name, list_in is nested list of ground truths
    filename, author, size = zip(*list_in) #essentially a reverse zip
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
        