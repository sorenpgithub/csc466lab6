import pandas as pd
import numpy as np
from numpy import linalg as LA
import sys
import random
import csv
import os
import re
import argparse
from datetime import datetime
from sklearn.metrics import confusion_matrix


"returns mode of row"
def find_mode(row): #minor helper func
    return row.mode().iloc[0]

"""
returns parity author from given neighbors
"""
def get_pred(neighbors, gt):
    return gt.iloc[neighbors]["author"].mode()[0] #narrows down to only given documents, then looks at author column and returns parity
    #assumes matrix is in same order 


"""
generates confusion matrix, should be all 50x50 since actual has every label
"""
def generate_cm(preds, gt):
    authors = gt["author"].unique()
    pred_c = pd.Categorical(preds, categories=authors)
    actu_c = pd.Categorical(gt["author"], categories=authors)
    cm = pd.crosstab(pred_c, actu_c, dropna=False) #dataframe, row represents predicted
    return cm

"""
cosine similarity, should be symmetric, referenced from stackoverflow
"""
def cosine_sim(v1,v2):
    return np.dot(v1, v2)/(LA.norm(v1)*LA.norm(v2))

"""
Implement TF-IDF
"""
def weight_vectors(vectors):
    n = vectors.shape[0] #number of documents
    numwords = vectors.shape[1] #number of columns
    docf = calc_df(vectors) #list of document frequnecy, where docf[i] is number of docs word i appears
    for j in range(n):
        norm_fac = np.max(vectors[j]) #nromalization factor max value of most commonw ord
        for i in range(numwords):
            vectors[j,i] = (vectors[j,i] / norm_fac) * np.log2(n / docf[i])



"""
Dense function, returns 2500x2500 numpy matrix of distances
for okapi each row [i] will represent okapi similarity of all other documnets to document i. essentially row i is the query
"""
def calc_dist_mat(mat, okapi, gt):
    rows = mat.shape[0]
    dist = np.zeros((rows,rows))
    if not okapi: #cosine symmalirty is symmetric
        weight_vectors(mat) #need to apply tf idf if using cosine similarity
        #should modify in place
        print("vectors weighted")
        for i in range(rows):
            for j in range(i+1,rows):
                dist[i,j] = cosine_sim(mat[i], mat[j])
        dist = dist + dist.T

    else: #okapi is not symmetric so need to calculate everything
        numdocs = mat.shape[0]
        numwords = mat.shape[1]
        dl = list(gt["size"]) #length of document dj (in bytes)
        avdl = gt["size"].mean() #average length (in bytes) of a document in D
        k1 = 1.5 #normalization parameter for dj, 1.0 − 2.0
        b = 0.75 #normalization parameter for document length usually 0.75
        k2 = 300 #normalization parameter for query q 1 − 1000
        docf = calc_df(mat) #numpy array of document frequencies for word i

        for q in range(rows): #row number so each row is query
            for j in range(rows): #column number, each column is d_j
                if j != q: #want similarity to itself to be 0, for knn reasons
                    sum = 0
                    for i in range(numwords): #mat.shape is number of columns in matrix, aka number of words
                        docf[i] 
                        x = np.log((numdocs - docf[i] + 0.5) / (docf[i] + 0.5))
                        y =  ((k1 + 1) * mat[j,i])  /  (k1 * (1 - b + (b * (dl[j]/avdl))))  #this is so hard to read but just check documentation
                        z = ((k2 + 1)*mat[q,i])/(k2 + mat[q,i])
                        #note that mat[j,i] is frequency of word i on document j, similar for q
                        sum += x * y * z
                    
                    dist[q,j] = sum
    print("distance matrix returned")
    return dist #reason for this is kind of stupid, my knn code uses row iteration, for cos it's the exact same since it's already symmetric
    #for okapi i made it have the columns as the query

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
Fancy string parser used in vectorizer, accounts for apostrophes but removes numbers and other punctuation
returns list
"""
def split_txt(input_strings):
    #fancy regex, \b are boundries, [a-zA-z] just means word must have a letter,
    # second chunk after + means take any letters after apostrophes,
    #drops all other punc, should be quick
    words = re.findall(r"\b(?:[a-zA-Z]+(?:'[a-zA-Z]+)?)\b", input_strings) #stolen from stackoverflow
    return words



"""
writes truth csv with col names filename, author and size
"""
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
converts numpy matrix to dist path with comma seperated values, should be csv
"""
def write_dist_mat(dist_mat, name):
    np.savetxt(name, dist_mat, delimiter=',')


"""
writes output csv to output directory, name is current time
should work for series
"""
def write_output(df, program_name):
    curr_time = datetime.now()
    name = curr_time.strftime("%d_%H-%M-%S")
    path = f"{program_name}outputs/{name}"
    df.to_csv(path, index=True, header=False)


"""
parse in stop file, if no stopfile will return empty set with stop set to false
reasoning for stop boolean is to prevent unneeded checking stopset
finding items in set should be O(1) though.
"""
def parse_stop(path):
    if path is not None:
        with open(path, 'r') as file:
            text = file.read()
            word_set = set(text.split())
        return word_set, True
    else:
        return set(), False


"""
want vec to be numpy array such that each row is document and each column is word
"""
def parse_vec(path, mat = True):
    df = pd.read_csv(path)
    words = df.columns
    if mat:
        matr = df.to_numpy()
    print("vectors parsed")
    return matr, words


"""
overkill but nice to have
"""
def parse_gt(path):
    return pd.read_csv(path)

"prediction csv will be of the form such that each row represents the prediction in order of the gt document"
def parse_pred(path):
    pass

def parse_dist(path):
    return np.loadtxt(path, delimiter=',')


"""
function that takes the variables set in the vectors csv file and adds classifiers from distance matrix settings to create a unique name.
"""
def create_dist_path(vec_path, okapi):
    delimiters = r'[_|.]'
    words = re.split(delimiters, vec_path) #want to extract stem and stp
    factors = words[1:-1]
    path = "distance"
    if okapi:
        factors.append("okapi")
    for factor in factors:
        path += f"_{factor}"
    path += ".csv"
    return "distmats/" + path


# Check if the file exists in cwd
def file_exists(name):
    curr_dir = os.getcwd()
    path = os.path.join(curr_dir, name)

    return os.path.exists(path)
        
