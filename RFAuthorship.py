from functions import *

#HOW TO RUN
#python3 RFAuthorship.py vectors_stop.csv ground_truth.csv [numTrees] [numAttributes] [numPoints] [thresh]

"""
Dataset to build individual trees in the forest is sufficient to prevent overfit), so, it should simply build the forest
and make the predictions.

Tune:
  number of trees (numTrees) --> should be fairly large (there are many words in the document collection, the Random forest should give each a chance to participate in helping recognize authorship)
  number of attributes in a tree (numAttr) --> 10-20 might be a good target 
  number of data points used to build a tree (numPts) --> want to have representative writings from all authors in the data. (you could simply use 50% as your target, but can probably get away with a smaller percentage to speed things up).
Set: can set a standard threshold, and commit to using either information gain or information gain ratio

the Random forest should give each word a chance to participate in helping recognize authorship) --> should we check this in some way?
"""

"""
Randomizes data based on intuitive parameters
Uses pandas
"""
# def rand_data(mat, numAtt, numData): #D is pandas DF, rest is defined
#     nrows, ncols= mat.shape
#     # pickless
#     numdata = min(numData, nrows)
#     numatt = min(numAtt, ncols)
#     # Generate n random column indices
#     row_indices = np.random.choice(nrows, numdata, relace = False)
#     col_indices = np.random.choice(ncols, numatt, replace=False)
    
#     # Extract the selected columns
#     newmat = mat[row_indices][:, col_indices]
#     return newmat

def rand_data(D, class_var, numAtt, numData): #D is pandas DF, rest is defined
    df = D.sample(numData, replace = True)
    #print(df)
    cols = list(D.columns)
    cols.remove(class_var)
    newcols = random.sample(cols, numAtt)
    newcols.append(class_var)
    df = df[newcols]
    return df


"""
Helper function for edge cases
Constructs and returns a leaf node for a decision tree based on the most frequent class label
"""
def create_node(D):
    print("in create_node")
    aut = "author_name"
    temp = find_freqlab(D, aut) #should be whatever datatype c_i is
    r = {"leaf":{}}#create node with label of only class label STAR
    r["leaf"]["decision"] = temp[0]
    r["leaf"]["p"] = temp[1]
    print("r: ", r)
    return r #leaf with decision and prob

"""
Identifies the most frequent class label in the column specified by class_var
Returns both the label and its probability
"""
def find_freqlab(D, class_var): #assuming D is df
    print("in find_freqlab")
    values = D[class_var].value_counts(normalize = True)
    c = values.idxmax()
    pr = values[c]
    return (c,pr)


"""
Finds split with maximum gain for continuous variable A_i by iterating over all unique values
"""
def findBestSplit(A_i, D):
    aut = "author_name"
    vals = D[A_i].unique()
    gains = []
    p0 = enthropy(D, aut)
    for val in vals:
        ent = enthropy_val(val, A_i, D)
        gain = p0 - ent
        gains.append(gain)
    m = max(gains) #fidning the maximal info gain
    max_ind = gains.index(m) #finding the list index of the maximal info gain
    return vals[max_ind]

"""
Helpfer function in calculating enthropy of split at \alpha
"""
def enthropy_val(alpha, A_i, D):
    aut = "author_name"
    D_left = D[D[A_i] <= alpha]
    D_right = D[D[A_i] > alpha]
    x = D_left.shape[0] * enthropy(D_left, aut)
    y = D_right.shape[0] * enthropy(D_right, aut)
    z = D.shape[0]
    sum = (x/z) + (y/z)
    #print(sum)
    return sum


"""
Calculates the entropy of a dataset D based on a class variable class_var 
Entropy = -SUM(p*log2(p))
Returns 
"""
def enthropy(D, class_var):
    sum = 0
    bar = D.shape[0]
    for i in D[class_var].unique(): #SHOULD THIS BE FROM DOMS!!!
        D_i = D[D[class_var] == i]
        foo = D_i.shape[0] #|D_i|
        pr = foo/bar
        sum += pr * np.log2(pr)
    return -sum

"""
splitting
"""
def selectSplittingAttribute(A, D, threshold): #information gain
    aut = "author_name" #just for simplicity, needed due to redundancies in c45 alg implementation
    p0 = enthropy(D, aut) #\in (0,1) -sum
    gain = [0] * len(A)
    for i, A_i in enumerate(A): #i is index, A_i is string of col name
            x = findBestSplit(A_i, D)
            p_i = enthropy_val(x, A_i, D) #double check to make sure right entropy
        #print(p0, p_i)
            gain[i] = p0 - p_i 
    #print(gain)
    m = max(gain) #fidning the maximal info gain
    print("max: ",  m)
    if m > threshold:
        print("IN IF")
        max_ind = gain.index(m) #finding the list index of the maximal info gain
        return A[max_ind] #returns the attribute that had the maximal info gain
    else:
        ("IN ELSE")
        return None

"""
Implements the C4.5 algorithm for building a decision tree 
from a dataset D based on a list of attributes A and a threshold value for information gain
Returns the (sub)tree T rooted at the current node
NO CATEGORICAL VARIABLES!
"""
def c45(D, A, threshold, class_var, doms, current_depth=0, max_depth=None): #going to do pandas approach, assume D is df and A is list of col names
    #print("in C45")
    #print("A: ", A)
    #print("DCLASSVAR: ", D[class_var])
    #print(D[class_var].nunique())
    class_var = "author_name" #hardcodede
    if (max_depth is not None and current_depth == max_depth) or D[class_var].nunique() == 1 or (not A):
        print("HERE")
    #print("bug")
        T = create_node(D)

    #"Normal" case
    else:
        #print("THERE")
        A_g = selectSplittingAttribute(A, D, threshold) #string of column name
        if A_g is None:
            #print("A_g none")
            T = create_node(D)
        else:
            r = {"node": {"var":A_g, "edges":[]} } #dblcheck with psuedo code
            T = r
            # print("doms: ", doms)
            # print("ag: ", A_g)
            # print(doms[A_g])
            for v in doms[A_g]: #iterate over each unique value (Domain) of attribute (South, West..)
                D_v = D[D[A_g] == v] #dataframe with where attribute equals value
                
                if not D_v.empty: #true if D_v \neq \emptyset
                        #print(A_temp)
                    print("doms2: ", doms)
                    T_v = c45(D_v, A, threshold, class_var, doms, current_depth + 1, max_depth)
                        #temp = {"edge":{"value":v}}
                    #modify to contain edge value, look at lec06 example
                    temp = {"edge":{"value":v}}
                    if "node" in T_v:
                        temp["edge"]["node"] = T_v["node"]
                    elif "leaf" in T_v:
                        temp["edge"]["leaf"] = T_v["leaf"]
                    else:
                        print("something is broken")
                        
                else: #ghost node
                    #print("GHOST PATH")
                    label_info = find_freqlab(D, "author_name") #determines the most frequent class label and its proportion
                    ghost_node = {"leaf":{}} #initialize a leaf node
                    ghost_node["leaf"]["decision"] = label_info[0] #set the decision to the most frequent class label
                    ghost_node["leaf"]["p"] = label_info[1] #set the probability or proportion
                    temp = {"edge": {"value": v, "leaf": ghost_node["leaf"]}}
                
                
                r["node"]["edges"].append(temp)
    #print("T: ", T)
    return T



def parse_cmd():
    parser = argparse.ArgumentParser(
        prog = "RFAuthorship.py",
        description = "Predicts authors given a vectorized version of word stuff blah blah")
    parser.add_argument("vectors", help="vectors.csv, frequency csv file")
    parser.add_argument("gt", help="ground_truth.csv, ground truth file as defined in documentaiton")
    parser.add_argument("numtree", help="number of of trees in random forest", type=int)
    parser.add_argument("numatt", help="number of attributes in single decision tree", type=int)
    parser.add_argument("numdata", help="number of data points ", type=int)
    parser.add_argument("threshold", help="threshold in C45 Algorithm", default=0.2, type=float)
    return parser.parse_args()


def generate_preds(D, tree, class_var):
    print("in generate_preds")
    df_A = D.drop(class_var, axis = 1)#makes new df, not inplace
    pred = []
  
    if "leaf" in tree: #first element is a leaf
        dec = tree["leaf"]["decision"]
        pred = [dec] * D.shape[0] #returns list of that node
        return pred
  
    for index, row in df_A.iterrows(): #row is series object, val accessed like dict
        leaf = False
        curr_node = tree["node"] #{"var":123: "edges":[....]}

        while not leaf:
            A_i = curr_node["var"] 
            obs_val = row[A_i] #value of observation in variable, A_i = # of bedroom \implies obs_val = 3

            for edge in curr_node["edges"]: #list of edges | edg = {"edge":{"value"}}    
                curr_edge = edge["edge"]
            #print("current edge", curr_edge)
            #print("observed", A_i, obs_val)

                if curr_edge["value"] == obs_val: 
                    if "node" in curr_edge:
                        curr_node = curr_edge["node"] #updating new node

                    else: #must be a leaf
                        pred_val = curr_edge["leaf"]["decision"]
                
                        pred.append(pred_val)
                
                        leaf = True
                
                    break #doesnt iterate over redundant edges
        print(pred)
      #print("broken")
    return pred


def dom_dict(df):
    temp = {}
    for column in df.columns:
        temp[column] = df[column].unique().tolist()
    return temp


def rf(D, numtree, numatt, numdata, threshold):
    aut = "author_name" #just for simplicity, needed due to redundancies in c45 alg implementation
    doms = dom_dict(D)#define!
    print("doms_org: ", doms)
    pred_df = pd.DataFrame() 
    

    for n in range(numtree):
        #randomizes data
        train = rand_data(D, aut, numatt, numdata) #dataframe 
        test_cols = list(train.columns) #column names
        test_cols.remove(aut)
        #test cols is due to redundancy from 
        
        
        tree = c45(train, test_cols, threshold, aut, doms)
        print("tree: ", tree)
        predictions = generate_preds(D, tree, aut)
        y_pred = pd.Series(predictions)
        

        col_name = f"T{n}"
        pred_df[col_name] = y_pred  #makes new column of predictions
    

    pred_df['mode'] = pred_df.apply(find_mode, axis=1)  #makes new column which is the mode

    preds = pred_df["mode"]
    return preds


def main():
    args = parse_cmd()
    path = args.vectors
    vec = parse_vec(path, mat = False) #want vec to be a dataframe
    gt = parse_gt(args.gt)
    print(vec.head())
    D = pd.concat([vec, gt], axis=1) #merges dataframes by concat will be filename and size in here 
    D.drop(["size", "filename"], axis = 1, inplace = True)
    preds = rf(D, args.numtree, args.numatt, args.numdata, args.threshold)
    print("preds::", preds)
    #write_output(preds, "rf")    
    write_output_rf(preds, args.numtree, args.numatt, args.numdata)


if __name__ == "__main__":
    main()
