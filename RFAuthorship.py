from functions import *

#python3 RFAuthorship.py <vectors.csv> <ground_truth.csv> <numTrees> <numAttr> <numPts> <threshold> ????

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

def rf(vectors, gt, numTrees, numAttr, numPts, thres, dist_name, dist_in):
  pass


#The output of each program shall be an authorship label predicted for each of the documents in the Reuters50-50 dataset.
def main():
    if len(sys.argv) == 1:
        print("RFAuthorship.py <vectors.csv> <gt.csv> <numTrees> <numAttr> <numPts> <threshold> ")
        quit()
    cos = True
    if "-o" in sys.argv:
        cos = False
    vec_in = sys.argv[1]
    gt_in = sys.argv[2]
    k = sys.argv[3]
    vec = parse_vec(vec_in)
    gt = parse_gt(gt_in)
    preds = rf(vec, gt, k, cos)
    write_output(preds, "rf")    

if __name__ == "__main__":
    main()
