from functions import *

#python3 knnAuthorship.py <vectors.csv> <ground_truth.csv> <k> [-o]

def knn(vectors, gt, k, okapi, dist_name, dist_in, multi):
    
    if dist_in is None:
        #calculates numpy matrix 2500x2500 of similarity metric. row represents query and column is d_j
        dist_mat = calc_dist_mat(vectors, okapi, gt, multi)
        write_dist_mat(dist_mat, dist_name) #will write the metric to the distmats dir with uniqueish name
    else:
        dist_mat = parse_dist(dist_in)
    preds = []
    for row in dist_mat:
        neighbors = np.argsort(row)[1:k+1] #indices of k nearest neighbors, excludes first element since that will be itself with k = 0
        pred = get_pred(neighbors, gt) #name of author, returns parity or most common author of given indices
        preds.append(pred) #adds to klist
    print("predictions completed")
     #maps indices to author name, will be list of length 2500 with 
     #dataframe, should be 50x50, predicted value is row while column is actual
    return preds


def parse_cmd():
    parser = argparse.ArgumentParser(
        prog = "knnAuthorship.py",
        description = "Predicts authors given a vectorized version of word stuff blah blah")
    parser.add_argument("vectors", help="vectors.csv, frequency csv file")
    parser.add_argument("gt", help="ground_truth.csv, ground truth file as defined in documentaiton")
    parser.add_argument("k", help="number of neighbors for K-nearest neighbors", type=int)
    parser.add_argument("-o", "--okapi", help="use okapi-BM25 similarity metric", action="store_true")
    parser.add_argument("-d", "--distance", nargs="?", help="pre-calculated distance matrix, insert filename path after", default=None)
    parser.add_argument("-m", "--multi",  help="multiprocessing UNSTABLE", action="store_true")

    return parser.parse_args()


def main():

    args = parse_cmd()

    dist_name = create_dist_path(args.vectors, args.okapi)#this will create a unique dist name based on the stemming, stop and okapi parameters
    gt = parse_gt(args.gt)
    vec = parse_vec(args.vectors) #numpy matrix where each row is a document and column is a word
    #ground truth file each row is document with filename, author and size in bytes. 
    preds = knn(vec, gt, args.k, args.okapi, dist_name, args.distance, args.multi) #returns list of predictions
    preds = pd.Series(preds) #convert to series so can be written to csv
    write_output(preds, "knn")
    


if __name__ == "__main__":
    main()