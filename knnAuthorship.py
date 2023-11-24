from functions import *

#python3 knnAuthorship.py <vectors.csv> <ground_truth.csv> <k> [-o]

def knn(vectors, gt, k, okapi, dist_name, dist_in):
    if dist_in is None:
        dist_mat = calc_dist_mat(vectors, okapi, gt)
        write_dist_mat(dist_mat, dist_name)
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





def main():

    parser = argparse.ArgumentParser(
        prog = "knnAuthorship.py",
        description = "Predicts authors given a vectorized version of word stuff blah")
    parser.add_argument("vectors", help="vectors.csv, frequency csv file")
    parser.add_argument("gt", help="ground_truth.csv, ground truth file as defined in documentaiton")
    parser.add_argument("k", help="number of neighbors for K-nearest neighbors", type=int)
    parser.add_argument("-o", "--okapi", help="use okapi-BM25 similarity metric", action="store_true")
    parser.add_argument("-d", "--distance", nargs="?", help="pre-calculated distance matrix, insert filename path after", default=None)
    args = parser.parse_args()

    dist_name = create_dist_path(args.vectors, args.okapi)


    vec, words = parse_vec(args.vectors)
    gt = parse_gt(args.gt)
    preds = knn(vec, gt, args.k, args.okapi, dist_name, dist_in=args.distance)
    preds = pd.Series(preds)
    write_output(preds, "knn")
    


if __name__ == "__main__":
    main()