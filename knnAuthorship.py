from functions import *

#python3 knnAuthorship.py <vectors.csv> <ground_truth.csv> <k> [-o]

def knn(vectors, gt, k, cos):
    dist_mat = calc_dist_mat(vectors, cos)
    preds = []
    for row in dist_mat:
        neighbors = np.argsort(row)[1:k+1] #indices of k nearest neighbors
        pred = get_pred(neighbors, gt) #name of author
        pred.append(preds)

    preds = [gt["authors"].iloc[index] for index in preds] #maps indices to author name
     #dataframe, should be 50x50, predicted value is row while column is actual
    return preds





def main():
    if len(sys.argv) == 1:
        print("knnAuthorship.py <vectors.csv> <k> [-o (okapi metric)]")
        quit()
    cos = True
    if "-o" in sys.argv:
        cos = False
    vec_in = sys.argv[1]
    gt_in = sys.argv[2]
    k = sys.argv[3]
    vec = parse_vec(vec_in)
    gt = parse_gt(gt_in)
    preds = knn(vec, gt, k, cos)
    write_output(preds, "knn")
    


if __name__ == "__main__":
    main()