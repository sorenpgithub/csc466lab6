from functions import *
#classifierEvaluation <pred.csv> <gt.csv> [-s (silent, only outputs total metrics)] [-w (write output)]

"""
for each author, need num correct predicted (TP), false positives, 
and (num?) documents written by author not attributed to them (--> TN?)
need precision, recall and f1 measure

also need overall values of above
"""
def generate_mets(cm):
    #cm is a numpy array by default (since it is an sklearn confusion matrix)
    
    # cm: predictions as rows, actuals as columns
    TP = np.diag(cm) #np.array of length 50 of the diagonal values
    FP = cm.sum(axis=1) - TP #sum over the columns, returning length 50 array of all predictions, subtract by true posiitives
    FN = cm.sum(axis=0) - TP #similar to above but over rows, indicating all predicted values
    TN = cm.sum() - (FP + FN + TP) #TN 

    precision = np.where((TP + FP) > 0, TP / (TP + FP), 0) #if TP + FP > 0, otherwise 0
    recall = np.where((TP + FN) > 0, TP / (TP + FN), 0)
    f1 = np.where((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
    #accuracy = np.where((TP + TN + FP + FN) > 0, (TP + TN) / (TP + TN + FP + FN), 0)

    mets = []
    for i in range(len(TP)): #len(TP) = 50 for this lab
        author = []
        author.append(TP[i])
        author.append(FP[i])
        author.append(FN[i])
        author.append(precision[i])
        author.append(recall[i])
        author.append(f1[i])
        #author.append(accuracy[i])
        mets.append(author)

    
    tot_tp, tot_fp, tot_tn, tot_fn  = np.sum(TP), np.sum(FP), np.sum(TN), np.sum(FN)
    tot_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0
    tot_recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0
    tot_f1 = (2*tot_prec*tot_recall)/(tot_prec+tot_recall) if (tot_prec+tot_recall)>0 else 0
    #tot_accuracy = (tot_tp + tot_tn) / (tot_tp + tot_tn + tot_fp + tot_fn) if (tot_tp + tot_tn + tot_fp + tot_fn)>0 else 0


    tot = [tot_tp, tot_fp, 0, tot_prec, tot_recall, tot_f1]
    mets.append(tot)
    return mets
"""
Mets [AUTHOR1(tp, fp, misses, precision, recall, f1), AUTHOR2(...)..., TOTAL(tp,..f1)]
if silent only print last one
"""
def output_mets(mets, gt, silent):
    if not silent: 
        unique_authors = gt['author_name'].unique().tolist()
        for i in range(len(mets)-1):
            print(f'{unique_authors[i]} -- TP:{mets[i][0]}, FP:{mets[i][1]}, FN:{mets[i][2]}, Precision:{mets[i][3]:.3f}, Recall:{mets[i][4]:.3f}, F1-score:{mets[i][5]:.3f}')
        print("______________________________________________________________________________")
    print(f'Overall scores -- TP:{mets[-1][0]}, FP:{mets[-1][1]}, Accuracy:{mets[-1][0]/ (mets[-1][0] + mets[-1][1])}')
    #, Precision:{mets[-1][3]}, Recall:{mets[-1][4]}, F1-score:{mets[-1][5]}
    print("______________________________________________________________________________")



def parse_cmd():
    parser = argparse.ArgumentParser(
        prog = "classifierEvaluation.py",
        description = "Outputs a confusion matrix along with a variety of metric given a set of predictions and ground_truth file")
    parser.add_argument("predictions", help="vectors.csv, frequency csv file")
    parser.add_argument("gt", help="ground_truth.csv, ground truth file as defined in documentaiton")
    parser.add_argument("-s", "--silent", help="will only output total metrics", action="store_true")
    parser.add_argument("-w", "--write", help="will write output to classifieroutputs directory", action="store_true")
    return parser.parse_args()



def main():
    args = parse_cmd()
    pred = parse_pred(args.predictions) #pred is a dataframe such that each row represents the corresponding document prediction in gt with the author name
    gt = parse_gt(args.gt)

    cm = generate_cm(pred["prediction"], gt) #50 x50 confusion matrix, such that the row repersents predicted and colunn is actual, will be numpy array
    mets = generate_mets(cm)

    output_mets(mets, gt, args.silent)
    
    if args.write:
        authors = gt["author_name"].unique() #should be in order of appearence, which should be same as confusion matrix
        cmdf = pd.DataFrame(cm, columns=authors, index = authors)
        write_output(cmdf, "classifier", header_b=True)
    





if __name__ == "__main__":
    main()
