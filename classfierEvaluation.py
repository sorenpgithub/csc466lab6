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
    TP = np.diag(cm)
    FP = cm.sum(axis=1) - TP
    FN = cm.sum(axis=0) - TP
    TN = cm.values.sum() - (FP + FN + TP)

    precision = np.where((TP + FP) > 0, TP / (TP + FP), 0) #if TP + FP > 0, otherwise 0
    recall = np.where((TP + FN) > 0, TP / (TP + FN), 0)
    f1 = np.where((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
    #accuracy = np.where((TP + TN + FP + FN) > 0, (TP + TN) / (TP + TN + FP + FN), 0)

    mets = []
    for i in range(len(TP)): #len(TP) = 50 for this lab
        author = [TP[i], FP[i], TN[i], precision[i], recall[i], f1[i]]
        mets.append(author)
    
    tot_tp, tot_fp, tot_tn, tot_fn  = np.sum(TP), np.sum(FP), np.sum(TN), np.sum(FN)
    tot_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0
    tot_recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0
    tot_f1 = (2*tot_prec*tot_recall)/(tot_prec+tot_recall) if (tot_prec+tot_recall)>0 else 0

    tot = [tot_tp, tot_fp, tot_tn, tot_prec, tot_recall, tot_f1]
    mets.append(tot)
    return mets
"""
Mets [AUTHOR1(tp, fp, misses, precision, recall, f1), AUTHOR2(...)..., TOTAL(tp,..f1)]
if silent only print last one
"""
def output_mets(mets, silent):
    #print("mets:", mets)
    pass

def main():
    if len(sys.argv) == 1:
        print("classifierEvaluation.py <pred.csv> <gt.csv>")
        quit()
    silent = False
    write = False
    if "-s" in sys.argv:
        silent = True
    if "-w" in sys.argv:
        write = True

    pred_in = sys.argv[1]
    gt_in = sys.argv[2]
    pred = parse_pred(pred_in) #pred is a dataframe such that each row represents the corresponding document prediction in gt with the author name
    gt = parse_gt(gt_in)
    cm = generate_cm(pred["prediction"], gt) #50 x50 confusion matrix, such that the row repersents predicted and colunn is actual
    mets = generate_mets(cm)
    output_mets(mets, silent)
    if write:
        authors = gt["author"].unique() #should be in order of appearence, which should be same as confusion matrix
        cmdf = pd.DataFrame(cm, columns = authors)
        write_output(cmdf, "classifier")
    





if __name__ == "__main__":
    main()
