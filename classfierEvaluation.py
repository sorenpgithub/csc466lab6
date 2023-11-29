from functions import *
#classifierEvaluation <pred.csv> <gt.csv> [-s (silent, only outputs total metrics)] [-w (write output)]

"""
for each author, need num correct predicted (TP), false positives, 
and (num?) documents written by author not attributed to them (--> TN?)
need precision, recall and f1 measure

also need overall values of above
"""
def generate_mets(cm):

    #From https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    #switched the axis though, because the stackoverflow case was based on that predictions were the colomns
    #and our confusion matrix has prediction as row
    FP = cm.sum(axis=1) - np.diag(cm)  
    FN = cm.sum(axis=0) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.values.sum() - (FP + FN + TP)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    mets = []
    #for author in len(cm):
    pass
    #return TP, FP, TN, precision, recall, f1
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
