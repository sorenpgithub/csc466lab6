from functions import *
#classifierEvaluation <pred.csv> <gt.csv> [-s (silent, only outputs total metrics)] [-w (write output)]

"""
for each author, need num correct predicted, false positives, and documents written by author not attributed to them, need precision, recall and f1 measure

also need overall values of above
"""
def generate_mets(cm):
    pass


"""
Mets [AUTHOR1(tp, fp, misses, precision, recall, f1), AUTHOR2(...)..., TOTAL(tp,..f1)]
if silent only print last one
"""
def output_mets(mets, silent):
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