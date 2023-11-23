from functions import *
#classifierEvaluation <pred.csv> <gt.csv>


def main():
    if len(sys.argv) == 1:
        print("classifierEvaluation.py <pred.csv> <gt.csv>")
        quit()

    pred_in = sys.argv[1]
    gt_in = sys.argv[2]
    pred = parse_pred(pred_in)
    gt = parse_gt(gt_in)




if __name__ == "__main__":
    main()