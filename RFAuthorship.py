from functions import *

def rf():
    pass

def main():
    if len(sys.argv) == 1:
        print("RFAuthorship.py <vectors.csv> <gt.csv> <k> [-o (okapi metric)]")
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