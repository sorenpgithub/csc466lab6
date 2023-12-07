Othilia Norell, Soren Paetau, Nicolas Tan \\ onorell@calpoly.edu  / spaetau@calpoly.edu / nktan@calpoly.edu
​
***HOW TO RUN CODE***
- All programs can be given parameters with [python3 program.py -h] for detailed inputs and information
- Require directories named classifieroutputs, knnoutputs, rfoutputs, distmats in working directory

textVectorizer.py <root_dir> [-st/--stem] [-sfile/--stopfile]->[stopwords.txt]:
Returns a csv file of ground truth and vectorized documents. Each column of vectors[_stem][_stop].csv represents a unique word and each row represents a document. The document each index can be accesseed in ground_truth.csv.

knnAuthorship.py <vectors.csv> <ground_truth.csv> <k> [-o/--okapi] [-d/--distance]->[dist_mat.csv]
Returns a csv file of predicted outputs in directory named knnoutputs. Predictions are in author name and indexes correlate to prediction on txt document in ground_truth.csv. Since distance matrix is majority of calculation, distmatrix is automatically outputted and can then be used as input.

RFAuthorship.py <vectors.csv> <ground_truth.csv> <numtree> <numatt> <numdata> [threshold]
Returns a csv file of predicted outputs in directory named rfoutputs.

classifierEvaluation.py <predictions.csv> <ground_truth.csv> <numtree> <numatt> <numdata> [threshold]
Returns a csv file of predicted outputs in directory named rfoutputs.