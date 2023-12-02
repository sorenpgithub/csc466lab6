from functions import *
from nltk.stem.snowball import SnowballStemmer

#python3 textvectorizer <root_dir> [stopwords.txt] [-st]
def vectorizer(dirpath, size, stem, stop, stop_set):
    stemmer = SnowballStemmer('english')
    vocab = {} #going to be of the form {word:[f1, .., fn]} where each f_i is how many times that word appears in document i
    #therefore the wordset is just all the keys of the document
    size = 2500

    ground_tru = [()] * size

    i = 0 #current document
    print("started")
    for root, dirs, files in os.walk(dirpath):

        for author in dirs: #name of dir, in this case the author

            #irrelevant
            author_path = os.path.join(root, author)
            txtfiles = os.listdir(author_path) #list of filenames
            

            for file in txtfiles:

                file_path = os.path.join(author_path, file)
                file_size = os.path.getsize(file_path) #returns in bytes
                ground_tru[i] = (file, author, file_size)#assigns ground truth value
                #ensures vectorizer is in same order of ground truth 
                with open(file_path, 'r') as curr:
                    text = curr.read()
                    words = split_txt(text) #will be list of words, no punctuation except apostrophes

                    for word in words:
                        word = word.lower()
                        
                        #stop comes before stemming
                        if stop and word in stop_set: #checks if stopping is true and then if in stop set
                            continue #skips word
                       
                        if stem:
                            word = stemmer.stem(word) #modifies word essentially changes to short version

                        if word not in vocab:
                            vocab[word] = np.zeros(size) #if this word is new, adds to dictionary with a numpy array of length size = num of ducments

                        vocab[word][i] += 1 #adds one to count of rows, since each i represents doc

                    i += 1 #next document so adds 1

    return vocab, ground_tru
            

"""
Learned how to use argparse, pretty fancy shmancy
"""
def parse_cmd():
    parser = argparse.ArgumentParser(
        prog = "textVectorizer.py",
        description = "vectorizes file based on term frequency and exports to vector")
    parser.add_argument("root", help="training root directory of authors")
    parser.add_argument("-sfile", "--stopfile", nargs="?", help="textfile of stopwords", default=None)
    parser.add_argument("-st", "--stem", help="use snoball stemming", action="store_true")
    return parser.parse_args()


def main():
    size = 2500 #hardcoded for now due to use of np arrays
    args = parse_cmd()
    stop_set, stop = parse_stop(args.stopfile)

    vocab, ground_tru = vectorizer(args.root, size, args.stem, stop, stop_set) #dirpath, size, stem, stop, stop_set
    #groun

    write_vocab(vocab, args.stem, stop)
    write_truth("ground_truth.csv", ground_tru) #ground truth will be csv of form [filename, author, filesize (bytes)]

    print("done")

if __name__ == "__main__":
    main()

