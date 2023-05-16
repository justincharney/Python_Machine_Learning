'''
SSpred.py
===================================================
Python script that makes a alpha-helix secondary structure prediction for a sequence file
===================================================
This script contains 3 functions:
    readInput(): A function that reads in input from a sequence file
    SS_random_prediction(): A function that makes a random prediction weighted by the fraction of alpha helices in the training data
    writeOutput(): A function that write a prediction to an output file
'''

import random
import ast
import numpy as np

inputFile = "../input_file/infile.txt"
parameters = "../src/parameters.txt"
predictionFile = "../output_file/outfile.txt"


def readInput(inputFile):
    '''
        Read the data in a FASTA format file, parse it into into a python dictionnary
        Args:
            inputFile (str): path to the input file
        Returns:
            training_data (dict): dictionary with format {name (str):sequence (str)}
    '''
    inputData = {}
    with open(inputFile, 'r') as f:
        while True:
            name = f.readline()
            seq = f.readline()
            if not seq: break
            inputData.update({name.rstrip(): seq.rstrip()})

    return inputData


def flatten(l: list):
    """Takes in a list of lists and flattens it"""
    return [item for sublist in l for item in sublist]


def one_hot(np_arr, categories=2):
    cat_seqs = []
    for idx, label in np.ndenumerate(np_arr):
        cat_seqs.append(np.zeros(categories))
        cat_seqs[-1][label] = 1.0
    return np.array(cat_seqs)


def encodeY(arr, dict):
    seqs2int = []
    for lst in arr:
        ints = []
        for i in list(lst):
            ints.append(dict[i.lower()])
        seqs2int.append(ints)
    return seqs2int


def encode(sequence_list: list, dct):
    """ Takes in a list of sequences and returns a
list of sequences after numerically encoding"""
    seqs2int = []
    # nested = any(isinstance(i, list) for i in sequence_list) # check if list is nested or not
    for seq in sequence_list:
        ints = []
        for i in list(seq):
            ints.append(list(dct.values()).index(i))
        seqs2int.append(ints)
    return seqs2int


class Preprocessor:
    def __init__(self):
        self.seq_dct = None
        self.label_dct = None
        self.flatX = None
        self.flatY = None
        self.flatYoh = None

    def transformX(self, seqs: list, N=3):
        """Takes in a sequence string or list of strings and creates input for NB model"""
        aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        dct = dict(enumerate(aas))  # create a dictionary of ammino acids
        self.seq_dct = dct
        enc = encode(seqs, dct)
        flat = flatten(enc)
        self.flatX = flat
        new_seq = flat.copy()
        # pad with N-1 dummy tokens (20)
        new_seq.extend([20] * (N - 1))
        toks = [new_seq[i:i + N] for i in range(len(new_seq) - N + 1)]
        return np.array(toks)

    def labeldict(self, y: list):
        set_labels = set([])
        for l in y:
            for seq in l:
                for char in list(seq):
                    set_labels.add(char.lower())
        label2idx = {l: i for i, l in enumerate(list(set_labels))}
        self.label_dct = label2idx
        return label2idx

    def transformY(self, labels: list):
        label_dict = self.labeldict(labels)
        enc_y = encodeY(labels, label_dict)
        flat_y = flatten(enc_y)
        self.flatY = flat_y
        oh_y = one_hot(np.array(flat_y))
        self.flatYoh = oh_y
        return oh_y


def make_prediction(aa, log_probs, prior_probs):
    # define static variables for the problem
    n_features = 3
    classes = np.array([0, 1])
    probs = np.zeros((1, 2))  # for 2 classes

    for i in range(n_features):  # num_features
        # get the category of the aa
        cat = aa[i]
        probs += log_probs[i][:, cat]

    # add log prior probability
    probs += prior_probs

    return classes[np.argmax(probs)]


def to_seq(seq, dct):
    s = ''
    for i in seq:
        s += dct[i]
    return s


def predict(inputData, parameters):
    label_dct = {'h': 0, '-': 1}

    # define an empty list
    arrays = []

    # open file and read the content in a list
    with open(parameters, 'r') as f:
        for line in f:
            # remove linebreak which is the last character of the string
            l = line[:-1]
            # add item to the list
            arrays.append(l)

    # get the log_likelihood ratios
    l1 = ast.literal_eval(arrays[0])
    log_probs = [np.array(l) for l in l1]

    # get the prior log odds
    prior_probs = np.array(ast.literal_eval(arrays[1]))



    # dictionary to decode predictions
    reverse_decoder_index = {value: key for key, value in label_dct.items()}

    my_predictions = {}
    for name in inputData:
        seq = inputData[name]
        x_i = Preprocessor().transformX(seq)  # properly encode the input
        x_i_preds = []
        for aa in x_i:
            x_i_preds.append(make_prediction(aa, log_probs, prior_probs))
        my_predictions.update({name: str(to_seq(x_i_preds, reverse_decoder_index).upper())})

    return my_predictions


def SS_random_prediction(inputData, parameters):
    '''
        Predict between alpha-helix (symbol: H) and non-alpha helix (symbol: -) for each amino acid in the input sequences
        The prediction is random but weighted by the overall fraction of alpha helices in the training data (stored in parameters)
        Args:
            inputData (dict): dictionary with format {name (str):sequence (str)}
            parameters (str): path to the file with with parameters obtained from training
        Returns:
            randomPredictions (dict): dictionary with format {name (str):ss_prediction (str)}
    '''

    with open(parameters, 'r') as f:
        fraction = float(next(f))

    randomPredictions = {}
    for name in inputData:
        seq = inputData[name]
        preds = ""

        for aa in seq:
            preds = preds + random.choices(["H", "-"], weights=[fraction, 1 - fraction])[0]

        randomPredictions.update({name: preds})

    return randomPredictions


def writeOutput(inputData, predictions, outputFile):
    '''
        Writes output file with the predictions in the correct format
        Args:
            inputData (dict): dictionary with format {name (str):sequence (str)}
            predictions (dict): dictionary with format {name (str):ss_prediction (str)}
            outputFile (str): path to the output file
    '''
    with open(outputFile, 'w') as f:
        for name in inputData:
            f.write(name + "\n")
            f.write(inputData[name] + "\n")
            f.write(predictions[name] + "\n")

    return


def main():
    inputData = readInput(inputFile)
    pre = Preprocessor().transformX(
        list(inputData.values()))  # intialize pre processor
    preds = predict(inputData, parameters)
    writeOutput(inputData, preds, predictionFile)


if __name__ == '__main__':
    main()
