# Python 3 script to edit for this project. 
# Note 1: Do not change the name of this file
# Note 2: Do not change the location of this file within the BIEN410_FinalProject package
# Note 3: This file can only read in "../input_file/input_file.txt" and "parameters.txt" as input
# Note 4: This file should write output to "../output_file/outfile.txt"
# Note 5: See example of a working SSPred.py file in ../scr_example folder

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


def encode(sequence_list: list, dct: dict):
    """ Takes in a list of sequences and returns a
list of sequences after numerically encoding"""
    seqs2int = []
    for seq in sequence_list:
        ints = []
        for i in list(seq):
            ints.append(list(dct.values()).index(i))
        seqs2int.append(ints)
    return seqs2int


class Preprocessor:
    def __init__(self):
        self.seq_dct = None
        self.flatX = None

    def transformX(self, seqs: list, N=3):
        """Takes in a sequence string or list of strings and performs preprocessing"""
        aas = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        dct = dict(enumerate(aas))  # create a dictionary of amino acids
        self.seq_dct = dct
        enc = encode(seqs, dct)  # numerically encode sequences using dictionary
        flat = flatten(enc)  # flatten the nested list
        self.flatX = flat
        new_seq = flat.copy()
        new_seq.extend([20] * (N - 1))  # pad with N-1 dummy tokens
        toks = [new_seq[i:i + N] for i in range(len(new_seq) - N + 1)]  # create 3-grams
        return np.array(toks)


def make_prediction(aa, log_probs, prior_probs):
    # define static variables for the problem
    n_features = 3  # 3 features since the input is 3-gram
    probs = np.zeros((1, 2))  # empty array to store probabilities for 2 classes 'H' or '-'

    for i in range(n_features):  # num_features
        # get aa category for each of the 3-gram input
        cat = aa[i]
        probs += log_probs[i][:, cat]

    # add log prior probability
    probs += prior_probs

    return np.argmax(probs)  # return the index of the highest probability


def to_seq(seq, dct):
    s = ''
    for i in seq:
        s += dct[i]
    return s


def predict(inputData, parameters):
    label_dct = {'h': 0, '-': 1}  # dictionary to decode the binary predictions

    arrays = []  # list to store the parameters
    with open(parameters, 'r') as f:  # open file and read content
        for line in f:
            l = line[:-1]
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
        x_i = Preprocessor().transformX(seq)  # properly encode the input sequence
        x_i_preds = []
        for aa in x_i:
            x_i_preds.append(make_prediction(aa, log_probs, prior_probs))
        my_predictions.update({name: str(to_seq(x_i_preds, reverse_decoder_index).upper())})

    return my_predictions

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
    preds = predict(inputData, parameters)
    writeOutput(inputData, preds, predictionFile)


if __name__ == '__main__':
    main()