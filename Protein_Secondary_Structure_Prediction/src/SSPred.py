# Python 3 script to edit for this project.
# Note 1: Do not change the name of this file
# Note 2: Do not change the location of this file within the BIEN410_FinalProject package
# Note 3: This file can only read in "../input_file/input_file.txt" and "parameters.txt" as input
# Note 4: This file should write output to "../output_file/outfile.txt"
# Note 5: See example of a working SSPred.py file in ../scr_example folder

import ast
import numpy as np

inputFile = "../input_file/infile.txt"
parameters = "../src/parameters.txt"
predictionFile = "../output_file/outfile.txt"

units = 4


def readInput(inputFile):
    '''
        Read the data in a FASTA format file, parse it into into a python dictionnary
        Args:
            inputFile (str): path to the input file
        Returns:
            inputData (dict): dictionary with format {name (str):sequence (str)}
    '''
    inputData = {}
    with open(inputFile, 'r') as f:
        while True:
            name = f.readline()
            seq = f.readline()
            if not seq: break
            inputData.update({name.rstrip(): seq.rstrip()})

    return inputData


# function to pad or truncate list to specific length
pad = lambda a, i: a[0:i] if len(a) > i else a + [0] * (i - len(a))


class PreprocessorLSTM:
    def __init__(self):
        self.ngrams = None
        self.ngram_dict = None

    def transform(self):
        """
        Numerically tokenize a list of n-grams using self.ngram_dict
        Returns:
            ints: list of sequence length with n-grams numerically tokenized
        """
        seqs2int = []
        ints = []
        for i in self.ngrams:
            if i.lower() in self.ngram_dict:  # checks if the token exists in the vocab
                ints.append(self.ngram_dict[i.lower()])
            else:
                ints.append(self.ngram_dict['unk'])  # unknown token
            seqs2int.append(ints)
        return ints

    def fit(self, seq, dct):
        """
        Pre-process an input protein sequence using n-grams, tokenization, and padding
        Args:
            seq (str): string of a protein sequence from inputData
            dct (dict): dictionary of n-gram numerical tokens with format {n-gram (str): token (int)}

        Returns:
            np_train_X (np.ndarray): numpy array of tokenized n-grams padded or truncated to a length of 1419

        """
        self.ngrams = np.array([seq[i:i + 3] for i in range(len(seq))], dtype=object)  # split into ngrams
        self.ngram_dict = dct
        train_X = self.transform()  # numerically tokenize
        padded = pad(train_X, 1419)
        np_train_X = np.array(padded)
        return np_train_X


def onehot_to_seq(oh_seq, index, seq_len):
    """
    Take a one-hot-encoded sequence prediction, get the categorical prediction, and decode it to a string of 'H' or '-'
    Args:
        oh_seq (np.ndarray): numpy array of shape (1419,3) corresponding to one-hot predictions
        index (dict): dictionary to decode the categorical predictions to 'H' or '-'
        seq_len (int): length of the input protein sequence for the prediction

    Returns:
        s (str): string of length 1419 consisting of 'H' or '-' as the predicted class for each amino acid

    """
    s = ''
    for o in oh_seq:
        i = np.argmax(o) # gets the index of the label with the highest probability
        if i != 0:
            s += index[i]
        else:
            s += 'H'  # if padding is predicted, simply predict H instead since s will be truncated
    if seq_len > 1419:
        s = s[:1419]
    else:
        s = s[:seq_len]
    return s



def sigmoid(x):
    """
    performs sigmoid activation for gates of LSTM
    Args:
        x (np.ndarray): numpy ndarray input from input, forget, or ouput gate

    Returns:
        (np.ndarray): numpy ndarray of shape (1,1,4)

    """
    return 1 / (1 + np.exp(-x))


def forward_LSTM(H_in, W, U, b, out_dim=4, batch_size=1):
    """
    Takes input from the ebedding layer and calculates the output of the LSTM layer with 4 units
    Args:
        H_in (np.ndarray): numpy ndarray of shape (1419,4) obtained after sequence is passed through the embedding layer
        W (np.ndarry): numpy ndarray of weights that multiply the sequence input to the LSTM
        U (np.ndarray): numpy ndarray of weights that multiply the previous hidden state input to the LSTM
        b (np.ndarray): numpy ndarray of biases for each of the gates and the candidate memory
        out_dim (int): number of LSTM units.
        batch_size (int): batch size used during prediction. Must be 1 here to avoid memory overflow

    Returns:

    """
    # These are all the weight arrays for LSTM
    W_i = W[:, :units]
    W_f = W[:, units: units * 2]
    W_c = W[:, units * 2: units * 3]
    W_o = W[:, units * 3:]

    U_i = U[:, :units]
    U_f = U[:, units: units * 2]
    U_c = U[:, units * 2: units * 3]
    U_o = U[:, units * 3:]

    b_i = b[:units]
    b_f = b[units: units * 2]
    b_c = b[units * 2: units * 3]
    b_o = b[units * 3:]

    L = H_in.shape[0]  # sequence length

    # calculate lstm output hidden states
    output = []
    h = np.zeros((1, batch_size, out_dim))
    c = np.zeros((1, batch_size, out_dim))
    for t in range(L):
        fg = sigmoid(H_in[t] @ W_f + h @ U_f + b_f)
        i = sigmoid(H_in[t] @ W_i + h @ U_i + b_i)
        n = np.tanh(H_in[t] @ W_c + h @ U_c + b_c)
        o = sigmoid(H_in[t] @ W_o + h @ U_o + b_o)

        c = i * n + fg * c
        h = o * np.tanh(c)
        output.append(h)

    return np.vstack(output)


def softmax(x):
    """
    calculates the softmax ouput for a np.ndarray of predictions after going through the dense layer
    Args:
        x (np.ndarray): numpy ndarray of shape (1419, 1, 3)

    Returns:
        (np.ndarray) numpy ndarray of shape (1419, 1, 3) where softmax has been applied to the last dimension
    """
    num = np.exp(x - np.max(x, axis=-1, keepdims=True))
    den = np.sum(num, axis=-1, keepdims=True)
    return num/den


def predict_LSTM(X: np.ndarray, W, U, b, w_e, w_, b_):
    """
    Performs a forward pass through the RNN model going through embedding layer, LSTM layer, and dense layer
    Args:
        X (np.ndarray): pre-processed input sequence of shape (1,1419)
        W (np.ndarray): input weight array for the LSTM layer of shape (4,16)
        U (np.ndarray): hidden weight array for the LSTM layer of shape (4, 16)
        b (np.ndarray): array of biases for the LSTM layer of shape (16,)
        w_e (np.ndarray): numpy array of embedding weights of shape (8414, 4)
        w_ (np.ndarray): numpy array of weights for the dense layer of shape (4,3)
        b_ (np.ndarray): numpy array of biases for the dense layer of shape (3,)

    Returns:
        outputs (list): list containing a one-hot-encoded np.ndarray of shape (1419,3) for the model predictions
    """
    outputs = []
    for s in range(X.shape[0]):
        # pass through embedding layer
        l0 = w_e[X[s], :]
        # pass through LSTM layer
        l1 = forward_LSTM(l0, W, U, b, 4)
        # pass through dense layer
        l2 = np.dot(l1, w_) + b_
        # perform softmax activation on last dimension
        l2 = softmax(l2)
        l2 = l2.reshape((1419, 3))
        outputs.append(l2)
    return outputs


def predict(inputData, parameters):
    """

    Args:
        inputData (dict): dictionary with format {name (str):sequence (str)}
        parameters (str): path to file containing model parameters

    Returns:
        my_predictions (dict): dictionary with format {name (str): prediction (str)}
    """
    label_dct = {'h': 2, '-': 1}  # dictionary used to encode labels

    arrays = []  # list to store the parameters
    with open(parameters, 'r') as f:  # open file and read content
        for line in f:
            l = line[:-1]
            arrays.append(l)

    W = np.array(ast.literal_eval(arrays[0]))
    U = np.array(ast.literal_eval(arrays[1]))
    b = np.array(ast.literal_eval(arrays[2]))
    w_e = np.array(ast.literal_eval(arrays[3]))
    w_ = np.array(ast.literal_eval(arrays[4]))
    b_ = np.array(ast.literal_eval(arrays[5]))
    ngram_dict = ast.literal_eval(arrays[6])

    # dictionary to decode predictions
    reverse_decoder_index = {value: key for key, value in label_dct.items()}

    my_predictions = {}
    # get the prediction for each sequence in inputData
    for name in inputData:
        seq = inputData[name]
        x_i = PreprocessorLSTM().fit(seq, ngram_dict)
        pred = predict_LSTM(x_i.reshape(1, 1419), W, U, b, w_e, w_, b_)  # get the prediction for the sequence
        my_predictions.update({name: str(onehot_to_seq(np.array(pred).squeeze(), reverse_decoder_index, len(seq)).upper())})

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
