import numpy as np
from keras.utils import to_categorical

feature_length = 84        ## if from fc2 of nn use this
# feature_length = 6        ## if from logits of nn use this
# feature_length = 120      ## if from fc1 of nn use this

def load_data_by_num(seq_select = 1):
    if seq_select==1:
        # dir_seq = './data/sequence_gen/rnn_sequence_cons_1.npy'
        dir_seq = './data/sequence_gen/rnn_sequence_fc2_1.npy'
    else:
        # dir_seq = './data/sequence_gen/rnn_sequence_cons_2.npy'
        dir_seq = './data/sequence_gen/rnn_sequence_fc2_2.npy'

    seq = np.load(dir_seq)
    seq = seq.item()
    max_length = feature_length*100  # len of encoded data per seq * max frames per Sequence

    X, y = load_data_sequencial(seq, max_length)
    return X, y

def explore():
    sequence_length = []
    number_of_sequence = 0
    for key, _ in seq.items():
        for key_in, val_in in seq[key].items():
            sequence_length.append(val_in.shape[0])
            number_of_sequence+=1
    print("Total number of sequences : ", number_of_sequence)
    print("max frames = {}, min frames = {}".format(max(sequence_length),min(sequence_length)))

def stackAndPad(array, max_length):
    temp = np.concatenate(array, axis=0)
    pad_array = np.zeros((max_length - temp.shape[0]))
    padded = np.hstack((pad_array, temp))
    assert padded.shape[0]==max_length, "Sequence overflow"
    return padded

def load_data_sequencial(seq, max_length):
    X, y = [], []
    for key, _ in seq.items():
        for key_in, val_in in seq[key].items():
            X.append(stackAndPad(val_in, max_length))
            y.append(int(key.split('_')[1]))

    X, y = np.asarray(X), np.asarray(y)
    y_one_hot = to_categorical(y, num_classes=6)
    return X, y_one_hot

def main():
    explore()

if __name__ == '__main__':
    main()
