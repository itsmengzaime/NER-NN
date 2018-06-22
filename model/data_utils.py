import numpy as np
import os

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "0"

class MyIOError(Exception):
    def __init__(self, filename):
        message = "IOError: Unable to locate file{}".format(filename)
        super(MyIOError, self).__init__(message)
        
class PreprocessData(object):
    def __init__(self, filename, processing_word=None, processing_tag=None, m_iteration=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.m_iteration = m_iteration
        self.length = None
        
    
    def _pad_sequences(sequences, pad_tok, max_length):
        sequence_padded = []
        sequence_length = []
        
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + pad_tok*max(max_length, len(seq),0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]
            return sequence_padded, sequence_length
    def pad_sequences(sequences, pad_tok, nlevels=1):
        if nlevels == 1 :
            max_length = max(map(lambda x: len(x), sequences))
            sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
        elif nlevels == 2 :
            max_length_word = max([max(map(lambda x: len(x), seq)) 
                              for seq in sequences])
            sequence_padded = []
            sequence_length = []
            for seq in sequences:
                sp, s1 = _pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [s1]
            
            max_length_sentence = max(map(lambda x: len(x), sequences))
            
            sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word, max_length_sentence)
            sequence_length, _ = _pad_sequences(sequence_length, 0 , max_length_sentence)
            