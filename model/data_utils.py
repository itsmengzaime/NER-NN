import numpy as np
import os

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = "ERROR: Unable to locate file {}".format(filename)
        super(MyIOError, self).__init__(message)

class PreProcessData(object):
    def __init__(self, filename, processing_word=None, processing_tag=None, m_iteration=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.m_iteration = m_iteration
        self.length = None

    def __iter__(self):
        num_iter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        num_iter += 1
                        if self.m_iteration is not None and niter > self.m_iteration:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    w, t = ls[0],ls[-1]
                    if self.processing_word is not None:
                        w = self.processing_word(w)
                    if self.processing_tag is not None:
                        t = self.processing_tag(t)
                    words += [w]
                    tags += [t]

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length


def processing_vocab(data):
    print("Building vocabulary ...")
    w_vocab = set()
    t_vocab = set()
    for dataset in data:
        for word, tag in dataset:
            w_vocab.update(word)
            t_vocab.update(tag)
    print("Done. {} tokens".format(len(w_vocab)))
    return w_vocab, t_vocab


def processing_char_vocab(data):
    c_vocab = set()
    for word, _ in data:
        for w in word:
            c_vocab.update(w)
    return c_vocab


def glove_vocab(filename):
    print("Building GloVe vocabulary ...")
    glove_vocab = set()
    with open(filename) as f:
        for line in f:
            w = line.strip().split(' ')[0]
            glove_vocab.add(w)
    print("Done. {} tokens".format(len(glove_vocab)))
    return glove_vocab


def writing_vocab(vocab, filename):
    print("Writing output file ...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("Done. {} tokens".format(len(vocab)))


def load_dict(filename):
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d

def exp_trimmed_glove_vector(vocab, f_glove, f_trimmed, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(f_glove) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(f_trimmed, embeddings=embeddings)


def processing_trimmed_glove_vector(filename):
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,lowercase=False, chars=False, allow_unk=True):
    def f(word):
        if vocab_chars is not None and chars == True:
            c_id = []
            for char in word:
                if char in vocab_chars:
                    c_id += [vocab_chars[char]]
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM
            
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("UnKnown Key. Please re-check the tag")

        if vocab_chars is not None and chars == True:
            return c_id, word
        else:
            return word
    return f


def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,[pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length


def minibatches(data, b_size):
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == b_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(token, idx_to_tag):
    tag_name = idx_to_tag[token]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    default = tags[NONE]
    tag_id = {idx: tag for tag, idx in tags.items()}
    chunk_list = []
    chunk_type, chunk_start = None, None
    for i, token in enumerate(seq):
        if token == default and chunk_type is not None:
            chunk = (chunk_type, chunk_start, i)
            chunk_list.append(chunk)
            chunk_type, chunk_start = None, None
        elif token != default:
            token_chunk_class, token_chunk_type = get_chunk_type(token, tag_id)
            if chunk_type is None:
                chunk_type, chunk_start = token_chunk_type, i
            elif token_chunk_type != chunk_type or token_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunk_list.append(chunk)
                chunk_type, chunk_start = token_chunk_type, i
        else:
            pass

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunk_list.append(chunk)

    return chunk_list

