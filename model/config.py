import os
from datetime import datetime as dt

from .utils import logging_file
from .data_utils import *


class Config():
    def __init__(self, load=True):
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.log = logging_file(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        # 1. vocabulary
        self.vocab_words = load_vocab(self.f_words)
        self.vocab_tags  = load_vocab(self.f_tags)
        self.vocab_chars = load_vocab(self.f_chars)

        self.num_word     = len(self.vocab_words)
        self.num_char     = len(self.vocab_chars)
        self.num_tag      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words, self.vocab_chars, lowercase=False, chars=(self.use_chars is not None))
        self.processing_tag  = get_processing_word(self.vocab_tags,lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_word_vectors(self.f_trimmed) if 'glove' in self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"
    now_str = dt.now().strftime('%d%m%Y_%H%M%S')
    f_result = "results/result.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    use_pretrained = "glove"

    # pretrained files
    f_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)

    # trimmed embeddings (created from word2vec_f with build_data.py)
    f_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)

    # dataset 
    f_dev = "data/valid.txt"#"data/eng.testb.iob"
    f_test = "data/test.txt"#"data/eng.testa.iob"
    f_train = "data/train.txt"#"data/eng.train.iob"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    f_words = "data/words.txt"
    f_tags = "data/tags.txt"
    f_chars = "data/chars.txt"

    # training
    train_embeddings = False
    num_epochs       = 100
    drop_out          = 0.5
    batch_size       = 20
    method        = "adam"
    lr_rate               = 0.005
    lr_decay         = 1.0
    clip             = -1 
    num_epoch_no_imprv  = 10

    # model hyperparameters
    hidden_size_char = 10 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = "cnn" # blstm, cnn or None
    max_len_of_word = 20  # used only when use_chars = 'cnn'
