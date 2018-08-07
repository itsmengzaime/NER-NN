import os

from .utils import logging_file
from .data_utils import processing_trimmed_glove_vector,load_dict, get_processing_word

class Config():
    def __init__(self, load=True):
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        self.log = logging_file(self.path_log)
        if load:
            self.load()

    def load(self):
        self.vocab_words = load_dict(self.f_words)
        self.vocab_tags  = load_dict(self.f_tags)
        self.vocab_chars = load_dict(self.f_chars)
        self.num_word = len(self.vocab_words)
        self.num_char = len(self.vocab_chars)
        self.num_tag = len(self.vocab_tags)

        self.processing_word = get_processing_word(self.vocab_words, self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,lowercase=False, allow_unk=False)
        self.embeddings = (processing_trimmed_glove_vector(self.f_trimmed) if self.use_pretrained else None)


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # glove files
    f_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    f_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    f_dev = "data/eng.testb.iob"#"data/pub.dev"
    f_test = "data/eng.testa.iob"#"data/pub.test"
    f_train = "data/eng.train.iob" #"data/pub.train"

    max_iter = None # if not None, max number of examples in Dataset
    f_words = "data/words.txt"
    f_tags = "data/tags.txt"
    f_chars = "data/chars.txt"

    train_embeddings = False
    num_epochs = 100
    drop_out = 0.3
    batch_size = 25
    method = "adam"
    lr_rate = 0.01
    lr_decay = 0.9
    clip = -1
    num_epoch_no_imprv = 10
    fil_size = [2,3,4,5]
    num_filter = 100
    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if char embedding, training is 3.5x slower on CPU
    use_cnn = False
