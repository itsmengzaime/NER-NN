#config
import os 

from utils import logging_file
from data_utils import PreProcessData #processing_trimmed_glove_vector,load_dict, get_processing_word

class Config():
    def __init__(self, load=True):
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        self.logger = logging_file(self.path_log)
        if load:
            self.load()
    
    def load(self):
        #load vocab dictionary
        self.vocab_words = PreProcessData.load_dict(self.f_words)
        self.vocab_tags = PreProcessData.load_dict(self.f_tags)
        self.vocab_chars = PreProcessData.load_dict(self.f_chars)
        
        self.num_word = len(self.vocab_words)
        self.num_tag = len(self.vocab_tags)
        self.num_char = len(self.vocab_chars) 
        
        #processing to map string to id
        self.processing_word = PreProcessData.get_processing_word(self.vocab_words,self.vocab_chars,lowercase=True, chars=self.use_chars)
        self.processing_tag = PreProcessData.get_processing_word(self.vocab_tags, lowercase=False, allow_unk=False)
        
        #pretrained embedding
        self.embbedings = (PreProcessData.processing_trimmed_glove_vector(self.f_trimmed) if self.use_pretrained else None)
    
    #general declaration
    dir_output = "result/test/"
    dir_model = dir_output + "model_weights/"
    path_log = dir_output + "log.txt"
    
    #word and char dimesion
    dim_word = 300
    dim_char = 100
    
    use_pretrained = True
    max_iter = None
    
    #dir location for parts 
    f_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    f_trimmed = "data/glove.6B/glove.6B.{}d.trimmed.npz".format(dim_word)
    # for CoNLL2003
    f_dev = "data/eng.testa.iob"
    f_test = "data/eng.testb.iob"
    f_train = "data/eng.train.iob"
    #f_dev = f_test = f_train = "/data/test.txt"
    f_words = "data/words.txt"
    f_tags = "data/tags.txt"
    f_chars = "data/chars.txt"
    
    #training
    train_embbedings = False
    num_epochs = 10
    drop_out = 0.1
    batch_size = 20
    method = "adam"
    lr_rate = 0.001
    lr_decay = 0.9
    clip = -1
    num_epoch_no_imprv = 3
    
    hidden_size_char = 100
    hidden_size_lstm = 300
    
    use_crf = True
    use_chars = True
    