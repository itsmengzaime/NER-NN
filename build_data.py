from __future__ import absolute_import
from model.config import Config
from model.data_utils import *


config = Config(load=False)
processing_word = get_processing_word(lowercase=True)

dev = PreProcessData(config.f_dev, processing_word)
test = PreProcessData(config.f_test, processing_word)
train = PreProcessData(config.f_train, processing_word)

vocab_words, vocab_tags = processing_vocab([train, dev, test])
vocab_glove = glove_vocab(config.f_glove)

vocab = vocab_words & vocab_glove
vocab.add(UNK) 
vocab.add(NUM)

writing_vocab(vocab, config.f_words)
writing_vocab(vocab_tags, config.f_tags)
vocab1 = load_dict(config.f_words)
exp_trimmed_glove_vector(vocab1, config.f_glove, config.f_trimmed,config.dim_word)
train_data = PreProcessData(config.f_train)
vocab_chars = processing_char_vocab(train_data)
writing_vocab(vocab_chars, config.f_chars)
