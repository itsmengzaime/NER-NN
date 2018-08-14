import os
import subprocess
from model.config import Config
from model.data_utils import *


def main():
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word(lowercase=False)

    # Generators
    dev   = PreProcessData(config.f_dev, processing_word)
    test  = PreProcessData(config.f_test, processing_word)
    train = PreProcessData(config.f_train, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])

    vocab_glove = get_word_vec_vocab(config.f_glove)
    vocab = vocab_words & vocab_glove
    vocab.add(NUM)
    vocab.add(UNK)

    # Save vocab
    write_vocab(vocab, config.f_words)
    write_vocab(vocab_tags, config.f_tags)

    # Trim GloVe Vectors
    if "glove" in config.use_pretrained:
        vocab = load_vocab(config.f_words)
        export_trimmed_word_vectors(vocab, config.f_glove,config.f_trimmed, config.dim_word)

    # Build and save char vocab
    train = PreProcessData(config.f_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.f_chars)


if __name__ == "__main__":
    main()
