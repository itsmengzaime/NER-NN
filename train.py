from __future__ import absolute_import
from model.ner_model import NERModel
from model.config import Config
from model.data_utils import *
import tensorflow as tf


config = Config()
model = NERModel(config)

model.build()

train = PreProcessData(config.f_train, config.processing_word,config.processing_tag, config.max_iter)
dev = PreProcessData(config.f_dev, config.processing_word, config.processing_tag, config.max_iter)

model.train(train, dev)

