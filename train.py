from model.data_utils import PreProcessData
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")s
    # create datasets
    dev   = PreProcessData(config.f_dev,config.processing_word, config.processing_tag, config.max_iter)
    train = PreProcessData(config.f_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main()
