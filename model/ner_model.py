import numpy as np
import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from .data_utils import *
from .utils import ProgressBar
from .model import Model


class NERModel(Model):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
    def add_placeholders(self):
        # shape = (batch size, max length of sentence in batch)
        self.w_id = tf.placeholder(tf.int32, shape=[None, None], name="word_id")
        # shape = (batch size)
        self.seq_len = tf.placeholder(tf.int32, shape=[None],name="sequence_length")
        # shape = (batch size, max length of sentence, max length of word)
        self.c_id = tf.placeholder(tf.int32, shape=[None, None,self.config.max_len_of_word if self.config.use_chars == 'cnn' else None], name="char_id")
        # shape = (batch_size, max_length of sentence)
        self.w_len = tf.placeholder(tf.int32, shape=[None, None], name="word_length")
        # shape = (batch size, max length of sentence in batch)
        self.label = tf.placeholder(tf.int32, shape=[None, None], name="label")

        # hyper parameters
        self.drop_out = tf.placeholder(dtype=tf.float32, shape=[],
                        name="drop_out")
        self.lr_rate = tf.placeholder(dtype=tf.float32, shape=[],
                        name="learning_rate")

    def feed_dict(self, words, labels=None, lr_rate=None, drop_out=None):
        # perform padding of the given data
        if self.config.use_chars:
            c_id, w_id = zip(*words)
            w_id, seq_len = pad_sequences(w_id, 0)
            c_id, w_len = pad_sequences(c_id, pad_tok=0,
                nlevels=2, max_len=self.config.max_len_of_word)
        else:
            w_id, seq_len = pad_sequences(words, 0)
        feed = { self.w_id: w_id, self.seq_len: seq_len }
        if self.config.use_chars:
            feed[self.c_id] = c_id
            feed[self.w_len] = w_len
        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.label] = labels
        if lr_rate is not None:
            feed[self.lr_rate] = lr_rate
        if drop_out is not None:
            feed[self.drop_out] = drop_out
        return feed, seq_len

    def word_char_emb_option(self):
        with tf.variable_scope("words"):
            if self.config.use_pretrained is None:
                self.log.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.num_word, self.config.dim_word])
                word_embeddings = tf.nn.embedding_lookup(_word_embeddings,self.w_id, name="word_embeddings")
            else:
                if "glove" in self.config.use_pretrained:
                    _word_embeddings = tf.Variable(
                            self.config.embeddings,
                            name="_word_embeddings",
                            dtype=tf.float32,
                            trainable=self.config.train_embeddings)
                    word_embeddings = tf.nn.embedding_lookup(_word_embeddings,self.w_id, name="word_embeddings")
                # Multiple embeddings are used at the same time, we should concatenate them
                if len(self.config.use_pretrained.split(',')) > 1:
                    _embeddings = list()
                    if "glove" in self.config.use_pretrained:
                        _embeddings.append(word_embeddings_w2v)
                    word_embeddings = tf.concat(_embeddings, axis=-1)

        with tf.variable_scope("chars"):
            if self.config.use_chars is not None:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.config.num_char, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.c_id, name="char_embeddings")
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)

                if self.config.use_chars == 'cnn':
                    # CNN2d 
                    char_embeddings = tf.reshape(char_embeddings, shape=[-1,self.config.max_len_of_word, self.config.dim_char])
                    #test CNN 2d
                    char_emb_expand = tf.expand_dims(char_embeddings, -1)
                    output = []
                    for i, ks in enumerate(self.config.kernel_size):
                        with tf.name_scope("conv-char-%s"%ks):
                            fil_shape = [ks, self.config.dim_char, 1, self.config.filters]
                            W = tf.Variable(tf.truncated_normal(fil_shape, stddev=0.1), name="W_char")
                            b = tf.Variable(tf.constant(0.1, shape=[self.config.filters]), name="b_char")
                            conv = tf.nn.conv2d(char_emb_expand, W,strides=[1, 1, 1, 1],padding="VALID", name="conv")
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") 
                            pooled = tf.nn.max_pool(h, ksize=[1, self.config.max_len_of_word-ks+1,1,1], strides=[1, 1, 1, 1],padding='VALID', name="pool")
                            output.append(pooled)
                    num_filters_total = self.config.filters * len(self.config.kernel_size)
                    h_pool = tf.concat(output, 3)
                    h_pool_flat = tf.reshape(h_pool, [s[0], s[1], -1])
                    ws = tf.shape(word_embeddings)
                    word_embeddings = tf.concat([word_embeddings,h_pool_flat ], axis=-1)
                    word_embeddings.set_shape((None, None, 428)) #812 128 #556 64 #428 32 # the shape of word embeddings = word_emb_sie + filter_size*num_filter 
                    # CNN 1d
#                     char_embeddings = tf.reshape(char_embeddings, shape=[-1, self.config.max_len_of_word, self.config.dim_char])
#                     conv1 = tf.layers.conv1d(inputs=char_embeddings, filters=64,kernel_size=3, padding="valid",activation=tf.nn.relu)
#                     conv2 = tf.layers.conv1d(inputs=conv1,filters=64, kernel_size=3, padding="valid",activation=tf.nn.relu)
#                     pool2 = tf.layers.average_pooling1d(inputs=conv2, pool_size=2, strides=1)
#                     output = tf.layers.dense(inputs=pool2, units=32, activation=tf.nn.relu)
#                     output = tf.reshape(output,shape=[s[0], s[1], -1])
#                     ws = tf.shape(word_embeddings)
#                     word_embeddings = tf.concat([word_embeddings, output], axis=-1)
#                     word_embeddings.set_shape((None, None, 780))
                else:
                    char_embeddings = tf.reshape(char_embeddings, shape=[s[0]*s[1],s[-2], self.config.dim_char])
                    w_len = tf.reshape(self.w_len, shape=[s[0] * s[1]])
                    # bi lstm on chars
                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,state_is_tuple=True)
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,state_is_tuple=True)
                    _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings, sequence_length=w_len, dtype=tf.float32)
                    _, ((_, output_fw), (_, output_bw)) = _output
                    output = tf.concat([output_fw, output_bw], axis=-1)
                    output = tf.reshape(output, shape=[s[0], s[1],2*self.config.hidden_size_char])
                    word_embeddings = tf.concat([word_embeddings, output], axis=-1)
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.drop_out)

    def build_main_model(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings, sequence_length=self.seq_len, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.drop_out)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[2*self.config.hidden_size_lstm, self.config.num_tag])

            b = tf.get_variable("b", shape=[self.config.num_tag], dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.num_tag])

    def predict_option(self):
        if not self.config.use_crf:
            self.label_pred = tf.cast(tf.argmax(self.logits, axis=-1),tf.int32)

    def add_crf_option(self):
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.label, self.seq_len)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
            mask = tf.sequence_mask(self.seq_len)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.word_char_emb_option()
        self.build_main_model()
        self.add_crf_option()
        self.predict_option()
        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.method, self.lr_rate, self.loss, self.config.clip)
        self.initialize_session()

    def predict_batch(self, words, return_feed=False):
        fd, seq_len = self.feed_dict(words, drop_out=1.0)
        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.session.run(
                    [self.logits, self.trans_params], feed_dict=fd)
            # iterate over the sentences because no batching in viterbi_decode
            for logit, sequence_length in zip(logits, seq_len):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                viterbi_sequences += [viterbi_seq]
            if return_feed:
                return viterbi_sequences, seq_len, fd
            return viterbi_sequences, seq_len
        else:
            labels_pred = self.session.run(self.label_pred, feed_dict=fd)

            return labels_pred, seq_len

    def run_epoch(self, train, dev, epoch):
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = ProgressBar(target=nbatches)
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.feed_dict(words, labels, self.config.lr_rate, self.config.drop_out)
            _, train_loss, summary = self.session.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)
        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.4f}".format(k, v) for k, v in metrics.items()])
        self.log.info(msg)

        return metrics["f1"]

    def run_evaluate(self, test, print_to_file=False):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        if print_to_file:
            idx_to_word = {idx: word for word, idx in self.config.vocab_words.items()}
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, seq_len, fd = self.predict_batch(words, return_feed=True)

            if print_to_file:
                for s_idx, sentence in enumerate(fd[self.w_id]):
                    for w_idx, word in enumerate(sentence):
                        # Prevent index error
                        if w_idx >= seq_len[s_idx]:
                            break
                        w_label = labels[s_idx][w_idx]
                        w_pred = labels_pred[s_idx][w_idx]
                        write_result(idx_to_word[word] + " " + self.idx_to_tag[w_label] + " " + self.idx_to_tag[w_pred], self.config.f_result)
                write_result("\n", self.config.f_result)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             seq_len):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return {"acc": 100*acc, "f1": 100*f1, "precision": p, "recall": r}

    def predict(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
