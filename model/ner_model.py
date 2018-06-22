import numpy as np
import os
import tensorflow as tf


from data_utils import minibatches, pad_sequences, get_chunks
from utils import Progbar

from baseModel import Model

class NERModel(Model):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.index_to_tag = {idx: tag for tag, idx in self.config.vocab_tag.items()}
        
    def initialize_placeholder_tensor(self):
        self.word_id = tf.placeholder(tf.int32, shape=[None, None], name="word_id")
        self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="sequence_length")
        self.char_id = tf.placeholder(tf.int32, shape=[None, None, None], name="char_id")
        self.word_length = tf.placeholder(tf.int32, shapnge=[None, None], name="word_length")
        self.label = tf.placeholder(tf.int32, shape=[None, None], name="label")
        self.drop_out = tf.placeholder(tf.float32, shape=[], name="drop_out")
        self.lr_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        
    def feed_dict(self, word, label=None, lr_rate=None, drop_out=None):
        if self.config.use_chars:
            char_id, word_id = zip(*word)
            word_id, sequence_length = pad_sequences(word_id, 0)
            char_id, word_length = pad_sequences(char_id, pad_tok=0, nlevel=2)
        else: 
            word_id , sequence_length = pad_sequences(word, 0)
        
        feed = {self.word_id: word_id, self.sequence_length: sequence_length}
        
        if self.config.use_chars:
            feed[self.char_id] = char_id
            feed[self.word_length] = word_length
            
        if label is not None:
            label, _ = pad_sequences(label, 0)
            feed[self.label] = label
        
        if lr_rate is not None:
            feed[self.lr_rate] = lr_rate
        
        if drop_out is not None:
            feed[self.drop_out] = drop_out
        
        return feed, sequence_length

    def word_embbeding_option(self):
        with tf.variable_scope("words"):
            if self.config.embbedings is None:
                self.log.info("WARNING: randomly initializing word vectors")
                _word_embbedings = tf.get_variable(name="_word_embbedings",dtype=tf.float32,shape=[self.config.nwords, self.config.dim_word])
                
            else:
                _word_embbedings = tf.Variable(self.config.embbedings, name="_word_embbedings", dtype=tf.float32, trainable=self.config.embbedings)
                
            word_embbedings = tf.nn.embedding_lookup(_word_embbedings, self.word_id, name="word_embbeding")
        
        
        with tf.variable_scope("chars"):
            if self.config.use_chars:
                _char_embbedings = tf.get_variable(name="_char_embbeding", dtype=tf.float32,shape=[self.config.nchars, self.config.dim_char])
                char_embbedings = tf.nn.embedding_lookup(_char_embbedings, self.char_id, name="char_embbedings")
                s = tf.shape(char_embbedings)
                char_embbedings = tf.reshape(char_embbedings, shape=[s[0]*s[1],s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_length, shape=[s[0]]*s[1])
                #define bi-LSTM neural network
                
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=char_embbedings,sequence_length=word_lengths, dtype=tf.float32)
                
                _, ((_, output_fw),(_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=1)
                
                output = tf.reshape(output, shape=[s[0]*s[1],2*self.config.hidden_size_char])
                word_embbedings = tf.concat([word_embbedings,output], axis=1)
                
        self.word_embbedings = tf.nn.dropout(word_embbedings, self.drop_out)
    
    def logits_option(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embbedings, sequence_length=self.sequence_length, dtype=tf.float32)
            
            output = tf.concat([output_fw, output_bw], axis=1)
            output = tf.nn.dropout(output, self.drop_out)
            
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[2*self.confog.hidden_size_lstm, self.config.num_tags])
            b= tf.get_variable("b", dtype=tf.float32, shape=[self.config.n_tags],initializer=tf.zeros_initializer())
            num_steps = tf.shape(output)[1]
            output = tf.reshape(output, shape=[-1,2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, num_steps, self.config.num_tags])
            
    def prediction_option(self):
        if not self.config.use_crf:
            self.label_pred = tf.cast(tf.argmax(self.logits,axis=-1), tf.int32)
            
    def loss_option(self):        
        if self.config.use_crf:
            log_similar, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.label, self.sequence_length)
            self.trans_params = trans_params
            self.loss = tf.reduce_mean(-log_similar)
            
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
            mask = tf.sequence_mask(self.sequence_length)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        
        tf.summary.scalar("loss", self.loss)
        
    def build(self):
        self.initialize_placeholder_tensor()
        self.word_embbeding_option()
        self.logits_option()
        self.prediction_option()
        self.loss_option()
        
        self.add_train_op(self.config.method, self.lr_rate, self.loss, self.config.clip)
        self.initialize_session()
        
        
    def predict_batch(self,words):
        fd, sequence_length = self.feed_dict(words, drop_out=1.0)
        if self.config.use_crf:
            bi_sequences = []
            loggits , trans_params = self.session.run([self.logits, self.trans_params], feed_dict=fd)
            
            for lg, seq_len in zip(logits, sequence_length):
                logit = logit[:seq_len]
                bi_seq , bi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                bi_seqs += [bi_seq]
            
            return bi_seqs, sequence_length
        
    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        num_batch = (len(train) + batch_size -1) // batch_size
        prog = Progbar(target=num_batch)
        
        for i, (word, label) in enumerate(minibatches(train, batch_size)):
            fd , _ = self.feed_dict(word, label, self.config.lr_rate, self.config.drop_out)
            _, train_loss, summary = self.session.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            
            prog.update(i+1, [("train loss", train_loss)])
            if (i%10 == 0):
                self.file_writer.add_summary(summary, epoch*num_batch+i)
                
        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]
    
    def evaluate(self, test):
        accs = []
        correct_pred. total_correct, total_pred = 0.,0.,0.
        for word, label in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(word)

        
            
        
        
        
        
        
       
