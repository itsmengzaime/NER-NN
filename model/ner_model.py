import tensorflow as tf
import os
import numpy as np

from model import Model
from utils import Progress
from data_utils import minibatches, pad_sequences, get_chunks

class NERModel(Model):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.tag_idx = {idx: tag for tag, idx in self.config.vocab_list.items()}
        
    def initialize_placeholder_tensor(self):
        self.c_id = tf.placeholder(tf.int32, shape=[None, None, None], name="char_id") # [batch_size, max_length_sentence, max_length_word]				
        self.w_id = tf.placeholder(tf.int32, shape=[None, None], name="word_id") #[batch_size, max_length_of_sentence_in_batch]
        self.w_len = tf.placeholder(tf.int32, shapnge=[None, None], name="word_len")  # [batch_size, max_length_sentence]
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="sequence_length") #[batch_size]
        self.label = tf.placeholder(tf.int32, shape=[None, None], name="label") # [batch size, max_length_of_sentence_in_batch]
        self.drop_out = tf.placeholder(tf.float32, shape=[], name="drop_out")
        self.lr_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        
    def feed_dict(self, word, label=None, lr_rate=None, drop_out=None):
        if self.config.use_chars:
            c_id, w_id = zip(*word)
            w_id, seq_len = pad_sequences(w_id, 0)
            c_id, w_len = pad_sequences(c_id, pad_tok=0, nlevel=2)
        else: 
            w_id , seq_len = pad_sequences(word, 0)
        
        feed = {self.w_id: w_id, self.seq_len: seq_len}
        
        if self.config.use_chars:
            feed[self.c_id] = c_id
            feed[self.w_len] = w_len
            
        if label is not None:
            label, _ = pad_sequences(label, 0)
            feed[self.label] = label
        
        if lr_rate is not None:
            feed[self.lr_rate] = lr_rate
        
        if drop_out is not None:
            feed[self.drop_out] = drop_out
        
        return feed, seq_len

    def word_embbeding_option(self):
        with tf.variable_scope("words"):
            if self.config.embbedings is None:
                self.log.info("WARNING: randomly initializing word vectors")
                _word_embbeding = tf.get_variable(name="_word_embbeding",dtype=tf.float32,shape=[self.config.num_word, self.config.dim_word])
            else:
                _word_embbeding = tf.Variable(self.config.embbedings, name="_word_embbeding", dtype=tf.float32, trainable=self.config.train_embbedings)
                
            word_embbedings = tf.nn.embedding_lookup(_word_embbeding, self.w_id, name="word_embbeding")
        
        with tf.variable_scope("chars"):
            if self.config.use_chars:
                _char_embbeding = tf.get_variable(name="_char_embbeding", dtype=tf.float32, shape=[self.config.num_char, self.config.dim_char])
                char_embbedings = tf.nn.embedding_lookup(_char_embbedings, self.c_id, name="char_embbeding")
                s = tf.shape(char_embbedings)

                char_embbedings = tf.reshape(char_embbedings, shape=[s[0]*s[1],s[-2], self.config.dim_char])

                w_len = tf.reshape(self.w_len, shape=[s[0]*s[1]])

                #define bi-LSTM
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,char_embbedings, sequence_length=w_len, dtype=tf.float32)
                
                _, ((_, output_fw),(_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)
                
                output = tf.reshape(output, shape=[s[0],s[1],2*self.config.hidden_size_char])
                word_embbeding = tf.concat([word_embbeding,output], axis=-1)
                
        self.word_embbeding = tf.nn.dropout(word_embbeding, self.drop_out)
    
    def logits_option(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embbeding, sequence_length=self.seq_len, dtype=tf.float32)
            
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.drop_out)
            
        with tf.variable_scope("projection"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[2*self.config.hidden_size_lstm, self.config.n_tag])
            b= tf.get_variable("b", dtype=tf.float32,  shape=[self.config.n_tag], initializer=tf.zeros_initializer())
            num_step = tf.shape(output)[1]
            output = tf.reshape(output, shape=[-1,2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logit = tf.reshape(pred, [-1, num_step, self.config.n_tag])
            
    def prediction_option(self):
        if not self.config.use_crf:
            self.label_pred = tf.cast(tf.argmax(self.logit,axis=-1), tf.int32)
            
    def loss_option(self):        
        if self.config.use_crf:
            log_similar, trans_params = tf.contrib.crf.crf_log_likelihood(self.logit, self.label, self.seq_len)
            self.trans_params = trans_params
            self.loss = tf.reduce_mean(-log_similar)       
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit, labels=self.label)
            mask = tf.sequence_mask(self.seq_len)
            loss = tf.boolean_mask(loss, mask)
            self.loss = tf.reduce_mean(loss)
        
        tf.summary.scalar("loss", self.loss)
        
    def build(self):
        self.initialize_placeholder_tensor()
        self.word_embbeding_option()
        self.logits_option()
        self.prediction_option()
        self.loss_option()
        
        self.add_train_op(self.config.method, self.lr_rate, self.loss, self.config.clip)
        self.initialize_session()
        
        
    def predict_batch(self,word):
        fd, seq_len = self.feed_dict(word, drop_out=1.0)
        if self.config.use_crf:
            viterbi_seq = []
            logit , trans_params = self.session.run([self.logit, self.trans_params], feed_dict=fd)            
            for lg, sl in zip(logit, seq_len):
                lg = lg[:sl]
               	vi_seq , vi_score = tf.contrib.crf.viterbi_decode(lg, trans_params)
                viterbi_seq += [vi_seq]
            
            return viterbi_seq, seq_len
        
    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        num_batch = (len(train) + batch_size -1) // batch_size
        prog = Progress(target=num_batch)
        
        for i, (word, label) in enumerate(minibatches(train, batch_size)):
            fd , _ = self.feed_dict(word, label, self.config.lr_rate, self.config.drop_out)
            _, train_loss, summary = self.session.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            
            prog.update(i+1, [("train loss", train_loss)])
            if (i%10 == 0):
                self.file_writer.add_summary(summary, epoch*num_batch+i)
                
        metric = self.evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        self.log.info(msg)

        return metrics["f1"]
    
    def evaluate(self, test):
        accuracy = []
        correct_prediction = 0.
        total_correct = 0.
        total_prediction = 0.
        for word, label in minibatches(test, self.config.batch_size):
            label_predict, seq_len = self.predict_batch(word)

        for lb, lb_pred, length in zip(label, label_predict, seq_len):
            lb = lb[:length]
            lb_pred = lb_pred[:length]
            accuracy += [a==b for (a,b) in zip(lb, lb_pred)]
            lb_chunks = set(get_chunks(lb, self.config.vocab_list))
            lb_pred_chunks = set(get_chunks(lb_pred, self.config.vocab_list))
            correct_prediction += len(lb_chunks & lb_pred_chunks)
            total_prediction += len(lb_pred_chunks)
            total_correct += len(lb_chunks)
            
        
        precision = correct_prediction / total_prediction if correct_prediction >0 else 0
        recall = correct_prediction / total_correct if correct_prediction >0 else 0
        f1 = 2*precision*recall / (precision+recall) if correct_prediction >0 else 0
        acc = np.mean(accuracy)
        
        return {"accuracy": 100*acc, "f1-score": 100*f1}
    
    def predict(self, raw_word):
        
        word = [self.config.processing_word(w) for w in raw_word]
        if type(word[0]) == tuple:
            word = zip(*word)
        p_id, _ = self.predict_batch([word])
        prediction = [self.tag_idx[idx] for idx in list(p_id[0])]
        
        return prediction


        