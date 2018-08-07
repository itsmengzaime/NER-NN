import numpy as np
import os
import tensorflow as tf
from utils import Progress
from model import Model
from data_utils import minibatches, pad_sequences, get_chunks

class NERModel(Model):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.tag_idx = {idx: tag for tag, idx in self.config.vocab_tags.items()}

    def initialize_placeholder_tensor(self):
        # shape = (batch size, max length of sentence, max length of word)
        self.c_id = tf.placeholder(tf.int32, shape=[None, None, None],name="char_id")
        # shape = (batch size, max length of sentence in batch)
        self.w_id = tf.placeholder(tf.int32, shape=[None, None], name="word_id")
        # shape = (batch size, max length of sentence in batch)
        self.label = tf.placeholder(tf.int32, shape=[None, None], name="label")
        # shape = (batch_size, max_length of sentence)
        self.w_len = tf.placeholder(tf.int32, shape=[None, None], name="word_length")
        # shape = (batch size)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name="sequence_length")
        # hyper parameters
        self.drop_out = tf.placeholder(dtype=tf.float32, shape=[], name="drop_out")
        self.lr_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

    def feed_dict(self, word, label=None, lr_rate=None, drop_out=None):
        if self.config.use_chars:
            c_id, w_id = zip(*word)
            w_id, seq_len = pad_sequences(w_id, 0)
            c_id, w_len = pad_sequences(c_id, pad_tok=0,nlevels=2)
        else:
            w_id, seq_len = pad_sequences(word, 0)
            
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
            if self.config.embeddings is None:
                self.log.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable( name="_word_embeddings",dtype=tf.float32, shape=[self.config.num_word, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(self.config.embeddings,name="_word_embeddings",dtype=tf.float32, trainable=self.config.train_embeddings)    
           
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,self.w_id, name="word_embeddings")
  
        # CNN for chars
#         with tf.variable_scope("chars"):
#             if self.config.use_chars:
#                 _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32, shape=[self.config.num_char, self.config.dim_char])
#                 char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.c_id, name="char_embeddings")
#                 s = tf.shape(char_embeddings)
#                 char_embeddings = tf.reshape(char_embeddings, shape=[s[0]*s[1], s[-2], self.config.dim_char])
#                 embedded_char_expand = tf.expand_dims(char_embeddings, -1)
#                 print(char_embeddings, embedded_char_expand)
#                 pooled_output =[]            
#                 for i , fs in enumerate(self.config.fil_size):
#                     with tf.name_scope("cnn-chars-%s"%fs):
#                         temp = embedded_char_expand.get_shape()[1] - fs +1
#                         print temp
#                         fil_shape = [fs, self.config.dim_char, 1, self.config.num_filter]
#                         W = tf.Variable(tf.truncated_normal(fil_shape, stddev=0.1), name="W_char")
#                         b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filter]),name="b_char")
#                         conv = tf.nn.conv2d(embedded_char_expand, W, strides=[1,1,1,1], padding='VALID', name='conv')
#                         h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
#                         pooled = tf.nn.max_pool(h, ksize=[1,self.config.num_char-fs+1,1,1],strides=[1,1,1,1], padding="VALID", name="pool")
# #                         print(fil_shape, W, b, conv, h, pooled)
# #                         print("\n")
#                         pooled_output.append(pooled)
                    
#                 total_fil = self.config.num_filter * len(self.config.fil_size)
#                 h_pool = tf.concat(pooled_output, 3)
#                 h_pool_flat = tf.reshape(h_pool, shape=[-1, s[1],total_fil])
#                 word_embeddings = tf.concat([word_embeddings, h_pool_flat], axis=-1)
#         self.word_embeddings = tf.nn.dropout(word_embeddings, self.drop_out)
        
         # CNN for chars test 
#         with tf.variable_scope("chars"):
#             if self.config.use_chars:
#                 _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32, shape=[self.config.num_char, self.config.dim_char])
#                 # shape = (batch, sentence, word, dim of char embeddings)
#                 char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.c_id, name="char_embeddings")
#                 s = tf.shape(char_embeddings)
#                 print _char_embeddings.get_shape(), char_embeddings.get_shape(), s, s.get_shape()
#                 #shape = [batch_size*seq_len, len_word, emb_dim]
#                 char_embeddings= tf.reshape(char_embeddings, shape=[s[0]*s[1], s[-2], self.config.dim_char])
#                 #expand 1 dimesion similar to image shape = [batch_size*seq_len, len_word, emb_dim, 1]
#                 embedded_char_expand = tf.expand_dims(char_embeddings, -1)
#                 print char_embeddings.get_shape(), embedded_char_expand.get_shape()
#                 pooled_output =[]            
#                 if self.config.use_cnn:
#                     for i,fs in enumerate(self.config.fil_size):
#                         with tf.name_scope("cnn-chars-%s"%fs):
#                             #shape = [kernel_size, emb_dim, 1 , filter_size]
#                             fil_shape = [fs, self.config.dim_char, int(embedded_char_expand.get_shape()[-1]), self.config.num_filter]
#                             W = tf.Variable(tf.truncated_normal(fil_shape, stddev=0.1), name="W_char")
#                             b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filter]),name="b_char")
#                             conv = tf.nn.conv2d(embedded_char_expand, W, strides=[1,1,1,1], padding='VALID', name='conv')
#     #                         h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
#                             pooled = tf.nn.max_pool(conv, ksize=[1,self.config.num_char-fs +1 ,1,1],strides=[1,1,1,1], padding="VALID", name="pool")
#                             pooled_output.append(pooled)

#                     total_fil = self.config.num_filter * len(self.config.fil_size)
#                     h_pool = tf.concat(pooled_output, 3)
#                     h_pool_flat = tf.reshape(h_pool, shape=[-1, s[1],total_fil])
#                     word_embeddings = tf.concat([word_embeddings, h_pool_flat], axis=-1)
#         self.word_embeddings = tf.nn.dropout(word_embeddings, self.drop_out)
        

        #bi-lstm for chars
        with tf.variable_scope("chars"):
            if self.config.use_chars:
                _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32, shape=[self.config.num_char, self.config.dim_char])        
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.c_id, name="char_embeddings")
                s = tf.shape(char_embeddings)
                print _char_embeddings.get_shape(), char_embeddings.get_shape(), s, s.get_shape()
                char_embeddings = tf.reshape(char_embeddings,shape=[s[0]*s[1], s[-2], self.config.dim_char])
                w_len = tf.reshape(self.w_len, shape=[s[0]*s[1]])
                print char_embeddings.get_shape()
                # bi-lstm chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn( cell_fw, cell_bw, char_embeddings, sequence_length=w_len, dtype=tf.float32)
                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)
                print output.get_shape()
                output = tf.reshape(output, shape=[s[0], s[1], 2*self.config.hidden_size_char])
                print output.get_shape()
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)
                print word_embeddings.get_shape()
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.drop_out)
                                              
    def logits_option(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn( cell_fw, cell_bw, self.word_embeddings, sequence_length=self.seq_len, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.drop_out)

        with tf.variable_scope("projection"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[2*self.config.hidden_size_lstm, self.config.num_tag])
            b = tf.get_variable("b", shape=[self.config.num_tag], dtype=tf.float32, initializer=tf.zeros_initializer())
            num_step = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logit = tf.reshape(pred, [-1, num_step, self.config.num_tag])


    def prediction_option(self):
        if not self.config.use_crf:
            self.label_pred = tf.cast(tf.argmax(self.logit, axis=-1), tf.int32)

    def loss_option(self):
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logit, self.label, self.seq_len)
            self.trans_params = trans_params
            self.loss = tf.reduce_mean(-log_likelihood)
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
                                              
        self.add_train_op(self.config.method, self.lr_rate, self.loss,self.config.clip)
        self.initialize_session() 

    def predict_batch(self, word):
        fd, seq_len = self.feed_dict(word, drop_out=1.0)
        if self.config.use_crf:
            viterbi_seq = []
            logit, trans_params = self.session.run([self.logit, self.trans_params], feed_dict=fd)
            for lg, sl in zip(logit, seq_len):
                lg = lg[:sl] # keep only the valid steps
                vi_seq, vi_score = tf.contrib.crf.viterbi_decode(lg, trans_params)
                viterbi_seq += [vi_seq]
            return viterbi_seq, seq_len
        else:
            label_pred = self.session.run(self.label_pred, feed_dict=fd)
            return label_pred, seq_len

    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        num_batch = (len(train) + batch_size - 1) // batch_size
        prog = Progress(target=num_batch)
                                              
        for i, (word, label) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.feed_dict(word, label, self.config.lr_rate, self.config.drop_out)
            _, train_loss, summary = self.session.run([self.train_op, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*num_batch + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        self.log.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        accuracy = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for word, label in minibatches(test, self.config.batch_size):
            label_pred, seq_len = self.predict_batch(word)
            for lab, lab_pred, length in zip(label, label_pred,seq_len):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accuracy    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,self.config.vocab_tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accuracy)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, raw_word):
        word = [self.config.processing_word(w) for w in raw_word]
        if type(word[0]) == tuple:
            word = zip(*word)
        p_id, _ = self.predict_batch([word])
        prediction = [self.idx_to_tag[idx] for idx in list(p_id[0])]
        return prediction

