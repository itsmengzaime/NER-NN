import os
import tensorflow as tf

class Model(object):
    def __init__(self, config):
        self.config = config
        self.log = config.log
        self.session = None
        self.save = None
    
    def initialize_weights(self, scope):
        variables = tf.contrib.framework.get_variables(scope)
        init = tf.variables_initializer(variables)
        self.session.run(init)
    
    def add_train_op(self, method, lr_rate, loss, clip=-1):
        _method = method.lower()
        
        with tf.variable_scope("train_step"):
            if _method == 'adam':
                optimizer = tf.train.AdamOptimizer(lr_rate)
            elif _method == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr_rate)
            elif _method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr_rate)
            elif _method == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr_rate)
            else:
                raise NotImplementedError("Unknown method {}".format(_method))
            
            if clip > 0 :
                grad, vs = zip(*optimizer.compute_gradients(loss))
                grad, gnorm = tf.clip_by_norm(grad, clip)
                self.train_op = optimizer.apply_gradients(zip(grad, vs))
            else:
                self.train_op = optimizer.minimize(loss)
        
    def initialize_session(self):
        self.log.info("Initialize Tensorflow Session")
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.save = tf.train.Saver()
        
    def restore_session(self, dir_model):
        self.log.info("Reload the latest trained model")
        self.save.restore(self.session, dir_model)
    
    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.save.save(self.session, self.config.dir_model)
    
    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config,dir_output, self.session.graph)
        
    def train(self,train, dev):
        fixed_score = 0
        num_epoch_no_imprv = 0
        self.add_summary()
        
        for epoch in range(self.config.num_epochs):
            self.log.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))
            score = self.run_epoch(train, dev, epoch)
            self.config.learning_rate *= self.config.lr_decay
            
            if score >= fixed_score:
                num_epoch_no_imprv = 0
                self.save_session()
                fixed_score = score
                self.log.info("-- new best score")
            else:
                num_epoch_no_imprv += 1
                if num_epoch_no_imprv >= self.config.num_epoch_no_imprv:
                    self.log.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break            
                    
    def evaluate(self, test):
        self.log.info("Testing model over test set")
        metrics  = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.log.info(msg)
