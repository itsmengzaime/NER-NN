import tensorflow as tf
import os

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
        _m = method.lower()
        
        with tf.variable_scope("train_step"):
            if _m == 'adam':
                optimizer = tf.train.AdamOptimizer(lr_rate)
            elif _m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr_rate)
            elif _m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr_rate)
            elif _m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr_rate)
            else:
                raise NotImplementedError("Unknown method {}".format(_m))
            
            if clip > 0 :
                gd, vs = zip(*optimizer.compute_gradients(loss))
                gd, gnorm = tf.clip_by_norm(gd, clip)
                self.train_op = optimizer.apply_gradients(zip(gd, vs))
            else:
                self.train_op = optimizer.minimize(loss)
        
    def initialize_session(self):
        self.log.info("Initialize tf session")
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.save = tf.train.Saver()
        
    def restore_session(self, dir_model):
        self.log.info("Reload the latest trained model...")
        self.save.restore(self.session, dir_model)
    
    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.save.save(self.session, self.config.dir_model)
    
    def close_session(self):
        self.session.close()

    
    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output, self.session.graph)
        
    def train(self,train, dev):
        record = 0
        num_epoch_no_imprv = 0
        self.add_summary()
        
        for epoch in range(self.config.num_epochs):
            self.log.info("Epoch {:} out of {:}".format(epoch + 1, self.config.num_epochs))
            score = self.run_epoch(train, dev, epoch)
            self.config.lr_rate *= self.config.lr_decay
            
            if score >= record:
                num_epoch_no_imprv = 0
                self.save_session()
                record = score
                self.log.info("New best score recorded...")
            else:
                num_epoch_no_imprv += 1
                if num_epoch_no_imprv >= self.config.num_epoch_no_imprv:
                    self.log.info("Early stopping {} epochs without "\
                            "improvement".format(num_epoch_no_imprv))
                    break            
                    
    def evaluate(self, test):
        self.log.info("Evaluating on test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.log.info(msg)
        