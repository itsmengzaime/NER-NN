import tensorflow as tf
import os
import numpy as np
import logging
import time
import sys

def logging_file(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

class Progress(object):
    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.verbose = verbose
        self.sum_value = {}
        self.unique_value = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        
    def update(self, current, value=[], exact=[], strict=[]):
        for k,v in value:
            if k not in self.sum_value:
                self.sum_value[k] =[v*(current - self.seen_so_far), current - self.seen_so_far]
                self.unique_value.append(k)
            else:
                self.sum_value[k][0] += v*(current - self.seen_so_far)
                self.sum_value[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_value:
                self.unique_value.append(k)
            self.sum_value[k] = [v,1]
        
        for k, v in strict:
            if k not in self.sum_value:
                self.unique_value.append(k)
            self.sum_value[k] = v
                
        self.seen_so_far = current
        
        now = time.time()
        
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")
            
            num_digit = int(np.floor(np.log10(self.target))) +1 
            string_bar = '%%%dd/%%%dd [' % (num_digit, num_digit)
            bar = string_bar % (current, self.target)
            prog = float(current)/ self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)
            
            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' -ETA: %ds '%eta
            else:
                info += ' -%ds' %(now - self.start)

            for k in self.unique_value:
                if type(self.sum_value[k]) is list:
                    info += ' -%s: %.4f ' %(k, self.sum_value[k][0] / max(1, self.sum_value[k][1]))
                else:
                    info += ' -%s: %s ' %(k, self.sum_value[k])

            self.total_width += len(info)

            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width)*" ")
            sys.stdout.write(info)
            sys.stdout.flush()

            if current > self.target: 
                sys.stdout.write("\n")
        
        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' %(now-self.start)
                
                for k in self.unique_value:
                    info += '- %s: %.4f' %(k, self.sum_value[k][0] /  max(1, self.sum_value[k][1]))
                    
                sys.stdout.write(info + "\n")
        
    def add(self, n, value=[]):
        self.update(self.seen_so_far+n, value)

                