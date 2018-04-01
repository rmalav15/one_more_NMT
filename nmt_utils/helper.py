from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import codecs
#import os
import tensorflow as tf
import codecs

from tensorflow.python.ops import lookup_ops

UNK_ID = 0

class data_helper:
    
    def __init__(self, opt):
        self.opt = opt
        #self.source_vocab_size = None #Test
        #self.dest_vocab_size = None
        
    def load_vocab(self, vocab_file):
        vocab = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            vocab_size = 0
            for word in f:
                vocab_size += 1
                vocab.append(word.strip())
        return vocab, vocab_size
    
    def create_vocab_tables(self):
        """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
        src_vocab_file, tgt_vocab_file, share_vocab  = self.opt.source_vocab_file, \
                        self.opt.dest_vocab_file, False
        
        src_vocab_table = lookup_ops.index_table_from_file(
                                  src_vocab_file, default_value=UNK_ID)
        if share_vocab:
            tgt_vocab_table = src_vocab_table
        else:
            tgt_vocab_table = lookup_ops.index_table_from_file(
                                 tgt_vocab_file, default_value=UNK_ID)
        return src_vocab_table, tgt_vocab_table
    
    
    