#!/usr/bin/env python
#coding:utf8
import numpy
import argparse
import logging
import sys
import cPickle as pkl
from helper import Config
from helper import Dataset
from helper import DataLoader
from helper import prepare_data
from helper import test
import data_reader as dr
import codecs
import time
import tensorflow as tf
from model import SummaRuNNer

logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--sen_len', type=int, default=40)
parser.add_argument('--doc_len', type=int, default=90)
parser.add_argument('--train_file', type=str, default='./data/split_90/')
parser.add_argument('--validation_file', type=str, default='./data/split_90/valid')
parser.add_argument('--model_dir', type=str, default='./runs/1532436443/checkpoints/')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--hidden', type=int, default=110)
parser.add_argument('--lr', type=float, default=1e-4)

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

args = parser.parse_args()
print("dc", args.doc_len)
max_sen_length = args.sen_len
max_doc_length = args.doc_len

logging.info('generate config')
word_vocab, word_tensors, max_doc_length, label_tensors = \
    dr.load_data(args.train_file, max_doc_length, max_sen_length)

batch_size = 1
time1 = time.time()
test_reader = dr.DataReader(word_tensors['test'], label_tensors['test'],
                         batch_size)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph('./runs/1532436443/checkpoints/model-2560.meta')
        module_file = tf.train.latest_checkpoint("./runs/1532436443/" + 'checkpoints/')
        saver.restore(sess, module_file)
        input_x =  graph.get_operation_by_name("inputs/x_input").outputs[0]
        predict =  graph.get_operation_by_name("score_layer/prediction").outputs[0]
        f = codecs.open(args.model_dir+"/scores" , "w", "utf-8")
        jk = 0
        loss_sum = 0
        for x, y in test_reader.iter():
            x = x[0]
            y = y[0]
            #print (x)
            y_ = sess.run(predict, feed_dict = {input_x: x})
             
            flag = 0
            max_len = 0
            for i,item in enumerate(x):
                #print item
                temp = 0
                for sub_item in item:
                    #print(type(int(sub_item)))
                    if sub_item > 0:
                        temp += 1
                        #print temp
                if temp == 0:
                    x = x[:i, :max_len]
                    y_ = y_[:i]
                    y = y[:i]
                    break
                if temp > max_len:
                    max_len = temp
            x = x[:, :max_len]
            target = y
            output = y_
            
            for score in y_:
                print score
                #print(type(score.float))
                #score = score.float
                f.write(str(score))
                f.write(" ")
                jk += 1
                #print("jk:", jk)
            f.write("\n")
        f.close()
