#! /usr/bin/env python
#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model import SummaRuNNer
#import data_input_helper as data_helpers
#from text_cnn import TextCNN
import math
from tensorflow.contrib import learn
import data_reader as dr
import tensorlayer as tl
import logging
# Parameters
# ==================================================

# Data loading params
#tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("train_data_file", "/var/proj/sentiment_analysis/data/cutclean_tiny_stopword_corpus10000.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("train_data_file", "./data/split_90", "Data source for the positive data.")
#tf.flags.DEFINE_string("valid_data_file", "../data/split90/valid", "Data source for the positive data.")
#tf.flags.DEFINE_string("test_data_file", "../data/split90/test", "Data source for the positive data.")
#tf.flags.DEFINE_string("train_label_data_file", "", "Data source for the label data.")
#需要修改
tf.flags.DEFINE_string("w2v_file", "../model.0201", "w2v_file path")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 150, "Dimensionality of character embedding (default: 128)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
'''
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
'''






def train(w2v_model):
    # Training
    # ==================================================
    max_sen_length = 40
    max_doc_length = 90
    word_vocab, word_tensors, max_doc_length, label_tensors = \
 \
        dr.load_data(FLAGS.train_data_file, max_doc_length, max_sen_length)

    train_reader = dr.DataReader(word_tensors['train'], label_tensors['train'], 1)

    valid_reader = dr.DataReader(word_tensors['valid'], label_tensors['valid'], 1)

    test_reader = dr.DataReader(word_tensors['test'], label_tensors['test'], 1)
    pretrained_embedding = dr.get_embed(word_vocab)
    print ("ppp", pretrained_embedding.shape)
    #x_train, x_dev, y_train, y_dev ,vocab_siz, pretrained_embedding= load_data(w2v_model)
    embedding_size = 150
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            #需修改
            '''
            cnn = TextCNN(
                w2v_model,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            '''
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it

            print word_vocab.size
            Summa = SummaRuNNer(
                word_vocab.size, embedding_size, pretrained_embedding
            )

            #loss_sum = tf.Variable(initial_value=0, dtype=tf.float32)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            '''
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(Summa.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            '''

            train_params = tf.trainable_variables()
            train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(Summa.loss, var_list=train_params)
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            batches = train_reader
            valid_batches = valid_reader
            sess.run(tf.global_variables_initializer())
            #step = 0
            min_eval_loss = float('Inf')
            #fetch_list = [Summa.xks, Summa.data_c, Summa.result]
            for epoch in range(FLAGS.num_epochs):
                step = 0
                loss_sum = 0
                #loss_sum.assign(value=0)
                #value_sum = 0

                for x_batch, y_batch in batches.iter():
                    step += 1
                    feed_dict = {
                      Summa.x: x_batch[0],
                      Summa.y: y_batch[0],
                    }
                    '''
                    lucky_boy, lucky_girl, data_cc = sess.run(fetch_list, feed_dict)
                    print ("lucky_boy, ", lucky_boy)
                    print ("lucky_girl, ", lucky_girl)
                    print ("data_cc:", data_cc)
                    '''
                    sess.run(train_op,feed_dict)
                    loss = sess.run(
                        [Summa.loss],
                        feed_dict)
                    predict = sess.run([Summa.y_], feed_dict)
                    loss_sum += loss[0]
                    #print predict
                    #grads_and_vars = optimizer.compute_gradients(Summa.loss)
                    #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                    #print step
                    #print len(y_batch[0])
                    #print len(predict[0])
                    #print step % 128
                    #print step
                    if step % 128 == 0 and step != 0:
                        #print ("here")
                        #logging.info('Epoch ' + str(epoch) + ' Loss: ' + str(loss_sum / 128.0))
                        print ('Epoch ' + str(epoch) + ' Loss: ' + str(loss_sum / 128.0))
                        loss_sum = 0
                    if step % 512 == 0 and step != 0:
                        eval_loss = 0
                        for x_batch, y_batch in valid_batches.iter():
                            feed_dict = {
                                Summa.x: x_batch[0],
                                Summa.y: y_batch[0],
                            }
                            loss = sess.run(
                                [Summa.loss],
                                feed_dict)
                            eval_loss += loss[0]
                        print ('epoch ' + str(epoch) + ' Loss in validation: ' + str(
                                eval_loss * 1.0 / valid_reader.length))
                        if eval_loss < min_eval_loss:
                            min_eval_loss = eval_loss

                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))



if __name__ == "__main__":
    #w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
    train("model.0201")
