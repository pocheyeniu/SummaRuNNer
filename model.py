#coding:utf-8
import tensorflow as tf
from tensorflow.python import debug as tfdbg

class SummaRuNNer(object):
    def __init__(self, vocabulary_size, embedding_size, pretrained_embedding):
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.pretrained_embedding = pretrained_embedding
        self.batch_size = None
        self.sent_len = 40
        self.hidden_size = 120
        self.doc_len = 90
        with tf.variable_scope('inputs'):
            self.x = tf.placeholder(tf.int32, shape=[self.doc_len, self.sent_len], name = "x_input")
            self.y = tf.placeholder(tf.float32, shape=[self.doc_len])
            self.sequence_length = tf.reduce_sum(tf.sign(self.x), axis = 1)
            self.doc_length = tf.reduce_sum(tf.sign(self.sequence_length), axis = 0)
        with tf.variable_scope('embedding_layer'):
            self.embeddings = tf.get_variable(name='embeddings',  initializer=tf.convert_to_tensor(self.pretrained_embedding), dtype=tf.float32)
            
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.x)
        with tf.variable_scope("sent_level_BiGRU"):
            fw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            self.sent_GRU_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embed, scope='bi-GRU',dtype=tf.float32)
            self.outputs_1 = tf.concat([self.sent_GRU_out[0], self.sent_GRU_out[1]], 2)
            self.result = []
            self.loca_out = self.outputs_1[:self.doc_length, :,:]
            self.input_unstack = tf.unstack(self.outputs_1, axis = 0)
            for index, data in enumerate(self.input_unstack):
                self.avg_pooling = tf.cond(tf.equal(self.sequence_length[index], 0), lambda:tf.reduce_mean(data, axis = 0), lambda:tf.reduce_mean(data[:self.sequence_length[index], :], axis = 0))
                self.result.append(self.avg_pooling)
            self.outputs = tf.stack(self.result)
            self.get_3d_shape = tf.reshape(self.outputs, (1,  -1, 2*self.hidden_size))
        with tf.variable_scope("doc_level_BiGRU"):
            fw_cell_2 = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            bw_cell_2 = tf.nn.rnn_cell.GRUCell(self.hidden_size)
            doc_GRU_out, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell_2, bw_cell_2, self.get_3d_shape,
                                                           dtype=tf.float32)
            self.outputs_2 = tf.concat([doc_GRU_out[0], doc_GRU_out[1]], 2)
            self.result_2 = []
            self.doc_unstack = tf.unstack(self.outputs_2, axis = 0)
            for index, data in enumerate(self.doc_unstack):
                avg_pooling = tf.reduce_mean(data[:self.doc_length, :], axis = 0)
                self.result_2.append(avg_pooling)
            self.outputs_3 = tf.stack(self.result_2)
            #dense layers 
            Wf0 = tf.Variable(tf.random_uniform([2*self.hidden_size, 100], -1.0, 1.0), name='W0')
            bf0 = tf.Variable(tf.zeros(shape=[100]), name='b0')
            self.f_outputs = tf.nn.relu(tf.matmul(self.outputs_3, Wf0) + bf0)
                
            #position embedding
            Wpe = tf.Variable(tf.random_normal([500, self.sent_len]))
            #dense layers 
            Wf = tf.Variable(tf.random_uniform([2*self.hidden_size, 100], -1.0, 1.0), name='W')
            bf = tf.Variable(tf.zeros(shape=[100]), name='b')
            s = tf.Variable(tf.zeros(shape=(100, 1), dtype=tf.float32))
            Wc = tf.Variable(tf.random_normal([1, 100]))
            Ws = tf.Variable(tf.random_normal([100, 100]))
            Wr = tf.Variable(tf.random_normal([100, 100]))
            Wp = tf.Variable(tf.random_normal([1, self.sent_len]))
            bias = tf.Variable(tf.random_normal([1]), name="biases")
            self.doc = tf.transpose(self.f_outputs, perm = (1, 0))
            sent_outputs = tf.reshape(self.outputs_2, (-1, 2 * self.hidden_size))
            scores = []
        with tf.variable_scope("score_layer"):

            for position, sent_hidden in enumerate(tf.unstack(sent_outputs, axis = 0)):
                sent_hidden_tran = tf.reshape(sent_hidden, (1, -1))
                sy = tf.nn.relu(tf.matmul(sent_hidden_tran, Wf) + bf)
                h = tf.transpose(sy, perm = (1, 0))
                pos_embed = tf.nn.embedding_lookup(Wpe, position)
                p = tf.reshape(pos_embed, (-1, 1))
                content = tf.matmul(Wc, h)
                salience = tf.matmul(tf.matmul(tf.reshape(h, (1, -1)), Ws), self.doc)
                novelty = -1 * tf.matmul(tf.matmul(tf.reshape(h, (1, -1)), Wr), tf.tanh(s))
                position = tf.matmul(Wp, p)

                Prob = tf.sigmoid(content + salience + novelty + position + bias)
                s = s + tf.matmul(h, Prob)
                scores.append(Prob[0][0])
            self.y_ = tf.convert_to_tensor(scores, name = "prediction")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):

            epsilon = 1e-8
            target = self.y[:self.doc_length]
            output = self.y_[:self.doc_length]
            self.loss = tf.reduce_mean(-(target * tf.log(output + epsilon) + (1. - target) * tf.log(1. - output + epsilon)), axis=0)

