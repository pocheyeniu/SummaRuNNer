from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
import json
import gensim

class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            #print(type(index))
            self._token2index[token] = long(index)
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    @property
    def token2index(self):
        return self._token2index

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        #print(token)
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_doc_length=15, max_sent_length=50):

    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')

    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    h = codecs.open("./index", "w", "utf-8")
    for fname in ('train', 'valid', 'test'):
        #print('reading', fname)
        pname = os.path.join(data_dir, fname)
        for dname in os.listdir(pname):
            if fname == "test":
                h.write(dname)
                h.write("\n")
                #print (dname)
            with codecs.open(os.path.join(pname, dname), 'r', 'utf-8') as f:
                word_doc = []
                label_doc = []
                for line in f.readlines():
                    #print (line)
                    line = line.strip()
                    #line = line.replace('}', '').replace('{', '').replace('|', '')
                    line = line.replace('<unk>', ' | ')
                    if "\t" not in line:
                        continue
                    #l_dict = json.loads(line)
                    label, sent = line.split('\t')
                    if label != "0" and label != "1":
                        label = "1"
                    #print(sent)
                    label_doc.append(label)
                    #print (sent)
                    #sent_list = sent.split(",")
                    sent_list = sent.split(" ")
                    #sent = json.loads(sent).keys()
                    #print(sent)
                    #sent--word list
                    #print(len(sent))
                    if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                        sent_list = sent_list[:max_sent_length-2]

                    word_array = [word_vocab.feed(c) for c in ['{'] + sent_list + ['}']]
                    #print(word_array)
                    word_doc.append(word_array)

                if len(word_doc) > max_doc_length:
                    word_doc = word_doc[:max_doc_length]
                    label_doc = label_doc[:max_doc_length]
                #print(actual_max_doc_length)
                actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

                word_tokens[fname].append(word_doc)
                #print(word_tokens)
                labels[fname].append(label_doc)
    assert actual_max_doc_length <= max_doc_length

    print("max_doc_len:", actual_max_doc_length)
    print('actual longest document length is:', actual_max_doc_length)
    print('size of word vocabulary:', word_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    label_tensors = {}
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length], dtype=np.int64)
        #print(word_tensors)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)
 
        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors

def get_embed(word_vocab):
    model = gensim.models.Word2Vec.load("model.0201")
    f = codecs.open("./oov_list_mm", "w", "utf-8")
    #print("space: ", model[" "])
    embed_list = []
    uni_list = [0 for i in range(150)]
    embed_list.append(uni_list)
    #print (embed_list)
    vocab_dict = word_vocab._token2index
    vocab_sorted = sorted(vocab_dict.iteritems(),key=lambda asd:asd[1], reverse=False)
    i = 0
    n_sum = 0
    w_dict = {}
    for item, index in vocab_sorted:
        #print("space:", item,index, ".")
        n_sum += 1
        if item not in model:
            f.write(item)
            f.write("\n")
            i += 1
            if item not in w_dict.keys():
                w_dict[item] = 1
            else:
                w_dict[item] += 1
            embed_list.append(np.random.uniform(-0.25, 0.25, 150).round(6).tolist())
           
            #embed_list.append(uni_list)
        else:
            #print(list(model[item]))
            embed_list.append(list(model[item]))
    print("no in:", i)
    print("all:", n_sum)
    f.close()
    embedding_array = np.array(embed_list, np.float32)
    print("no in:", i)
    print("all:", n_sum)
    f = codecs.open("./fre_sum", "w", "utf-8")
    for key, value in w_dict.items():
        f.write(key)
        f.write("\t")
        f.write(str(value))
        f.write("\n")
    f.close()
    return embedding_array

class DataReader:

    def __init__(self, word_tensor, label_tensor, batch_size):

        length = word_tensor.shape[0]
        #print (length)
        doc_length = word_tensor.shape[1]
        #print (doc_length)
        sent_length = word_tensor.shape[2]
        #print (sent_length)

        # round down length to whole number of slices

        clipped_length = int(length / batch_size) * batch_size
        #print (clipped_length)
        word_tensor = word_tensor[:clipped_length]
        label_tensor = label_tensor[:clipped_length]
        #print(word_tensor.shape)
        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        #print(x_batches.shape)
        y_batches = label_tensor.reshape([batch_size, -1, doc_length])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        #print(x_batches.shape)
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        #print(len(self._x_batches))
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        #print (self.length)
        self.batch_size = batch_size
        self.max_sent_length = sent_length

    def iter(self):

        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y





if __name__ == '__main__':

    vocab, word_tensors, max_length, label_tensors = load_data('data/demo', 1292, 10)

    count = 0
    for x, y in DataReader(word_tensors['valid'], label_tensors['valid'], 6).iter():
        count += 1
        print (x.shape, y.shape)
        if count > 0:
            break

