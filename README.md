# SummaRuNNer
Tensorflow implementation  of SummaRuNNer

the version of Tensorflow is 1.2.0

Usage:
   the data(train, test, valid) directory contains many docs, each line in the doc is label(0 or not important, 1 for important) and sentence(for Chinese, they are segmented words), label and sentence is splited by '\t'.
   
   You also need an Word2vec model instead.

Reference:
   hpzhao 's PyTorch implementation of SummaRuNNer
   https://github.com/hpzhao/SummaRuNNer

Result:
   in our data, it gets almost the same result with the Pytorch one. for ndcg@5, the difference between them is no more than 0.004(~0.5%). 
   
If you have any questions, please feel free to contact me.
