## Features
model: model_name (str)  
source_train: path to train file of source side.  
target_train: path to train file of target side.  
source_dev: path to file of development set.  
source_test: path to file of test set.

method: [separated_source / separated_target / shared_source / shared_target / shared_mix] (str)

use_train_gpu: (bool)  
use_dev_gpu: (bool)  
use_test_gpu: (bool)  
gpu_device: GPU's ID (This program doesn't have multi-node mode.) (int)  

word2vec: [Make / Load / None] (str)  
word2vec_method: [CBoW / Skip-Gram] (str)  
word2vec_window_size: (int)  
source_word2vec_file: path to word2vec file of source side. (str)  
target_word2vec_file: path to word2vec file of target side. (str)  

vocabulary: [Make / Load] (str)  
source_vocabulary_file: path to vocabulary file of source side (str)  
target_vocabulary_file: path to vocabulary file of source side (str)  

bilstm_method: [FinalAdd, FinalConcat]  
attention_method: [Bahdanau, LuongDot, LuongGeneral, LuongConcat]  
activation_method: [tanh, sigmoid, relu, None]

epoch: The number of training epochs (int)  
optimizer: [Adam, AdaDelta, AdaGrad, MomentumSGD, SGD] (str)  
learning_rate: (int)  
clipping_threshold: (int)  
train_batch_size: (int)  
dev_batch_size: (int)  
test_batch_size: (int)  

use_pretrain: (Bool)  
pretrain_epoch: (int)  

layer_size: (int)  
source_vocabulary_size: min((int), actual_token_types)  
target_vocabulary_size: min((int), actual_token_types)  
embed_size: (int)  
hidden_size: (int)  
use_dropout: (bool)  
dropout_rate: (int)
use_residual: (bool)

generation_limit: (int)  
use_beamsearch: (bool)  
beam_size: (int)  
make_summarized_log: (bool)  
