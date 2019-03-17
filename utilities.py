import sys
import numpy
import random
import datetime
from chainer import *
from itertools import zip_longest
from gensim.models import word2vec
from collections import defaultdict

class Configuration:
    def __init__(self, mode, path):
        self.mode = mode
        self.path = path
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    exec('self.{}'.format(line))
        try:
            if self.mode not in ['train', 'test', 'dev']:
                raise ValueError('you must set mode = \'train\' or \'test\' or \'dev\'')
            if self.source_vocabulary_size < 1:
                raise ValueError('you must set source_vocabulary_size >= 1')
            if self.target_vocabulary_size < 1:
                raise ValueError('you must set target_vocabulary_size >= 1')
            if self.embed_size < 1:
                raise ValueError('you must set embed_size >= 1')
            if self.hidden_size < 1:
                raise ValueError('you must set hidden_size >= 1')
            if self.epoch < 1:
                raise ValueError('you must set epoch >= 1')
            if self.use_dropout not in [True, False]:
                raise ValueError('you must set use_dropout = True or False')
            if self.generation_limit < 1:
                raise ValueError('you must set generation_limit >= 1')
            if self.use_beamsearch not in [True, False]:
                raise ValueError('you must set use_beamsearch = True or False')
        except Exception as ex:
            print(ex)
            sys.exit()

        if self.mode == 'train':
            self.use_gpu = self.use_train_gpu
            self.batch_size = self.train_batch_size
        elif self.mode == 'dev':
            self.use_gpu = self.use_dev_gpu
            self.batch_size = self.dev_batch_size
            self.use_beamsearch = False
        elif self.mode == 'test':
            self.use_gpu = self.use_test_gpu
            self.batch_size = self.test_batch_size
            if self.batch_size != 1:
                self.use_beamsearch = False

        if self.use_gpu:
            import cupy
            self.library = cupy
        else:
            self.library = numpy
        
        if self.mode == 'train': 
            self.optimizer = self.set_optimizer(self.optimizer)
        
        if not self.use_dropout:
            self.dropout_rate = 0.0
        
        if not self.use_beamsearch:
            self.beam_size = 1

    def set_optimizer(self, opt):
        if opt == 'AdaGrad':
            opt = optimizers.AdaGrad(lr = self.learning_rate)
        elif opt == 'AdaDelta':
            opt = optimizers.AdaDelta()
        elif opt == 'Adam':
            opt = optimizers.Adam()
        elif opt == 'SGD':
            opt = optimizers.SGD(lr = self.learning_rate)
        elif opt == 'MomentumSGD':
            opt = optimizers.MomentumSGD(lr = self.learning_rate)
        elif opt == 'NesterovAG':
            opt = optimizers.NesterovAG(lr = self.learning_rate)
        elif opt == 'RMSprop':
            opt = optimizers.RMSprop(lr = self.learning_rate)
        elif opt == 'RMSpropGraves':
            opt = optimizers.RMSpropGraves(lr = self.learning_rate)
        elif opt == 'SMORMS3':
            opt = optimizers.SMORMS3(lr = self.learning_rate)
        return opt

class Vocabulary:
    def make(path, vocabulary_size):
        self = Vocabulary()
        self.word2id = defaultdict(lambda: 0)
        self.id2word = dict()
        with open(path, 'r') as f:
            word_frequency = defaultdict(lambda: 0)
            for words in f:
                for word in words.strip('\n').split(' '):
                    word_frequency[word] += 1
            self.word2id['<unk>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id[''] = -1 #for padding
            self.id2word[0] = '<unk>'
            self.id2word[1] = '<s>'
            self.id2word[2] = '</s>'
            self.id2word[-1] = '' #for padding
            for i, (word, frequency) in zip(range(vocabulary_size - 3), sorted(sorted(word_frequency.items(), key = lambda x: x[0]), key = lambda x: x[1], reverse = True)):
                self.word2id[word] = i + 3
                self.id2word[i + 3] = word
        self.size = len(self.word2id) - 1
        return self

    def save(self, path):
        with open(path, 'w') as f:
            for i in range(self.size):
                f.write(self.id2word[i] + '\n')

    def load(path):
        self = Vocabulary()
        self.word2id = defaultdict(lambda: 0)
        self.id2word = dict()
        with open(path, 'r') as f:
            for i, word in enumerate(f):
                word = word.strip('\n')
                self.word2id[word] = i
                self.id2word[i] = word
        self.size = len(self.word2id)
        self.word2id[''] = -1 #for padding
        self.id2word[-1] = '' #for padding
        return self

class Word2vec:
    def make(path, method, embed_size, window_size):
        self = Word2vec()
        if method == 'CBoW':
            self.model = word2vec.Word2Vec(word2vec.LineSentence(path), sg = 0, size = embed_size, window = window_size, min_count = 1)
        elif method == 'Skip-Gram':
            self.model = word2vec.Word2Vec(word2vec.LineSentence(path), sg = 1, size = embed_size, window = window_size, min_count = 1)
            
        return self

    def save(self, path):
        self.model.save(path)

    def load(path):
        self = Word2vec()
        self.model = word2vec.Word2Vec.load(path)
        return self

    def read_corpus(path):
        data = list()
        with open(path, 'r') as f:
            for line in f:
                data.append(line.strip('\n').split(' '))
        return data


def load_pretrain_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        
        if type(child) != type(dst_child) and child.name != 'decoder':
            continue
        
        if isinstance(child, link.Chain):
            load_pretrain_model(child, dst_child)
        
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape and child.name != 'tilde':
                    match = False
                    break
            if not match:
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[1].data.shape != b[1].data.shape:
                    b[1].data = functions.concat((a[1], b[1][:, a[1].data.shape[1]:])).data
                else:
                    b[1].data = a[1].data

def sort_train_batch(pool, batch_size):
    batch = list()
    result = list()
    for document in sorted(pool, key=lambda x:len(x), reverse=True):
        batch.append(document)
        if len(batch) == batch_size:
            max_length = len(batch[0])
            source_batch = list()
            target_batch = list()
            for doc in batch:
                if len(doc) > 0:
                    transposed_document = list(zip(*doc))
                    source_batch.append(list(transposed_document[0]))
                    target_batch.append(list(transposed_document[1]))
            for i in range(len(source_batch)):
                while len(source_batch[i]) < max_length:
                    source_batch[i].append([-1])
                    target_batch[i].append([-1])
            result.append((list(zip(*source_batch)), list(zip(*target_batch))))

            batch = list()
    
    if len(batch) > 0:
        max_length = len(batch[0])
        source_batch = list()
        target_batch = list()
        for doc in batch:
            if len(doc) > 0:
                transposed_document = list(zip(*doc))
                source_batch.append(list(transposed_document[0]))
                target_batch.append(list(transposed_document[1]))
        for i in range(len(source_batch)):        
            while len(source_batch[i]) < max_length:
                source_batch[i].append([-1])
                target_batch[i].append([-1])
        result.append((list(zip(*source_batch)), list(zip(*target_batch))))

    return result


def make_train_batch(source_path, source_vocab, target_path, target_vocab, batch_size):
    pool = list()
    document = list()
    sbos = [source_vocab.word2id['<s>']]
    seos = [source_vocab.word2id['</s>']]
    tbos = [target_vocab.word2id['<s>']]
    teos = [target_vocab.word2id['</s>']]
    for sline, tline in zip(open(source_path), open(target_path)):
        if len(sline.strip()) == 0:
            pool.append(document)
            document = list()
        else:
            source_sentence = sbos + [source_vocab.word2id[word] for word in sline.strip().split()] + seos
            target_sentence = tbos + [target_vocab.word2id[word] for word in tline.strip().split()] + teos
            document.append((source_sentence, target_sentence))

    
    if len(pool) > 0:
        batch = sort_train_batch(pool, batch_size)
        random.shuffle(batch)
        for source_batch, target_batch in batch:
            yield source_batch, target_batch


def arrange_test_batch(pool, batch_size):
    batch = list()
    result = list()
    for document in pool:
        batch.append(document)
        if len(batch) == batch_size:
            max_length = max([len(x) for x in batch])
            for i in range(len(batch)):
                while len(batch[i]) < max_length:
                    batch[i].append([-1])
            result.append(list(zip(*batch)))
            batch = list()
    
    if len(batch) > 0:
        max_length = max([len(x) for x in batch])
        for i in range(len(batch)):
            while len(batch[i]) < max_length:
                batch[i].append([-1])
        result.append(list(zip(*batch)))

    return result


def make_test_data(data_path, vocabulary, batch_size):
    pool = list()
    document = list()
    bos = [vocabulary.word2id['<s>']]
    eos = [vocabulary.word2id['</s>']]

    for line in open(data_path):
        if len(line.strip()) == 0:
            pool.append(document)
            document = list()
        else:
            sentence = bos + [vocabulary.word2id[word] for word in line.strip().split()] + eos
            document.append(sentence)

    if len(document) > 0:
        pool.append(document)

    pool = arrange_test_batch(pool, batch_size)
    for batch in pool:
        yield batch


def make_pretest_batch(data_path, vocabulary, batch_size):
    batch = list()
    bos = [vocabulary.word2id['<s>']]
    eos = [vocabulary.word2id['</s>']]
    
    for line in open(data_path):
        #if len(line.strip()) == 0:
        #    continue
        
        sentence = bos + [vocabulary.word2id[word] for word in line.strip().split()] + eos
        batch.append(sentence)
        if len(batch) == batch_size:
            yield batch
            batch = list()
        
    if len(batch) > 0:
        yield batch
    else:
        trace('There is no sentence in this file.')
        exit()


def sort_pretrain_batch(pool, batch_size):
    batch = list()
    batch_pool = list()
    for pair in sorted(pool, key=lambda x:len(x[1]), reverse=True):
        batch.append(pair)
        if len(batch) == batch_size:
            source_batch = list()
            target_batch = list()
            for ssent, tsent in batch:
                source_batch.append(ssent)
                target_batch.append(tsent)

            batch_pool.append((source_batch, target_batch))
            batch = list()
    
    if len(batch) > 0:
        source_batch = list()
        target_batch = list()
        for ssent, tsent in batch:
            source_batch.append(ssent)
            target_batch.append(tsent)

        batch_pool.append((source_batch, target_batch))
        batch = list()

    return batch_pool


def make_pretrain_batch(source_path, source_vocab, target_path, target_vocab, batch_size):
    pool = list()
    document = list()
    sbos = [source_vocab.word2id['<s>']]
    seos = [source_vocab.word2id['</s>']]
    tbos = [target_vocab.word2id['<s>']]
    teos = [target_vocab.word2id['</s>']]
    
    for sline, tline in zip(open(source_path), open(target_path)):
        #if len(sline.strip()) == 0:
        #    continue

        source_sentence = sbos + [source_vocab.word2id[word] for word in sline.strip().split()] + seos
        target_sentence = tbos + [target_vocab.word2id[word] for word in tline.strip().split()] + teos
        pool.append((source_sentence, target_sentence))
    
    if len(pool) > 0:
        batch = sort_pretrain_batch(pool, batch_size)
        random.shuffle(batch)
        for source_batch, target_batch in batch:
            yield source_batch, target_batch

    else:
        trace('There is no sentence in this file.')
        exit()


def make_sentence(word_list, vocabulary):
    sentence = [vocabulary.id2word[i] for i in list(word_list)]
    
    if '</s>' in sentence:
        return ' '.join(sentence[sentence.index('<s>') + 1: sentence.index('</s>')])
    elif '<s>' not in sentence:
        return 'NULL'
    else:
        return ' '.join(sentence[sentence.index('<s>') + 1:])


def trace(*args):
	print(datetime.datetime.now(), '...', *args, file=sys.stderr)
	sys.stderr.flush()
