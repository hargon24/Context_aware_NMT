import os
import sys
import numpy
import random
import pickle
from chainer import *
from itertools import zip_longest
from collections import defaultdict
from utilities import *

def sort_batch_length(pool, batch_size):
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
        batch = sort_batch_length(pool, batch_size)
        random.shuffle(batch)
        for source_batch, target_batch in batch:
            yield source_batch, target_batch


def make_test_batch(pool, batch_size):
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


def monolingual_pooling_nodocs(data_path, vocabulary, batch_size, pooling_size=100):
    pool = list()
    document = list()
    actual_pooling_size = pooling_size * batch_size
    bos = vocabulary.word2id['<s>']
    eos = vocabulary.word2id['</s>']
    
    for line in open(data_path):
        if len(line.strip()) == 0:
            continue
        sentence = [vocabulary.word2id[word] for word in line.strip().split()]
        sentence.insert(0, bos)
        sentence.append(eos)
        pool.append(sentence)

        if len(pool) == actual_pooling_size:
            batch = list()
            for sent in sorted(pool, key=lambda x:len(x)):
                batch.append(sent)
                if len(batch) > batch_size:
                    yield batch
                    batch = list()
            
            if len(batch) > 0:
                yield batch
            pool = list()
    
    if len(pool) > 0:
        batch = list()
        for sent in sorted(pool, key=lambda x:len(x)):
            batch.append(sent)
            if len(batch) > batch_size:
                yield batch
                batch = list()
        
        if len(batch) > 0:
            yield batch


def make_test_data(data_path, vocabulary, batch_size, boundary=10):
    pool = list()
    document = list()
    bos = vocabulary.word2id['<s>']
    eos = vocabulary.word2id['</s>']

    if boundary is True:
        for line in open(data_path):
            if len(line.strip()) == 0:
                pool.append(document)
                document = list()
            else:
                sentence = [vocabulary.word2id[word] for word in line.strip().split()]
                sentence.insert(0, bos)
                sentence.append(eos)
                document.append(sentence)

    elif isinstance(boundary, int):
        for line in open(data_path):
            if len(line.strip()) == 0:
                continue
            sentence = [vocabulary.word2id[word] for word in line.strip().split()]
            sentence.insert(0, bos)
            sentence.append(eos)
            document.append(sentence)

            if len(document) == boundary:
                pool.append(document)
                document = list()
    
    else:
        trace("Boundary must be set to an integer or a boolean.")
        exit()
    
    if len(document) > 0:
        pool.append(document)

    pool = make_test_batch(pool, batch_size)
    for batch in pool:
        yield batch
