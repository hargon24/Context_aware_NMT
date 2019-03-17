import os
import sys
import numpy
import random
import pickle
from chainer import *
from itertools import zip_longest
from collections import defaultdict
from utilities import *

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
