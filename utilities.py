import numpy
from chainer import *
from itertools import zip_longest
from collections import defaultdict
import sys
import datetime
from gensim.models import word2vec
import random

class TransformerOptimizer:
    def __init__(self, model_size):
        self.opt = optimizers.Adam(beta1 = 0.9, beta2 = 0.98, eps = 10 ** -9)
        self.warmup_steps = 4000
        self.model_size = model_size

    def setup(self, model):
        self.opt.setup(model)

    def add_hook(self, hook):
        self.opt.add_hook(hook)
    
    def update(self):
        self.opt.lr = (self.model_size ** -0.5) * min(self.opt.t ** -0.5, self.opt.t * (self.warmup_steps ** -1.5))
        self.opt.update()

class Configuration:
    def __init__(self, mode, path):
        self.mode = mode
        self.path = path
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    exec("self.{}".format(line))
        try:
            if self.mode not in ['train', 'test', 'dev', 'pretrain', 'pretest', 'predev', 'lm', 'lmdev']:
                raise ValueError('you must set mode = \'(pre)train\' or \'(pre)test\' or \'(pre)dev\'')
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

        if self.mode.endswith("train") or self.mode.endswith("lm"):
            self.use_gpu = self.use_train_gpu
            self.batch_size = self.train_batch_size
        elif self.mode.endswith("dev") or self.mode.endswith("lmdev"):
            self.use_gpu = self.use_dev_gpu
            self.batch_size = self.dev_batch_size
            self.use_beamsearch = False
            self.use_reconstructor_beamsearch = False
        elif self.mode.endswith("test"):
            self.use_gpu = self.use_test_gpu
            self.batch_size = self.test_batch_size
            if self.batch_size != 1:
                self.use_beamsearch = False
                self.use_reconstructor_beamsearch = False

        if self.use_gpu:
            import cupy
            self.library = cupy
        else:
            self.library = numpy
      
        if not hasattr(self, "generator_optimizer"):
            self.generator_optimizer = self.optimizer
        if not hasattr(self, "discriminator_optimizer"):
            self.discriminator_optimizer = self.optimizer
        
        if self.mode.endswith("train") or self.mode.endswith("lm"): 
            if hasattr(self, "optimizer"):
                self.optimizer = self.set_optimizer(self.optimizer)
            if hasattr(self, "generator_optimizer"):
                self.generator_optimizer = self.set_optimizer(self.generator_optimizer)
            if hasattr(self, "discriminator_optimizer"):
                self.discriminator_optimizer = self.set_optimizer(self.discriminator_optimizer)
        
        if not self.use_dropout:
            self.dropout_rate = 0.0
        
        if not self.use_beamsearch:
            self.beam_size = 1

    def set_optimizer(self, opt):
        if opt == "AdaGrad":
            opt = optimizers.AdaGrad(lr = self.learning_rate)
        elif opt == "AdaDelta":
            opt = optimizers.AdaDelta()
        elif opt == "Adam":
            opt = optimizers.Adam()
        elif opt == "SGD":
            opt = optimizers.SGD(lr = self.learning_rate)
        elif opt == "MomentumSGD":
            opt = optimizers.MomentumSGD(lr = self.learning_rate)
        elif opt == "NesterovAG":
            opt = optimizers.NesterovAG(lr = self.learning_rate)
        elif opt == "RMSprop":
            opt = optimizers.RMSprop(lr = self.learning_rate)
        elif opt == "RMSpropGraves":
            opt = optimizers.RMSpropGraves(lr = self.learning_rate)
        elif opt == "SMORMS3":
            opt = optimizers.SMORMS3(lr = self.learning_rate)
        elif opt == "Transformer":
            opt = TransformerOptimizer(self.embed_size)
        return opt

class Vocabulary:
    def make(path, vocabulary_size):
        self = Vocabulary()
        self.word2id = defaultdict(lambda: 0)
        self.id2word = dict()
        with open(path, "r") as f:
            word_frequency = defaultdict(lambda: 0)
            for words in f:
                for word in words.strip("\n").split(" "):
                    word_frequency[word] += 1
            self.word2id["<unk>"] = 0
            self.word2id["<s>"] = 1
            self.word2id["</s>"] = 2
            self.word2id[""] = -1 #for padding
            self.id2word[0] = "<unk>"
            self.id2word[1] = "<s>"
            self.id2word[2] = "</s>"
            self.id2word[-1] = "" #for padding
            for i, (word, frequency) in zip(range(vocabulary_size - 3), sorted(sorted(word_frequency.items(), key = lambda x: x[0]), key = lambda x: x[1], reverse = True)):
                self.word2id[word] = i + 3
                self.id2word[i + 3] = word
        self.size = len(self.word2id) - 1
        return self

    def save(self, path):
        with open(path, "w") as f:
            for i in range(self.size):
                f.write(self.id2word[i] + "\n")

    def load(path):
        self = Vocabulary()
        self.word2id = defaultdict(lambda: 0)
        self.id2word = dict()
        with open(path, "r") as f:
            for i, word in enumerate(f):
                word = word.strip("\n")
                self.word2id[word] = i
                self.id2word[i] = word
        self.size = len(self.word2id)
        self.word2id[""] = -1 #for padding
        self.id2word[-1] = "" #for padding
        return self

class Word2vec:
    def make(path, method, embed_size, window_size):
        self = Word2vec()
        if method == "CBoW":
            self.model = word2vec.Word2Vec(word2vec.LineSentence(path), sg = 0, size = embed_size, window = window_size, min_count = 1)
        elif method == "Skip-Gram":
            self.model = word2vec.Word2Vec(word2vec.LineSentence(path), sg = 1, size = embed_size, window = window_size, min_count = 1)
        #elif method == "Glove":
            #corpus = Corpus()
            #corpus.fit(read_corpus(path), window = window_size)
            #glove = Glove(no_components=100, learning_rate=0.05)
            #glove.fit(corpus.matrix, epochs=int(args.train), no_threads=args.parallelism, verbose=True)
            #glove.add_dictionary(corpus.dictionary)
            
        return self

    def save(self, path):
        self.model.save(path)

    def load(path):
        self = Word2vec()
        self.model = word2vec.Word2Vec.load(path)
        return self

    def read_corpus(path):
        data = list()
        with open(path, "r") as f:
            for line in f:
                data.append(line.strip("\n").split(" "))
        return data

class UnknownDictionary:
    def load(path):
        self = UnknownDictionary()
        self.dictionary = dict()
        with open(path, "r") as f:
            for line in f:
                source, target = line.strip("\n").split("\t")
                self.dictionary[source] = target
        return self
    
    def replace(self, source, target, attention):
        source = source.split(" ")
        target = target.split(" ")
        attention = cuda.to_cpu(attention)
        for i, word in enumerate(target):
            if word == "<unk>":
                alignment_index = numpy.argmax(attention[i])
                if alignment_index < len(source):
                    if source[alignment_index] in self.dictionary:
                        word = self.dictionary[source[alignment_index]]
                    else:
                        word = source[alignment_index]
                    target[i] = word
        return " ".join(target)

def convert_wordlist(batch, vocabulary):
    for sentence in list(cuda.to_cpu(functions.transpose(functions.vstack(batch)).data)):
        word_list = list()
        for i in list(sentence):
            word_list.append(vocabulary.id2word[i])
        if "</s>" in word_list:
            yield " ".join(word_list[word_list.index("<s>")+1:word_list.index("</s>")])
        else:
            yield " ".join(word_list[word_list.index("<s>")+1:])

def convert_sentence(sentence, vocabulary):
    word_list = [vocabulary.id2word[i] for i in list(cuda.to_cpu(sentence.data))]
    if "</s>" in word_list:
        return " ".join(word_list[word_list.index("<s>")+1:word_list.index("</s>")])
    else:
        return " ".join(word_list[word_list.index("<s>")+1:])



def mono_batch(path, vocabulary, batch_size, lib):
    with open(path, "r") as f:
        batch = list()
        for line in f:
            wordid_list = list()
            wordid_list.append(vocabulary.word2id["<s>"])
            for word in line.strip("\n").split():
                wordid_list.append(vocabulary.word2id[word])
            wordid_list.append(vocabulary.word2id["</s>"])
            batch.append(wordid_list)
            if len(batch) == batch_size:
                yield [Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch, fillvalue = -1)]
                batch = list()
        if len(batch) > 0:
            yield [Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch, fillvalue = -1)]

def random_sorted_parallel_batch(source_path, target_path, source_vocabulary, target_vocabulary, batch_size, pooling, lib):
    batch_list = list()
    batch = list()
    for n_pairs in generate_n_pairs(source_path, target_path, source_vocabulary, target_vocabulary, batch_size * pooling):
        for st_pair in sorted(n_pairs, key = lambda x: len(x[0]), reverse = True):
            batch.append(st_pair)
            if len(batch) == batch_size:
                batch_list.append(batch)
                batch = list()
    if len(batch) > 0:
        batch_list.append(batch)
    random.shuffle(batch_list)
    for batch in batch_list:
        batch_source = [batch[i][0] for i in range(len(batch))]
        batch_target = [batch[i][1] for i in range(len(batch))]
        yield ([Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch_source, fillvalue = -1)], [Variable(lib.array(list(y), dtype = lib.int32)) for y in zip_longest(*batch_target, fillvalue = -1)])


def random_batch_document(source_path, target_path, source_vocabulary, target_vocabulary, batch_size, context_pooling, initialize, lib):
    pool = list()
    for n_pairs in generate_document(source_path, target_path, source_vocabulary, target_vocabulary, batch_size, initialize):
        pool.append(n_pairs)
        if len(pool) == context_pooling:
            #random.shuffle(pool)
            for batch in pool:
                for batch_pair in zip(*batch):
                    batch_source, batch_target = list(zip(*batch_pair))
                    yield ([Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch_source, fillvalue = -1)], [Variable(lib.array(list(y), dtype = lib.int32)) for y in zip_longest(*batch_target, fillvalue = -1)])
            pool = list()

    if len(pool) > 0:
        random.shuffle(pool)
        for batch in pool:
            for batch_pair in list(zip(*batch)):
                batch_source, batch_target = list(zip(*batch_pair))
                yield ([Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch_source, fillvalue = -1)], [Variable(lib.array(list(y), dtype = lib.int32)) for y in zip_longest(*batch_target, fillvalue = -1)])

def generate_document(source_path, target_path, source_vocabulary, target_vocabulary, batch_size, doc_len):
    sbos = source_vocabulary.word2id["<s>"]
    tbos = target_vocabulary.word2id["<s>"]
    seos = source_vocabulary.word2id["</s>"]
    teos = target_vocabulary.word2id["</s>"]
    with open(source_path, "r") as fs, open(target_path, "r") as ft:
        n_pairs = list()
        document = list()
        for line_source, line_target in zip(fs, ft):
            wordid_source = list()
            wordid_target = list()
            wordid_source.append(sbos)
            for word in line_source.strip("\n").split():
                wordid_source.append(source_vocabulary.word2id[word])
            wordid_source.append(seos)
            wordid_target.append(tbos)
            for word in line_target.strip("\n").split():
                wordid_target.append(target_vocabulary.word2id[word])
            wordid_target.append(teos)
            document.append([wordid_source, wordid_target])
            
            if len(document) == doc_len:
                n_pairs.append(document)
                document = list()

            if len(n_pairs) == batch_size:
                yield n_pairs
                n_pairs = list()
        
        if len(document) > 0:
            while len(document) < doc_len:
                document.append([[sbos, seos], [tbos, teos]])
            n_pairs.append(document)

        if len(n_pairs) > 0:
            yield n_pairs

def batch_mono_document(source_path, source_vocabulary, batch_size, initialize, lib):
    for batch in generate_mono_document(source_path, source_vocabulary, batch_size, initialize):
        for batch_source in list(zip(*batch)):
            yield [Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch_source, fillvalue = -1)]


def generate_mono_document(source_path, source_vocabulary, batch_size, doc_len):
    sbos = source_vocabulary.word2id["<s>"]
    seos = source_vocabulary.word2id["</s>"]
    with open(source_path, "r") as fs:
        n_sents = list()
        document = list()
        for line in fs:
            wordid_source = list()
            wordid_source.append(sbos)
            for word in line.strip("\n").split():
                wordid_source.append(source_vocabulary.word2id[word])
            wordid_source.append(seos)
            document.append(wordid_source)
            
            if len(document) == doc_len:
                n_sents.append(document)
                document = list()

            if len(n_sents) == batch_size:
                yield n_sents
                n_sents = list()
        
        if len(document) > 0:
            while len(document) < doc_len:
                document.append([-1])
            n_sents.append(document)

        if len(n_sents) > 0:
            yield n_sents

def random_sorted_3parallel_batch(source_path, target_path, output_path, source_vocabulary, target_vocabulary, batch_size, pooling, lib):
    batch_list = list()
    batch = list()
    for n_pairs in generate_n_3pairs(source_path, target_path, output_path, source_vocabulary, target_vocabulary, batch_size * pooling):
        for sto_pair in sorted(n_pairs, key = lambda x: len(x[0]), reverse = True):
            batch.append(sto_pair)
            if len(batch) == batch_size:
                batch_list.append(batch)
                batch = list()
    if len(batch) > 0:
        batch_list.append(batch)
    random.shuffle(batch_list)
    for batch in batch_list:
        batch_source = [batch[i][0] for i in range(len(batch))]
        batch_target = [batch[i][1] for i in range(len(batch))]
        batch_output = [batch[i][2] for i in range(len(batch))]
        yield ([Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch_source, fillvalue = -1)], [Variable(lib.array(list(y), dtype = lib.int32)) for y in zip_longest(*batch_target, fillvalue = -1)], [Variable(lib.array(list(z), dtype = lib.int32)) for z in zip_longest(*batch_output, fillvalue = -1)])

def generate_n_pairs(source_path, target_path, source_vocabulary, target_vocabulary, n):
    with open(source_path, "r") as fs, open(target_path, "r") as ft:
        n_pairs = list()
        for line_source, line_target in zip(fs, ft):
            wordid_source = list()
            wordid_target = list()
            wordid_source.append(source_vocabulary.word2id["<s>"])
            for word in line_source.strip("\n").split():
                wordid_source.append(source_vocabulary.word2id[word])
            wordid_source.append(source_vocabulary.word2id["</s>"])
            wordid_target.append(target_vocabulary.word2id["<s>"])
            for word in line_target.strip("\n").split():
                wordid_target.append(target_vocabulary.word2id[word])
            wordid_target.append(target_vocabulary.word2id["</s>"])
            n_pairs.append([wordid_source, wordid_target])
            if len(n_pairs) == n:
                yield n_pairs
                n_pairs = list()
        if len(n_pairs) > 0:
            yield n_pairs

def generate_n_3pairs(source_path, target_path, output_path, source_vocabulary, target_vocabulary, n):
    with open(source_path, "r") as fs, open(target_path, "r") as ft, open(output_path, "r") as fo:
        n_pairs = list()
        for line_source, line_target, line_output in zip(fs, ft, fo):
            wordid_source = list()
            wordid_target = list()
            wordid_output = list()
            wordid_source.append(source_vocabulary.word2id["<s>"])
            for word in line_source.strip("\n").split():
                wordid_source.append(source_vocabulary.word2id[word])
            wordid_source.append(source_vocabulary.word2id["</s>"])
            wordid_target.append(target_vocabulary.word2id["<s>"])
            for word in line_target.strip("\n").split():
                wordid_target.append(target_vocabulary.word2id[word])
            wordid_target.append(target_vocabulary.word2id["</s>"])
            wordid_output.append(target_vocabulary.word2id["<s>"])
            for word in line_output.strip("\n").split():
                wordid_output.append(target_vocabulary.word2id[word])
            wordid_output.append(target_vocabulary.word2id["</s>"])
            n_pairs.append([wordid_source, wordid_target, wordid_output])
            if len(n_pairs) == n:
                yield n_pairs
                n_pairs = list()
        if len(n_pairs) > 0:
            yield n_pairs

def random_sorted_3parallel_batch_scoring(source_path, output_path, label_path, source_vocabulary, target_vocabulary, batch_size, pooling, lib):
    batch_list = list()
    batch = list()
    for n_pairs in generate_n_3pairs_scoring(source_path, output_path, label_path, source_vocabulary, target_vocabulary, batch_size * pooling):
        for sto_pair in sorted(n_pairs, key = lambda x: len(x[0]), reverse = True):
            batch.append(sto_pair)
            if len(batch) == batch_size:
                batch_list.append(batch)
                batch = list()
    if len(batch) > 0:
        batch_list.append(batch)
    random.shuffle(batch_list)
    for batch in batch_list:
        batch_source = [batch[i][0] for i in range(len(batch))]
        batch_output = [batch[i][1] for i in range(len(batch))]
        batch_label = [batch[i][2] for i in range(len(batch))]
        yield ([Variable(lib.array(list(x), dtype = lib.int32)) for x in zip_longest(*batch_source, fillvalue = -1)], [Variable(lib.array(list(y), dtype = lib.int32)) for y in zip_longest(*batch_output, fillvalue = -1)], Variable(lib.array(batch_label, dtype = lib.float32)))

def generate_n_3pairs_scoring(source_path, output_path, label_path, source_vocabulary, target_vocabulary, n):
    with open(source_path, "r") as fs, open(output_path, "r") as fo, open(label_path, "r") as fl:
        n_pairs = list()
        for line_source, line_output, line_label in zip(fs, fo, fl):
            wordid_source = list()
            wordid_output = list()
            wordid_source.append(source_vocabulary.word2id["<s>"])
            for word in line_source.strip("\n").split():
                wordid_source.append(source_vocabulary.word2id[word])
            wordid_source.append(source_vocabulary.word2id["</s>"])
            wordid_output.append(target_vocabulary.word2id["<s>"])
            for word in line_output.strip("\n").split():
                wordid_output.append(target_vocabulary.word2id[word])
            wordid_output.append(target_vocabulary.word2id["</s>"])
            n_pairs.append([wordid_source, wordid_output, float(line_label.strip("\n"))])
            if len(n_pairs) == n:
                yield n_pairs
                n_pairs = list()
        if len(n_pairs) > 0:
            yield n_pairs

def make_weights(initial_seed, in_size, out_size, library):
    if initial_seed:
        library.random.seed(0)
        return library.random.normal(0, library.sqrt(1. / in_size), (out_size, in_size)).astype("float32")
    else:
        return None

def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        
        if type(child) != type(dst_child) and child.name != 'decoder':
            continue
        
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        
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


def trace(*args):
	print(datetime.datetime.now(), '...', *args, file=sys.stderr)
	sys.stderr.flush()
