#python cnmt.py [method] [mode (train or dev or test)] [config_path] [best_epoch (only testing)]
import sys
from chainer import *
from networks import *
from utilities import *
from method import *
from document_utilities import *

def train(config):
    if (config.mode == 'train' and config.use_pretrain) or len(sys.argv) == 4:
        if len(sys.argv) == 4:
            trace('Start Re-Training from Epoch {} ...'.format(int(sys.argv[3])))
            start = int(sys.argv[3]) - 1
        else:
            start = 0
        trace('Loading Vocabulary ...')
        source_vocabulary = Vocabulary.load('{}.source_vocabulary'.format(config.model))
        target_vocabulary = Vocabulary.load('{}.target_vocabulary'.format(config.model))
        source_word2vec = None
        target_word2vec = None
    else:
        if config.mode == 'train':
            trace('Start Training ...')
        else:
            trace('Start Pretraining ...')
        start = 0

        if config.vocabulary_method == 'Make':
            trace('Making Vocabulary ...')
            source_vocabulary = Vocabulary.make(config.source_train, config.source_vocabulary_size)
            target_vocabulary = Vocabulary.make(config.target_train, config.target_vocabulary_size)
            source_vocabulary.save('{}.source_vocabulary'.format(config.model))
            target_vocabulary.save('{}.target_vocabulary'.format(config.model))
        else:
            trace('Loading Vocabulary ...')
            source_vocabulary = Vocabulary.load(config.source_vocabulary_file)
            target_vocabulary = Vocabulary.load(config.target_vocabulary_file)
        
        if config.word2vec_method == 'Load':
            trace('Loading Word2vec ...')
            source_word2vec = Word2vec.load(config.source_word2vec_file)
            target_word2vec = Word2vec.load(config.target_word2vec_file)
            source_word2vec.save('{}.source_word2vec'.format(config.model))
            target_word2vec.save('{}.target_word2vec'.format(config.model))
        elif config.word2vec_method == 'Make':
            trace('Making Word2vec ...')
            source_word2vec = Word2vec.make(config.source_train, config.word2vec_method, config.embed_size, config.word2vec_window_size)
            target_word2vec = Word2vec.make(config.target_train, config.word2vec_method, config.embed_size, config.word2vec_window_size)
            source_word2vec.save('{}.source_word2vec'.format(config.model))
            target_word2vec.save('{}.target_word2vec'.format(config.model))
        else:
            source_word2vec = None
            target_word2vec = None
            
    config.source_vocabulary_size = source_vocabulary.size
    config.target_vocabulary_size = target_vocabulary.size

    if config.method == 'separated_source':
        trace('Choosing Separated Source Model ... ')
        nmt = SeparatedCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, config.use_dropout, config.dropout_rate, config.use_residual, None, False, None, config.library, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec, 'source')
    elif config.method == 'separated_target':
        trace('Choosing Separated Target Model ... ')
        nmt = SeparatedCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, config.use_dropout, config.dropout_rate, config.use_residual, None, False, None, config.library, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec, 'target')
    elif config.method == 'shared_source':
        trace('Choosing Shared Source Model ... ')
        nmt = SharedSourceCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, config.use_dropout, config.dropout_rate, config.use_residual, None, False, None, config.library, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec)
    elif config.method == 'shared_target':
        trace('Choosing Shared Target Model ... ')
        nmt = SharedTargetCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, config.use_dropout, config.dropout_rate, config.use_residual, None, False, None, config.library, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec)
    elif config.method == 'shared_mix':
        trace('Choosing Shared Mix Model ... ')
        nmt = SharedMixCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, config.use_dropout, config.dropout_rate, config.use_residual, None, False, None, config.library, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec)

    if config.use_pretrain:
        trace('Loading the Weights of Pretrained Model (Epoch = {}) ...'.format(config.pretrain_epoch))
        pretrain_nmt = AttentionalNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, config.use_dropout, config.dropout_rate, config.use_residual, None, False, None, config.library, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec)
        serializers.load_npz('{}.{:03d}.pretrain_weights'.format(config.model, config.pretrain_epoch), pretrain_nmt)
        
        trace('Copying the Parameters from Pretrained Model ...')
        load_pretrain_model(pretrain_nmt, nmt)

        del pretrain_nmt

    if config.use_gpu:
        cuda.get_device(config.gpu_device).use()
        nmt.to_gpu()

    opt = config.optimizer
    opt.setup(nmt)
    opt.add_hook(optimizer.GradientClipping(config.clipping_threshold))

    if start > 0:
        trace('Loading the weights and optimizer at Epoch {} ...'.format(config.start))
        serializers.load_npz('{}.{:03d}.weights'.format(config.model, start), nmt)
        serializers.load_npz('{}.{:03d}.optimizer'.format(config.model, start), opt)

    for epoch in range(start, config.epoch):
        trace('Epoch {}/{}'.format(epoch + 1, config.epoch))
        accum_loss = 0.0
        batch_num = 0
        sample_num = 0
        random.seed(epoch)

        document_num = 0
        for document_source, document_target in make_train_batch(config.source_train, source_vocabulary, config.target_train, target_vocabulary, config.batch_size):
            batch_num += 1
            nmt.init_context_state(len(document_source[0]))
            result_predict = list()

            sentence_num = 0
            for batch_source, batch_target in zip(document_source, document_target):
                nmt.cleargrads()
                sentence_num += 1

                embed_source = [Variable(config.library.array(list(x), dtype=config.library.int32)) for x in zip_longest(*batch_source, fillvalue=-1)]
                embed_target = [Variable(config.library.array(list(x), dtype=config.library.int32)) for x in zip_longest(*batch_target, fillvalue=-1)]
                loss, batch_predict = nmt(embed_source, embed_target)
                accum_loss += loss.data * batch_predict[0].shape[0]
                trace('Batch: {}, Sentence: {}, Loss: {}'.format(batch_num, sentence_num, loss.data))
                loss.backward()
                loss.unchain_backward()
                opt.update()

                result_predict.append(list(cuda.to_cpu(functions.transpose(functions.vstack(batch_predict)).data)))

            if config.make_summarized_log is False:
                for i in range(len(batch_source)):
                    document_num += 1
                    trace('Document ID: {}'.format(document_num))
                    for j in range(len(result_predict)): 
                        source_sentence = make_sentence(document_source[j][i], source_vocabulary)
                        if source_sentence == 'NULL':
                            break
                        target_sentence = make_sentence(document_target[j][i], target_vocabulary)
                        predict_sentence = make_sentence(result_predict[j][i], target_vocabulary)

                        sample_num += 1
                        trace('Sample : {}'.format(sample_num))
                        trace('Source : {}'.format(source_sentence))
                        trace('Target : {}'.format(target_sentence))
                        trace('Predict: {}'.format(predict_sentence))

        trace('Average_loss: {}'.format(accum_loss / sample_num))
        trace('Saving Model ...')
        model = '{}.{:03d}'.format(config.model, epoch + 1)
        serializers.save_npz('{}.weights'.format(model), nmt)
        serializers.save_npz('{}.optimizer'.format(model), opt)

    trace('Finished.')


def test(config):
    trace('Loading Vocabulary ...')
    source_vocabulary = Vocabulary.load('{}.source_vocabulary'.format(config.model))
    target_vocabulary = Vocabulary.load('{}.target_vocabulary'.format(config.model))
    config.source_vocabulary_size = source_vocabulary.size
    config.target_vocabulary_size = target_vocabulary.size

    trace('Loading Model ...')

    if config.method == 'separated_source':
        trace('Choosing Separated Source Model ... ')
        nmt = SeparatedCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, False, 0.0, config.use_residual, config.generation_limit, config.use_beamsearch, config.beam_size, config.library, source_vocabulary, target_vocabulary, None, None, 'source')
    elif config.method == 'separated_target':
        trace('Choosing Separated Target Model ... ')
        nmt = SeparatedCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, False, 0.0, config.use_residual, config.generation_limit, config.use_beamsearch, config.beam_size, config.library, source_vocabulary, target_vocabulary, None, None, 'target')
    elif config.method == 'shared_source':
        trace('Choosing Shared Source Model ... ')
        nmt = SharedSourceCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, False, 0.0, config.use_residual, config.generation_limit, config.use_beamsearch, config.beam_size, config.library, source_vocabulary, target_vocabulary, None, None)
    elif config.method == 'shared_target':
        trace('Choosing Shared Target Model ... ')
        nmt = SharedTargetCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, False, 0.0, config.use_residual, config.generation_limit, config.use_beamsearch, config.beam_size, config.library, source_vocabulary, target_vocabulary, None, None)
    elif config.method == 'shared_mix':
        trace('Choosing Shared Mix Model ... ')
        nmt = SharedMixCNMT(config.source_vocabulary_size, config.target_vocabulary_size, config.layer_size, config.embed_size, config.hidden_size, config.bilstm_method, config.attention_method, config.activation_method, False, 0.0, config.use_residual, config.generation_limit, config.use_beamsearch, config.beam_size, config.library, source_vocabulary, target_vocabulary, None, None)

    serializers.load_npz('{}.{:03d}.weights'.format(config.model, config.model_number), nmt)
        
    if config.use_gpu:
        cuda.get_device(config.gpu_device).use()
        nmt.to_gpu()
    

    trace('Translating ...')
    sample_num = 0
    
    with open(config.predict_file, 'w') as wf:
        batch_num = 0
        document_num = 0
        for document_source in make_test_data(config.source_file, source_vocabulary, config.batch_size):
            batch_num += 1
            nmt.init_context_state(len(document_source[0]))
            result_predict = list()

            sentence_num = 0
            for batch_source in document_source:
                nmt.cleargrads()
                sentence_num += 1
                trace('Batch: {}, Sentence: {}'.format(batch_num, sentence_num))

                embed_source = [Variable(config.library.array(list(x), dtype=config.library.int32)) for x in zip_longest(*batch_source, fillvalue=-1)]
                _, batch_predict = nmt(embed_source, None)

                result_predict.append(list(cuda.to_cpu(functions.transpose(functions.vstack(batch_predict)).data)))

            for i in range(len(batch_source)):
                document_num += 1
                trace('Document ID: {}'.format(document_num))
                for j in range(len(result_predict)): 
                    source_sentence = make_sentence(document_source[j][i], source_vocabulary)
                    if source_sentence == 'NULL':
                        break
                    predict_sentence = make_sentence(result_predict[j][i], target_vocabulary)

                    sample_num += 1
                    trace('Writing the sample {} ...'.format(sample_num))
                    wf.write('{}\n'.format(predict_sentence))
                
                wf.write('\n')


if __name__ == '__main__':
    config = Configuration(sys.argv[1], sys.argv[2])
    if config.mode == 'train':
        train(config)

    elif config.mode == 'test':
        config.model_number = int(sys.argv[3])
        trace('Start Testing ...')
        config.source_file = config.source_test
        if config.use_beamsearch:
            config.predict_file = '{}.{:03d}.test_result.beam{}'.format(config.model, config.model_number, config.beam_size)
        else:
            config.predict_file = '{}.{:03d}.test_result'.format(config.model, config.model_number)

        test(config)

        trace('Finished.')

    elif config.mode == 'dev':
        trace('Start Validation ...')
        config.source_file = config.source_dev
        model = config.model
        if len(sys.argv) == 5:
            start = int(sys.argv[3]) - 1
            end = int(sys.argv[4])
        elif len(sys.argv) == 4:
            start = int(sys.argv[3]) - 1
            end = config.epoch
        else:
            start = 0
            end = config.epoch

        for i in range(start, end):
            config.model_number = i + 1
            trace('Epoch {}/{}'.format(i + 1, config.epoch))
            config.predict_file = '{}.{:03d}.dev_result'.format(config.model, config.model_number)

            test(config)
        trace('Finished.')
