from chainer import *

import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import *
from sklearn.manifold import TSNE

class NetworkFunctions():
    def activate(self, x):
        if self.activation_method == "tanh":
            return functions.tanh(x)
        if self.activation_method == "sigmoid":
            return functions.sigmoid(x)
        if self.activation_method == "relu":
            return functions.relu(x)
        elif self.activation_method == "None":
            return x
    
    def dropout(self, x):
        return functions.dropout(x, ratio = self.dropout_rate) if self.use_dropout else x

    def residual(self, x, previous_x):
        return x + previous_x if self.use_residual and x.shape == previous_x.shape else x
    
    @staticmethod
    def stack_list_to_axis_1(x_list):
        return functions.stack(x_list, axis = 1)
   
    @staticmethod
    def separate_list_from_axis_1(xs):
        return functions.separate(xs, axis = 1)

    @staticmethod
    def variable_axis_into_batch_axis_for_1d(xs):
        batch_size, variable_size = xs.shape
        x = functions.reshape(xs, (batch_size * variable_size,))
        return x, batch_size, variable_size 
 
    @staticmethod
    def variable_axis_into_batch_axis_for_2d(xs):
        batch_size, variable_size, vector_size = xs.shape
        x = functions.reshape(xs, (batch_size * variable_size, vector_size))
        return x, batch_size, variable_size
    
    @staticmethod
    def variable_axis_from_batch_axis_for_2d(xs, batch_size, variable_size):
        vector_size = xs.shape[1]
        return functions.reshape(xs, (batch_size, variable_size, vector_size))

    def stack_variable_list_to_batch_axis_for_1d(self, x_list):
        #list of (batch_size,) → (batch_size * list_size, )
        stacked = self.stack_list_to_axis_1(x_list)
        return self.variable_axis_into_batch_axis_for_1d(stacked)

    def stack_variable_list_to_batch_axis_for_2d(self, x_list):
        #list of (batch_size, vector_size) → (batch_size * list_size, vector_size)
        stacked = self.stack_list_to_axis_1(x_list)
        return self.variable_axis_into_batch_axis_for_2d(stacked)

    def separate_variable_list_from_batch_axis_for_2d(self, xs, batch_size, list_size):
        #(batch_size * list_size, vector_size) → list of (batch_size, vector_size)
        changed = self.variable_axis_from_batch_axis_for_2d(xs, batch_size, list_size)
        return self.separate_list_from_axis_1(changed)

    def mask_output(self, inputs, outputs):
        return functions.where(functions.broadcast_to(functions.reshape(self.library.any(inputs.data != 0.0, axis = 1), (inputs.shape[0], 1)), outputs.shape), outputs, self.library.zeros(outputs.shape, dtype = self.library.float32))

########################################################
#Base Networks

class FeedForwardNetwork(Chain, NetworkFunctions):
    def __init__(self, input_size, output_size, activation_method, use_dropout, dropout_rate, use_residual, no_bias, library):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.no_bias = no_bias
        self.library = library
        super(FeedForwardNetwork, self).__init__(
            network = links.Linear(input_size, output_size, nobias = no_bias),
        )

    def __call__(self, x):
        #hidden = self.network(x)
        hidden = self.mask_output(x, self.network(x))
        return self.dropout(self.residual(self.activate(hidden), x))

class NlayerFeedForwardNetwork(ChainList):
    def __init__(self, layer_size, input_size, hidden_size, output_size, activation_method, use_dropout, dropout_rate, use_residual, no_bias, library):
        self.layer_size = layer_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.no_bias = no_bias
        self.library = library
        layers = list()
        for i in range(layer_size):
            if i == 0:
                layer_input_size = input_size
                layer_output_size = hidden_size
            elif i == layer_size - 1:
                layer_input_size = hidden_size
                layer_output_size = output_size
            else:
                layer_input_size = hidden_size
                layer_output_size = hidden_size
            layers.append(FeedForwardNetwork(layer_input_size, layer_output_size, activation_method, use_dropout, dropout_rate, use_residual, no_bias, library))
        super(NlayerFeedForwardNetwork, self).__init__(*layers)

    def __call__(self, x):
        for i, layer in enumerate(self.children()):
            hidden = layer(hidden if i != 0 else x)
        return hidden

class WordEmbedding(Chain, NetworkFunctions):
    def __init__(self, vocabulary_size, embed_size, activation_method, use_dropout, dropout_rate, vocabulary, word2vec):
        self.vocabulary_size = vocabulary_size
        self.embed_size = embed_size
        self.use_dropout = use_dropout
        self.activation_method = activation_method
        self.dropout_rate = dropout_rate
        self.vocabulary = vocabulary
        super(WordEmbedding, self).__init__(
            word2embed = links.EmbedID(vocabulary_size, embed_size, ignore_label = -1),
        )
        if word2vec is not None:
            for i in range(vocabulary_size):
                word = vocabulary.id2word[i]
                if word in word2vec.model:
                    self.word2embed.W.data[i] = word2vec.model[word]

    def __call__(self, x):
        return self.dropout(self.activate(self.word2embed(x)))

    def visualization(self, path, start, end):
        words = [self.vocabulary.id2word[i] for i in range(start - 1, end)]
        vectors = [self.word2embed.W.data[i] for i in range(start - 1, end)]

        fp = font_manager.FontProperties(fname='/usr/share/fonts/truetype/fonts-japanese-gothic.ttf')
        tsne = TSNE(init = "pca", random_state = 1)
        new_vectors = tsne.fit_transform(vectors)
        fig, ax = pyplot.subplots()
        ax.scatter(new_vectors[:, 0], new_vectors[:, 1], s=5)
        for index, label in enumerate(words):
            ax.annotate(label, xy=(new_vectors[index, 0], new_vectors[index, 1]), fontproperties = fp, fontsize=8)
        pyplot.savefig(path)

class UniLSTM(Chain, NetworkFunctions):
    def __init__(self, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        super(UniLSTM, self).__init__(
            uniLSTM = links.LSTM(input_size, hidden_size),
        )

    def __call__(self, x):
        prev_c, prev_h = self.get_state()
        #hidden = self.uniLSTM(x)
        hidden = self.mask_output(x, self.uniLSTM(x))
        if prev_c is not None and prev_h is not None:
            c, h = self.get_state()
            self.set_state(functions.where(hidden.data != 0.0, c, prev_c), functions.where(hidden.data != 0.0, h, prev_h))
        return self.dropout(self.residual(self.activate(hidden), x))

    def get_state(self):
        return self.uniLSTM.c, self.uniLSTM.h
    
    def set_state(self, c, h):
        self.uniLSTM.set_state(c, h)
    
    def reset_state(self):
        self.uniLSTM.reset_state()
    
class NlayerUniLSTM(ChainList):
    def __init__(self, layer_size, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.layer_size = layer_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.activation_method = activation_method
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        layers = list()
        for i in range(layer_size):
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size
            layers.append(UniLSTM(layer_input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library))
        super(NlayerUniLSTM, self).__init__(*layers)
        
    def __call__(self, x):
        for i, layer in enumerate(self.children()):
            hidden = layer(hidden if i != 0 else x)
        return hidden

    def get_state(self):
        states = list()
        for layer in self.children():
            states.append(layer.get_state())
        return states
    
    def set_state(self, ch_list):
        for layer, (c, h) in zip(self.children(), ch_list):
            layer.set_state(c, h)
    
    def reset_state(self):
        for layer in self.children():
            layer.reset_state()

class ConcatBiLSTM(Chain, NetworkFunctions):
    def __init__(self, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.input_size = input_size
        self.backward_hidden_size = int(hidden_size / 2)
        self.forward_hidden_size = hidden_size - self.backward_hidden_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        super(ConcatBiLSTM, self).__init__(
            forwardLSTM = links.LSTM(input_size, self.forward_hidden_size),
            backwardLSTM = links.LSTM(input_size, self.backward_hidden_size),
        )

    def __call__(self, x_list):
        backward_hidden_states = list()
        hidden_states = list()
        for x_backward in x_list[::-1]:
            prev_c = self.backwardLSTM.c
            prev_h = self.backwardLSTM.h
            backward_hidden = self.mask_output(x_backward, self.backwardLSTM(x_backward))
            if prev_c is not None and prev_h is not None:
                c = self.backwardLSTM.c
                h = self.backwardLSTM.h
                self.backwardLSTM.set_state(functions.where(backward_hidden.data != 0.0, c, prev_c), functions.where(backward_hidden.data != 0.0, h, prev_h))
            backward_hidden_states.insert(0, backward_hidden)
        for x_forward, backward_hidden in zip(x_list, backward_hidden_states): 
            prev_c = self.forwardLSTM.c
            prev_h = self.forwardLSTM.h
            forward_hidden = self.mask_output(x_forward, self.forwardLSTM(x_forward))
            if prev_c is not None and prev_h is not None:
                c = self.forwardLSTM.c
                h = self.forwardLSTM.h
                self.forwardLSTM.set_state(functions.where(forward_hidden.data != 0.0, c, prev_c), functions.where(forward_hidden.data != 0.0, h, prev_h))
            hidden_states.append(self.dropout(self.residual(self.activate(functions.concat((forward_hidden, backward_hidden))), x_forward)))
        return hidden_states

    def get_state(self):
        return functions.concat((self.forwardLSTM.c, self.backwardLSTM.c)), functions.concat((self.forwardLSTM.h, self.backwardLSTM.h))
    
    def set_state(self, c, h):
        forward_c, backward_c = functions.split_axis(c, 2, axis = 1)
        forward_h, backward_h = functions.split_axis(h, 2, axis = 1)
        self.forwardLSTM.set_state(forward_c, forward_h)
        self.backwardLSTM.set_state(backward_c, backward_h)
    
    def reset_state(self):
        self.forwardLSTM.reset_state()
        self.backwardLSTM.reset_state()

class AddBiLSTM(Chain, NetworkFunctions):
    def __init__(self, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        super(AddBiLSTM, self).__init__(
            forwardLSTM = links.LSTM(input_size, hidden_size),
            backwardLSTM = links.LSTM(input_size, hidden_size),
        )

    def __call__(self, x_list):
        backward_hidden_states = list()
        hidden_states = list()
        for x_backward in x_list[::-1]:
            prev_c = self.backwardLSTM.c
            prev_h = self.backwardLSTM.h
            #backward_hidden = self.backwardLSTM(x_backward)
            backward_hidden = self.mask_output(x_backward, self.backwardLSTM(x_backward))
            if prev_c is not None and prev_h is not None:
                c = self.backwardLSTM.c
                h = self.backwardLSTM.h
                self.backwardLSTM.set_state(functions.where(backward_hidden.data != 0.0, c, prev_c), functions.where(backward_hidden.data != 0.0, h, prev_h))
            backward_hidden_states.insert(0, backward_hidden)
        for x_forward, backward_hidden in zip(x_list, backward_hidden_states): 
            prev_c = self.forwardLSTM.c
            prev_h = self.forwardLSTM.h
            #forward_hidden = self.forwardLSTM(x_forward)
            forward_hidden = self.mask_output(x_forward, self.forwardLSTM(x_forward))
            if prev_c is not None and prev_h is not None:
                c = self.forwardLSTM.c
                h = self.forwardLSTM.h
                self.forwardLSTM.set_state(functions.where(forward_hidden.data != 0.0, c, prev_c), functions.where(forward_hidden.data != 0.0, h, prev_h))
            hidden_states.append(self.dropout(self.residual(self.activate(forward_hidden + backward_hidden), x_forward)))
        return hidden_states

    def get_state(self):
        return self.forwardLSTM.c + self.backwardLSTM.c, self.forwardLSTM.h + self.backwardLSTM.h
    
    def set_state(self, c, h):
        self.forwardLSTM.set_state(c, h)
        self.backwardLSTM.set_state(c, h)
    
    def reset_state(self):
        self.forwardLSTM.reset_state()
        self.backwardLSTM.reset_state()

class NlayerBiLSTM(ChainList):
    def __init__(self, bilstm_method, layer_size, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.bilstm_method = bilstm_method
        self.layer_size = layer_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.activation_method = activation_method
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        layers = list()
        for i in range(layer_size):
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size
            layers.append(self.select_bilstm(layer_input_size))
        super(NlayerBiLSTM, self).__init__(*layers)
        
    def __call__(self, x_list):
        for i, layer in enumerate(self.children()):
            hidden_states = layer(hidden_states if i != 0 else x_list)
        return hidden_states

    def get_state(self):
        states = list()
        for layer in self.children():
            states.append(layer.get_state())
        return states
    
    def set_state(self, ch_list):
        for layer, (c, h) in zip(self.children(), ch_list):
            layer.set_state(c, h)
    
    def reset_state(self):
        for layer in self.children():
            layer.reset_state()

    def select_bilstm(self, input_size):
        if self.bilstm_method == "Concat":
            return ConcatBiLSTM(input_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        elif self.bilstm_method == "Add":
            return AddBiLSTM(input_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)

class NlayerFinalConcatBiLSTM(Chain):
    def __init__(self, layer_size, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.layer_size = layer_size
        self.input_size = input_size
        self.backward_hidden_size = int(hidden_size / 2)
        self.forward_hidden_size = hidden_size - self.backward_hidden_size
        self.use_dropout = use_dropout
        self.activation_method = activation_method
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        super(NlayerFinalConcatBiLSTM, self).__init__(
            forwardLSTM = NlayerUniLSTM(layer_size, input_size, self.forward_hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library),
            backwardLSTM = NlayerUniLSTM(layer_size, input_size, self.backward_hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library),
        )
        
    def __call__(self, x_list):
        backward_hidden_states = list()
        hidden_states = list()
        for x_backward in x_list[::-1]:
            backward_hidden = self.backwardLSTM(x_backward)
            backward_hidden_states.insert(0, backward_hidden)
        for x_forward, backward_hidden in zip(x_list, backward_hidden_states): 
            forward_hidden = self.forwardLSTM(x_forward)
            hidden_states.append(functions.concat((forward_hidden, backward_hidden)))
        return hidden_states

    def get_state(self):
        states = list()
        for (forward_c, forward_h), (backward_c, backward_h) in zip(self.forwardLSTM.get_state(), self.backwardLSTM.get_state()):
            c = functions.concat((forward_c, backward_c))
            h = functions.concat((forward_h, backward_h))
            states.append((c, h))
        return states
    
    def set_state(self, ch_list):
        self.forwardLSTM.set_state(ch_list)
        self.backwardLSTM.set_state(ch_list)
    
    def reset_state(self):
        self.forwardLSTM.reset_state()
        self.backwardLSTM.reset_state()

class NlayerFinalAddBiLSTM(Chain):
    def __init__(self, layer_size, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.layer_size = layer_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.activation_method = activation_method
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        super(NlayerFinalAddBiLSTM, self).__init__(
            forwardLSTM = NlayerUniLSTM(layer_size, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library),
            backwardLSTM = NlayerUniLSTM(layer_size, input_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library),
        )
        
    def __call__(self, x_list):
        backward_hidden_states = list()
        hidden_states = list()
        for x_backward in x_list[::-1]:
            backward_hidden = self.backwardLSTM(x_backward)
            backward_hidden_states.insert(0, backward_hidden)
        for x_forward, backward_hidden in zip(x_list, backward_hidden_states): 
            forward_hidden = self.forwardLSTM(x_forward)
            hidden_states.append(forward_hidden + backward_hidden)
        return hidden_states

    def get_state(self):
        states = list()
        for (forward_c, forward_h), (backward_c, backward_h) in zip(self.forwardLSTM.get_state(), self.backwardLSTM.get_state()):
            c = forward_c + backward_c
            h = forward_h + backward_h
            states.append((c, h))
        return states
    
    def set_state(self, ch_list):
        self.forwardLSTM.set_state(ch_list)
        self.backwardLSTM.set_state(ch_list)
    
    def reset_state(self):
        self.forwardLSTM.reset_state()
        self.backwardLSTM.reset_state()

########################################################
#TBA
#class CNN(Chain, NetworkFunctions):
#class NlayerCNN(ChainList):
########################################################

########################################################
#RNN Encoder-Decoder Networks

class RNNEncoder(Chain, NetworkFunctions):
    def __init__(self, bilstm_method, layer_size, source_vocabulary_size, embed_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library, source_vocabulary, source_word2vec):
        self.bilstm_method = bilstm_method
        self.layer_size = layer_size
        self.vocabulary_size = source_vocabulary_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.vocabulary = source_vocabulary
        self.library = library
        super(RNNEncoder, self).__init__(
            embedding = WordEmbedding(source_vocabulary_size, embed_size, activation_method, use_dropout, dropout_rate, source_vocabulary, source_word2vec),
            lstm = self.select_bilstm(),
        )

    def __call__(self, sentence):
        return self.forward(sentence)[:2]

    def forward(self, sentence):
        xs, batch_size, sentence_length = self.stack_variable_list_to_batch_axis_for_1d(sentence)
        embed_states = self.separate_variable_list_from_batch_axis_for_2d(self.embedding(xs), batch_size, sentence_length)
        hidden_states = self.lstm(embed_states)
        lstm_states = self.get_state()
        return hidden_states, lstm_states, embed_states
    
    def get_state(self):
        return self.lstm.get_state()

    def reset_state(self):
        self.lstm.reset_state()
    
    def select_bilstm(self):
        if self.bilstm_method == "FinalConcat":
            return NlayerFinalConcatBiLSTM(self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        elif self.bilstm_method == "FinalAdd":
            return NlayerFinalAddBiLSTM(self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        
        else:
            return NlayerBiLSTM(self.bilstm_method, self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)

class RNNDecoder(Chain, NetworkFunctions):
    def __init__(self, layer_size, target_vocabulary_size, embed_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, attention_method, generation_limit, use_beamsearch, beam_size, library, target_vocabulary, target_word2vec):
        self.layer_size = layer_size
        self.vocabulary_size = target_vocabulary_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.attention_method = attention_method
        self.generation_limit = generation_limit
        self.use_beamsearch = use_beamsearch
        self.beam_size = beam_size
        self.library = library
        self.vocabulary = target_vocabulary
        super(RNNDecoder, self).__init__(
            embedding = WordEmbedding(target_vocabulary_size, embed_size, activation_method, use_dropout, dropout_rate, target_vocabulary, target_word2vec),
            lstm = NlayerUniLSTM(layer_size, embed_size + hidden_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, self.library),
            attention = self.select_attention(),
            tilde = FeedForwardNetwork(2 * hidden_size, hidden_size, activation_method, use_dropout, dropout_rate, False, False, library),
            output = FeedForwardNetwork(hidden_size, target_vocabulary_size, "None", False, 0.0, False, False, library),
        )

    def __call__(self, encoder_hidden_states, encoder_lstm_states, sentence):
        return self.forward(encoder_hidden_states, encoder_lstm_states, sentence)[:2]

    def forward(self, encoder_hidden_states, encoder_lstm_states, sentence):
        predicts = list()
        target_embed_states = list()
        predict_embed_states = list()
        hidden_states = list()
        attention_weights_matrix = list()

        if sentence is not None:
            loss = Variable(self.library.zeros((), dtype = self.library.float32))
            predicts.append(sentence[0])
            hidden_tilde = Variable(self.library.zeros((sentence[0].shape[0], self.hidden_size), dtype = self.library.float32))
            xs, batch_size, sentence_length = self.stack_variable_list_to_batch_axis_for_1d(sentence)
            target_embed_states = self.separate_variable_list_from_batch_axis_for_2d(self.embedding(xs), batch_size, sentence_length)
            self.set_state(self.stack_list_to_axis_1(encoder_hidden_states), encoder_lstm_states, None)
            for i, (previous_embed, correct_word) in enumerate(zip(target_embed_states, sentence[1:])):
                score, hidden_tilde, _ = self.decode_one_step(previous_embed, hidden_tilde)
                hidden_states.append(hidden_tilde)
                predict = functions.argmax(score, axis = 1)
                loss += functions.softmax_cross_entropy(score, correct_word, ignore_label = -1)
                predicts.append(functions.where(correct_word.data != -1, predict, correct_word))
            xs, batch_size, sentence_length = self.stack_variable_list_to_batch_axis_for_1d(predicts)
            predict_embed_states = self.separate_variable_list_from_batch_axis_for_2d(self.embedding(xs), batch_size, sentence_length)
            lstm_states = self.get_state()
            return loss, predicts, hidden_states, lstm_states, target_embed_states, predict_embed_states, None

        elif not self.use_beamsearch:
            batch_size = encoder_hidden_states[0].shape[0]
            predict = Variable(self.library.array([self.vocabulary.word2id["<s>"]] * batch_size, dtype = self.library.int32)) 
            predicts.append(predict)
            hidden_tilde = Variable(self.library.zeros((batch_size, self.hidden_size), dtype = self.library.float32))
            self.set_state(self.stack_list_to_axis_1(encoder_hidden_states), encoder_lstm_states, None)
            while len(predicts) - 1 < self.generation_limit:
                previous_embed = self.embedding(predict)
                predict_embed_states.append(previous_embed)
                score, hidden_tilde, attention_weights = self.decode_one_step(previous_embed, hidden_tilde)
                attention_weights_matrix.append(attention_weights)
                hidden_states.append(hidden_tilde)
                predict = functions.argmax(score, axis = 1)
                predicts.append(predict)
                if batch_size == 1 and predict.data[0] == self.vocabulary.word2id["</s>"]:
                    break
            predict_embed_states.append(self.embedding(predict))

            if batch_size == 1 and predict.data[0] != self.vocabulary.word2id["</s>"]:
                eos = Variable(self.library.array([self.vocabulary.word2id["</s>"]], dtype = self.library.int32))
                eos_embed = self.embedding(eos)
                predicts.append(eos)
                predict_embed_states.append(eos_embed)
        
            lstm_states = self.get_state()
            attention_weights_matrix = self.stack_list_to_axis_1(attention_weights_matrix)
            return None, predicts, hidden_states, lstm_states, None, predict_embed_states, attention_weights_matrix
        
        else:
            bos = Variable(self.library.array([1], dtype = self.library.int32))
            hidden_tilde = Variable(self.library.zeros((1, self.hidden_size), dtype = self.library.float32))
            initial_beam = [(0, [bos], encoder_hidden_states, encoder_lstm_states, [hidden_tilde], None, None, [self.embedding(bos)], list(), None)]
            for _, predicts, _, _, hidden_states, lstm_states, _, predict_embed_states, attention_weights_matrix, _ in sorted(self.beam_search(initial_beam), key = lambda x: self.beam_search_normalization(x[0].data, len(x[1])), reverse = True):
                attention_weights_matrix = functions.stack(attention_weights_matrix, axis = 1)
                break
            return None, predicts, hidden_states, lstm_states, None, predict_embed_states, attention_weights_matrix

    def decode_one_step(self, previous_embed, previous_hidden_tilde):
        hidden = self.lstm(functions.concat((previous_embed, previous_hidden_tilde)))
        attention, attention_weights = self.attention(hidden)
        hidden_tilde = self.tilde(functions.concat((attention, hidden)))
        score = self.output(hidden_tilde)
        return score, hidden_tilde, attention_weights

    def beam_search(self, initial_beam):
        beam = [0] * self.generation_limit
        for i in range(self.generation_limit):
            beam[i] = list()
            if i == 0:
                new_beam = list()
                for logprob, predicts, encoder_hidden_states, encoder_lstm_states, decoder_hidden_states, decoder_lstm_states, decoder_attention_states, predict_embed_states, attention_weights_matrix, finished in initial_beam:
                    self.set_state(functions.stack(encoder_hidden_states, axis = 1), encoder_lstm_states, decoder_attention_states)
                    previous_embed = self.embedding(predicts[-1])
                    score, hidden_tilde, attention_weights = self.decode_one_step(previous_embed, decoder_hidden_states[-1])
                    prob = functions.softmax(score)
                    lstm_states, attention_states = self.get_state()
                    for predict in self.library.argsort(prob.data[0])[-1:-self.beam_size-1:-1]:
                        predict_variable = Variable(self.library.array([predict], dtype = self.library.int32))
                        new_beam.append((logprob + functions.log(prob[0][predict]), predicts + [predict_variable], encoder_hidden_states, encoder_lstm_states, decoder_hidden_states + [hidden_tilde], lstm_states, attention_states, predict_embed_states + [functions.tanh(self.embedding(predict_variable))], attention_weights_matrix + [functions.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[1]))], True if predict == 2 else False)) 
            else:
                new_beam = list()
                for logprob, predicts, encoder_hidden_states, encoder_lstm_states, decoder_hidden_states, decoder_lstm_states, decoder_attention_states, predict_embed_states, attention_weights_matrix, finished in beam[i - 1]:
                    if finished is not True:
                        self.set_state(functions.stack(encoder_hidden_states, axis = 1), decoder_lstm_states, decoder_attention_states)
                        previous_embed = self.embedding(predicts[-1])
                        score, hidden_tilde, attention_weights = self.decode_one_step(previous_embed, decoder_hidden_states[-1])
                        prob = functions.softmax(score)
                        lstm_states, attention_states = self.get_state()
                        for predict in self.library.argsort(prob.data[0])[-1:-self.beam_size-1:-1]:
                            predict_variable = Variable(self.library.array([predict], dtype = self.library.int32))
                            new_beam.append((logprob + functions.log(prob[0][predict]), predicts + [predict_variable], encoder_hidden_states, encoder_lstm_states, decoder_hidden_states + [hidden_tilde], lstm_states, attention_states, predict_embed_states + [functions.tanh(self.embedding(predict_variable))], attention_weights_matrix + [functions.reshape(attention_weights, (attention_weights.shape[0], attention_weights.shape[1]))], True if predict == 2 else False)) 
                    else:
                        new_beam.append((logprob, predicts, encoder_hidden_states, encoder_lstm_states, decoder_hidden_states, decoder_lstm_states, decoder_attention_states, predict_embed_states, attention_weights_matrix, finished))
            for _, items in zip(range(self.beam_size), sorted(new_beam, key = lambda x: self.beam_search_normalization(x[0].data, len(x[1])), reverse = True)):
                beam[i].append(items)
        return beam[-1]

    '''
    def beam_search_new(self, initial_beam):
        candidates = list()
        for i in range(self.generation_limit + 1):
            previous_beam = initial_beam if i == 0 else new_beam
            n_previous_word = functions.concat((items[1][-1] for items in previous_beam), axis = 0)
            n_previous_hidden_tilde = functions.concat((items[4][-1] for items in previous_beam), axis = 0)
            n_encoder_hidden_states = functions.concat((functions.stack(items[2], axis = 1) for items in previous_beam), axis = 0)
            n_lstm_states = list()
            for j in range(self.layer_size):
                n_lstm_states.append((functions.concat((items[3 if i == 0 else 5][j][0] for items in previous_beam), axis = 0), functions.concat((items[3 if i == 0 else 5][j][1] for items in previous_beam), axis = 0)))
            n_attention_states = (functions.concat((items[6][0] for items in previous_beam), axis = 0), functions.concat((items[6][1] for items in previous_beam), axis = 0), functions.concat((items[6][2] for items in previous_beam), axis = 0)) if self.attention_method == "Tu" and i != 0 else None
            
            self.set_state(n_encoder_hidden_states, n_lstm_states, n_attention_states)
            n_previous_embed = self.embedding(n_previous_word)
            n_score, n_hidden_tilde, n_attention_weights = self.decode_one_step(n_previous_embed, n_previous_hidden_tilde)
            n_prob = functions.softmax(n_score)
            n_lstm_states, n_attention_states = self.get_state()

            hidden_tilde_tuple = functions.separate(n_hidden_tilde, axis = 0)
            attention_weights_tuple = functions.separate(n_attention_weights, axis = 0)
            lstm_states_list = [list()] * len(previous_beam)
            for n_cell, n_hidden in n_lstm_states:
                cell_tuple = functions.separate(n_cell, axis = 0)
                hidden_tuple = functions.separate(n_hidden, axis = 0)
                for k, (cell, hidden) in enumerate(zip(cell_tuple, hidden_tuple)):
                    lstm_states_list[k].append((functions.reshape(cell, (1, cell.shape[0])), functions.reshape(hidden, (1, hidden.shape[0]))))
            if self.attention_method == "Tu":
                attention_states_list = list()
                attention_cell_tuple = functions.separate(n_attention_states[0], axis = 0)
                attention_hidden_tuple = functions.separate(n_attention_states[1], axis = 0)
                attention_coverage_tuple = functions.separate(n_attention_states[2], axis = 0)
                for cell, hidden, coverage in zip(attention_cell_tuple, attention_hidden_tuple, attention_coverage_tuple):
                    attention_states_list.append((functions.reshape(cell, (1, cell.shape[0], cell.shape[1])), functions.reshape(hidden, (1, hidden.shape[0], hidden.shape[1])), functions.reshape(coverage, (1, coverage.shape[0], coverage.shape[1]))))
            else:
                attention_states_list = None
            
            now_beam = list()
            for j, prob in enumerate(n_prob.data):
                for predict_index in self.library.argsort(prob)[-1:-self.beam_size-1:-1]:
                    predict = Variable(self.library.array([predict_index], dtype = self.library.int32))
                    log_prob = previous_beam[j][0] + functions.log(prob[predict_index])
                    sentence = previous_beam[j][1] + [predict]
                    encoder_hidden_states = previous_beam[j][2]
                    encoder_lstm_states = previous_beam[j][3]
                    decoder_hidden_states = previous_beam[j][4] + [functions.reshape(hidden_tilde_tuple[j], (1, hidden_tilde_tuple[j].shape[0]))]
                    decoder_lstm_states = lstm_states_list[j]
                    decoder_attention_states = attention_states_list[j] if self.attention_method == "Original" else None
                    predict_embed_states = previous_beam[j][7] + [self.embedding(predict)]
                    attention_weights_matrix = previous_beam[j][8] + [functions.reshape(attention_weights_tuple[j], (1, attention_weights_tuple[j].shape[0]))]
                    finished = True if predict_index == self.vocabulary.word2id["</s>"] else False
                    now_beam.append((log_prob, sentence, encoder_hidden_states, encoder_lstm_states, decoder_hidden_states, decoder_lstm_states, decoder_attention_states, predict_embed_states, attention_weights_matrix, finished))

            new_beam = list()
            now_finish = list()
            for _, items in zip(range(self.beam_size), sorted(now_beam + candidates, key = lambda x: self.beam_search_normalization(x[0].data, len(x[1])), reverse = True)):
                if items[9] == True or i == self.generation_limit:
                    now_finish.append(items)
                else:
                    new_beam.append(items)
            candidates = now_finish[:]
            
            if len(candidates) == self.beam_size:
                break

        return candidates
    '''

    def beam_search_normalization(self, score, length):
        return score / length

    def select_attention(self):
        if self.attention_method == "Tu":
            return AttentionTu(self.hidden_size, 10, self.library)
        elif self.attention_method == "Bahdanau":
            return AttentionBahdanau(self.hidden_size)
        elif self.attention_method == "LuongDot":
            return AttentionLuongDot()
        elif self.attention_method == "LuongGeneral":
            return AttentionLuongGeneral(self.hidden_size)
        elif self.attention_method == "LuongConcat":
            return AttentionLuongConcat(self.hidden_size)

    def get_state(self):
        return self.lstm.get_state(), self.attention.get_state() if self.attention_method == "Tu" else None

    def set_state(self, encoder_hidden_states, ch_list, attention_chc):
        self.lstm.set_state(ch_list)
        self.attention.add_encoder_hidden_states(encoder_hidden_states)
        if self.attention_method == "Tu" and attention_chc is not None:
            self.attention.set_state(*attention_chc)

    def reset_state(self):
        self.lstm.reset_state()
        self.attention.reset_state()

########################################################
#Attention Networks

class AttentionLuongDot(Chain):
    def __init__(self):
        self.reset_state()
        super(AttentionLuongDot, self).__init__()

    def __call__(self, decoder_hidden):
        attention_weights = functions.softmax(functions.batch_matmul(self.encoder_hidden_states, decoder_hidden)) #bl1
        context_vector = functions.reshape(functions.batch_matmul(self.encoder_hidden_states, attention_weights, transa = True), (self.batch_size, self.encoder_hidden_size)) #bh
        return context_vector, attention_weights

    def add_encoder_hidden_states(self, encoder_hidden_states):
        self.encoder_hidden_states = encoder_hidden_states
        self.batch_size, self.sentence_length, self.encoder_hidden_size = encoder_hidden_states.shape

    def reset_state(self):
        self.encoder_hidden_states = None
        self.batch_size = None
        self.sentence_length = None
        self.encoder_hidden_size = None
        
class AttentionLuongGeneral(Chain, NetworkFunctions):
    def __init__(self, hidden_size):
        self.reset_state()
        self.hidden_size = hidden_size
        super(AttentionLuongGeneral, self).__init__(
            weight_matrix = links.Linear(hidden_size, hidden_size, nobias = True),
        )

    def __call__(self, decoder_hidden):
        attention_weights = functions.softmax(functions.batch_matmul(self.converted_encoder_hidden_states, decoder_hidden)) 
        context_vector = functions.reshape(functions.batch_matmul(self.encoder_hidden_states, attention_weights, transa = True), (self.batch_size, self.hidden_size))
        return context_vector, attention_weights

    def add_encoder_hidden_states(self, encoder_hidden_states):
        self.encoder_hidden_states = encoder_hidden_states
        self.batch_size, self.sentence_length, self.encoder_hidden_size = encoder_hidden_states.shape
        encoder_hidden_states_2d = functions.reshape(encoder_hidden_states, (self.batch_size * self.sentence_length, self.encoder_hidden_size))
        self.converted_encoder_hidden_states = functions.reshape(self.mask_output(encoder_hidden_states_2d, self.weight_matrix(encoder_hidden_states_2d)), (self.batch_size, self.sentence_length, self.hidden_size))

    def reset_state(self):
        self.encoder_hidden_states = None
        self.converted_encoder_hidden_states = None
        self.batch_size = None
        self.sentence_length = None
        self.encoder_hidden_size = None

class AttentionLuongConcat(Chain, NetworkFunctions):
    def __init__(self, hidden_size):
        self.reset_state()
        self.hidden_size = hidden_size
        super(AttentionLuongConcat, self).__init__(
            source_weight_matrix = links.Linear(hidden_size, 1, nobias = True),
            target_weight_matrix = links.Linear(hidden_size, 1, nobias = True),
        )

    def __call__(self, decoder_hidden):
        converted_decoder_hidden = functions.broadcast_to(self.mask_output(decode_hidden, self.target_weight_matrix(decoder_hidden)), self.converted_encoder_hidden_states.shape)
        attention_weights = functions.softmax(functions.add(self.converted_encoder_hidden_states, converted_decoder_hidden))
        context_vector = functions.reshape(functions.batch_matmul(self.encoder_hidden_states, attention_weights, transa = True), (self.batch_size, self.hidden_size))
        return context_vector, attention_weights

    def add_encoder_hidden_states(self, encoder_hidden_states):
        self.encoder_hidden_states = encoder_hidden_states
        self.batch_size, self.sentence_length, self.encoder_hidden_size = encoder_hidden_states.shape
        encoder_hidden_states_2d = functions.reshape(encoder_hidden_states, (self.batch_size * self.sentence_length, self.encoder_hidden_size))
        self.converted_encoder_hidden_states = functions.reshape(self.mask_output(encoder_hidden_states_2d, self.source_weight_matrix(encoder_hidden_states_2d)), (self.batch_size, self.sentence_length))
        #self.converted_encoder_hidden_states = functions.reshape(self.source_weight_matrix(functions.reshape(encoder_hidden_states, (self.batch_size * self.sentence_length, self.encoder_hidden_size))), (self.batch_size, self.sentence_length))

    def reset_state(self):
        self.encoder_hidden_states = None
        self.converted_encoder_hidden_states = None
        self.batch_size = None
        self.sentence_length = None
        self.encoder_hidden_size = None

class AttentionBahdanau(Chain):
    def __init__(self, hidden_size):
        self.reset_state()
        self.hidden_size = hidden_size
        super(AttentionBahdanau, self).__init__(
            source_weight_matrix = links.Linear(hidden_size, hidden_size, nobias = True),
            target_weight_matrix = links.Linear(hidden_size, hidden_size, nobias = True),
            final_weight_matrix = links.Linear(hidden_size, 1, nobias = True),
        )

    def __call__(self, decoder_hidden):
        converted_decoder_hidden = functions.transpose(functions.broadcast_to(self.target_weight_matrix(decoder_hidden), (self.sentence_length, self.batch_size, self.hidden_size)), (1, 0, 2)) #blh
        attention_weights = functions.softmax(functions.reshape(self.final_weight_matrix(functions.reshape(functions.tanh(functions.add(self.converted_encoder_hidden_states, converted_decoder_hidden)), (self.batch_size * self.sentence_length, self.hidden_size))), (self.batch_size, self.sentence_length)))
        context_vector = functions.reshape(functions.batch_matmul(self.encoder_hidden_states, attention_weights, transa = True), (self.batch_size, self.hidden_size))
        return context_vector, attention_weights

    def add_encoder_hidden_states(self, encoder_hidden_states):
        self.encoder_hidden_states = encoder_hidden_states
        self.batch_size, self.sentence_length, self.encoder_hidden_size = encoder_hidden_states.shape
        self.converted_encoder_hidden_states = functions.reshape(self.source_weight_matrix(functions.reshape(encoder_hidden_states, (self.batch_size * self.sentence_length, self.encoder_hidden_size))), (self.batch_size, self.sentence_length, self.hidden_size))

    def reset_state(self):
        self.encoder_hidden_states = None
        self.converted_encoder_hidden_states = None
        self.batch_size = None
        self.sentence_length = None
        self.encoder_hidden_size = None

class AttentionTu(Chain):
    def __init__(self, hidden_size, coverage_size, library):
        self.hidden_size = hidden_size
        self.coverage_size = coverage_size
        self.library = library
        super(AttentionTu, self).__init__(
            source_weight_matrix = links.Linear(hidden_size, hidden_size, nobias = True),
            target_weight_matrix = links.Linear(hidden_size, hidden_size, nobias = True),
            coverage_weight_matrix = links.Linear(coverage_size, hidden_size, nobias = True),
            final_weight_matrix = links.Linear(hidden_size, 1, nobias = True),
            coverage_lstm = uniLSTM(2 * hidden_size + 1, coverage_size, None, False, None, False, library),
        )
        self.reset_state()

    def __call__(self, decoder_hidden):
        converted_decoder_hidden = functions.reshape(functions.transpose(functions.broadcast_to(self.target_weight_matrix(decoder_hidden), (self.sentence_length, self.batch_size, self.hidden_size)), (1, 0, 2)), (self.converted_encoder_hidden_states.shape)) #b*l,h
        converted_coverage = self.coverage_weight_matrix(functions.reshape(self.coverage, (self.batch_size * self.sentence_length, self.coverage_size)))
        attention_weights = functions.softmax(functions.reshape(self.final_weight_matrix(functions.tanh(self.converted_encoder_hidden_states + converted_decoder_hidden + converted_coverage)), (self.batch_size, self.sentence_length)))
        context_vector = functions.reshape(functions.batch_matmul(self.encoder_hidden_states, attention_weights, transa = True), (self.batch_size, self.hidden_size))
        self.coverage = functions.reshape(self.coverage_lstm(functions.reshape(functions.dstack((self.encoder_hidden_states, functions.transpose(functions.broadcast_to(decoder_hidden, (self.sentence_length, self.batch_size, self.hidden_size)), (1, 0, 2)), functions.reshape(attention_weights, (self.batch_size, self.sentence_length, 1)))), (self.batch_size * self.sentence_length, 2 * self.hidden_size + 1))), (self.batch_size, self.sentence_length, self.coverage_size)) #b*l, 2*h+1
        return context_vector, attention_weights

    def add_encoder_hidden_states(self, encoder_hidden_states):
        self.encoder_hidden_states = encoder_hidden_states
        self.batch_size, self.sentence_length, self.hidden_size = encoder_hidden_states.shape
        self.converted_encoder_hidden_states = self.source_weight_matrix(functions.reshape(encoder_hidden_states, (self.batch_size * self.sentence_length, self.hidden_size)))
        self.coverage = Variable(self.library.zeros((self.batch_size, self.sentence_length, self.coverage_size), dtype = self.library.float32))

    def get_state(self):
        return functions.reshape(self.coverage_lstm.c, (self.batch_size, self.sentence_length, self.coverage_size)), functions.reshape(self.coverage_lstm.h,(self.batch_size, self.sentence_length, self.coverage_size)), self.coverage
    
    def set_state(self, c, h, coverage):
        self.coverage_lstm.set_state(functions.reshape(c, (self.batch_size * self.sentence_length, self.hidden_size)), functions.reshape(h, (self.batch_size * self.sentence_length, self.hidden_size)))
        self.coverage = coverage
    
    def reset_state(self):
        self.coverage = None
        self.coverage_lstm.reset_state()
        self.encoder_hidden_states = None
        self.converted_encoder_hidden_states = None
        self.batch_size = None
        self.sentence_length = None
        self.encoder_hidden_size = None

class SelfAttentionLin(Chain, NetworkFunctions):
    def __init__(self, hidden_size, matrix_size):
        self.hidden_size = hidden_size
        self.matrix_size = matrix_size
        super(SelfAttentionLin, self).__init__(
            first_weight_matrix = links.Linear(hidden_size, hidden_size, nobias = True), 
            final_weight_matrix = links.Linear(hidden_size, matrix_size, nobias = True), 
        )

    def __call__(self, hidden_states):
        hidden_states = self.stack_list_to_axis_1(hidden_states) #b*l*h  
        batch_hidden_states, batch_size, sentence_length = self.variable_axis_into_batch_axis_for_2d(hidden_states)
        attention_weights = functions.softmax(self.variable_axis_from_batch_axis_for_2d(self.final_weight_matrix(functions.tanh(self.first_weight_matrix(batch_hidden_states))), batch_size, sentence_length), axis = 1) #b*l*m
        sentence_matrix = functions.batch_matmul(attention_weights, hidden_states, transa = True) #b*m*h
        return sentence_matrix


########################################################
#Transformer Networks

class PositionalWordEmbedding(Chain, NetworkFunctions):
    def __init__(self, vocabulary_size, embed_size, activation_method, use_dropout, dropout_rate, vocabulary, word2vec, library):
        super(PositionalWordEmbedding, self).__init__(
            word2embed = WordEmbedding(vocabulary_size, embed_size, activation_method, use_dropout, dropout_rate, vocabulary, word2vec),
        )
        self.vocabulary_size = vocabulary_size
        self.embed_size = embed_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.vocabulary = vocabulary
        self.library = library

    def __call__(self, sentence, sentence_length): #bl
        batch_size = int(sentence.shape[0] / sentence_length)
        word_embed = (self.embed_size ** 0.5) * self.word2embed(sentence) #bl*e
        position, _, _ = self.variable_axis_into_batch_axis_for_2d(functions.transpose(functions.broadcast_to(Variable(self.library.arange(sentence_length, dtype = self.library.float32)), (batch_size, self.embed_size, sentence_length)), axes = (0, 2, 1)))
        i, _, _ = self.variable_axis_into_batch_axis_for_2d(functions.broadcast_to(Variable(self.library.arange(self.embed_size, dtype = self.library.float32)), (batch_size, sentence_length, self.embed_size)))
        position_embed = functions.where(i.data % 2 == 0, functions.sin(position / (10000 ** (i / self.embed_size))), functions.cos(position / (10000 ** ((i - 1) / self.embed_size))))
        return word_embed + self.dropout(position_embed) #bl*e

class MultiHeadAttention(Chain, NetworkFunctions):
    def __init__(self, embed_size, head_size, key_size, value_size):
        super(MultiHeadAttention, self).__init__(
            key_layer = links.Linear(embed_size, embed_size, nobias = True),
            value_layer = links.Linear(embed_size, embed_size, nobias = True),
            query_layer = links.Linear(embed_size, embed_size, nobias = True),
            final_layer = links.Linear(embed_size, embed_size, nobias = True),
        )
        self.embed_size = embed_size
        self.head_size = head_size

    def __call__(self, key, value, query, batch_size): #bl*e
        new_key = functions.concat(functions.split_axis(self.variable_axis_from_batch_axis_for_2d(self.key_layer(key), batch_size, int(key.shape[0] / batch_size)), self.head_size, axis = 2), axis = 0) #bh*l*k
        new_value = functions.concat(functions.split_axis(self.variable_axis_from_batch_axis_for_2d(self.value_layer(value), batch_size, int(value.shape[0] / batch_size)), self.head_size, axis = 2), axis = 0) #bh*l*v
        new_query = functions.concat(functions.split_axis(self.variable_axis_from_batch_axis_for_2d(self.query_layer(query), batch_size, int(query.shape[0] / batch_size)), self.head_size, axis = 2), axis = 0) #bh*l*q
        concat_heads, _, _ = self.variable_axis_into_batch_axis_for_2d(functions.concat(functions.split_axis(self.scaled_dot_product_attention(new_key, new_value, new_query), self.head_size, axis = 0), axis = 2)) #bl*e
        return self.final_layer(concat_heads)
 
    def scaled_dot_product_attention(self, key, value, query):
        return functions.softmax(functions.batch_matmul(functions.batch_matmul(query, key, transb = True) / ((self.embed_size / self.head_size) ** 0.5), value)) #bh*l*v

class PositionWiseFeedForwardNetworks(Chain, NetworkFunctions):
    def __init__(self, embed_size, hidden_size):
        super(PositionWiseFeedForwardNetworks, self).__init__(
        layer1 = links.Linear(embed_size, hidden_size),
        layer2 = links.Linear(hidden_size, embed_size),
    )

    def __call__(self, x): #bl*e
        return self.layer2(functions.relu(self.layer1(x))) #bl*e

class OneLayerTransformerEncoder(Chain, NetworkFunctions):
    def __init__(self, head_size, embed_size, hidden_size, key_size, value_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        super(OneLayerTransformerEncoder, self).__init__(
            multi_head_attention = MultiHeadAttention(embed_size, head_size, key_size, value_size),
            layer_norm_1 = links.LayerNormalization(),
            feed_forward_networks = PositionWiseFeedForwardNetworks(embed_size, hidden_size),
            layer_norm_2 = links.LayerNormalization(),
        )
        self.use_residual = use_residual

    def __call__(self, x, batch_size): #bl*e
        hidden1 = self.layer_norm_1(self.residual(self.multi_head_attention(x, x, x, batch_size), x))
        hidden2 = self.layer_norm_2(self.residual(self.feed_forward_networks(hidden1), hidden1))
        return hidden2 #bl*e

class TransformerEncoder(ChainList, NetworkFunctions):
    def __init__(self, layer_size, head_size, vocabulary_size, embed_size, hidden_size, key_size, value_size, activation_method, use_dropout, dropout_rate, use_residual, library, vocabulary, word2vec):
        layers = list()
        layers.append(PositionalWordEmbedding(vocabulary_size, embed_size, activation_method, use_dropout, dropout_rate, vocabulary, word2vec, library))
        for _ in range(layer_size):
            layers.append(OneLayerTransformerEncoder(head_size, embed_size, hidden_size, key_size, value_size, activation_method, use_dropout, dropout_rate, use_residual, library))
        super(TransformerEncoder, self).__init__(*layers)
        self.layer_size = layer_size
    
    def __call__(self, sentence):
        hidden_states = list()
        for i, layer in enumerate(self.children()):
            if i == 0:
                xs, batch_size, sentence_length = self.stack_variable_list_to_batch_axis_for_1d(sentence)
                embed = layer(xs, sentence_length)
            else:
                hidden = layer(hidden if i != 1 else embed, batch_size)
                hidden_states.append(hidden)
        return hidden_states #bl*v
            
class OneLayerTransformerDecoder(Chain, NetworkFunctions):
    def __init__(self, head_size, embed_size, hidden_size, key_size, value_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        super(OneLayerTransformerDecoder, self).__init__(
            masked_multi_head_attention = MultiHeadAttention(embed_size, head_size, key_size, value_size),
            layer_norm_1 = links.LayerNormalization(),
            multi_head_attention = MultiHeadAttention(embed_size, head_size, key_size, value_size),
            layer_norm_2 = links.LayerNormalization(),
            feed_forward_networks = PositionWiseFeedForwardNetworks(embed_size, hidden_size),
            layer_norm_3 = links.LayerNormalization(),
        )
        self.use_residual = use_residual

    def __call__(self, x, encoder_hidden, batch_size): #bl*e 
        hidden1 = self.layer_norm_1(self.residual(self.multi_head_attention(x, x, x, batch_size), x))
        hidden2 = self.layer_norm_2(self.residual(self.multi_head_attention(encoder_hidden, encoder_hidden, hidden1, batch_size), hidden1))
        hidden3 = self.layer_norm_3(self.residual(self.feed_forward_networks(hidden2), hidden2))
        return hidden3 #bl*e

class TransformerDecoder(ChainList, NetworkFunctions):
    def __init__(self, layer_size, head_size, vocabulary_size, embed_size, hidden_size, key_size, value_size, activation_method, use_dropout, dropout_rate, use_residual, generation_limit, use_beamsearch, beam_size, library, vocabulary, word2vec):
        layers = list()
        layers.append(PositionalWordEmbedding(vocabulary_size, embed_size, activation_method, use_dropout, dropout_rate, vocabulary, word2vec, library))
        for _ in range(layer_size):
            layers.append(OneLayerTransformerDecoder(head_size, embed_size, hidden_size, key_size, value_size, activation_method, use_dropout, dropout_rate, use_residual, library))
        layers.append(FeedForwardNetwork(embed_size, vocabulary_size, "None", False, 0.0, False, False))
        super(TransformerDecoder, self).__init__(*layers)
        self.layer_size = layer_size
        self.library = library
    
    def __call__(self, sentence, encoder_hidden_states):
        if sentence is not None:
            loss = Variable(self.library.zeros((), dtype = self.library.float32))
            for i, layer in enumerate(self.children()):
                if i == 0:
                    xs, batch_size, sentence_length = self.stack_variable_list_to_batch_axis_for_1d(sentence)
                    embed = self.variable_axis_from_batch_axis_for_2d(layer(xs, sentence_length), batch_size, sentence_length)
                    break
            for i in range(len(sentence) - 1):
                target_embed_states, _, _ = self.variable_axis_into_batch_axis_for_2d(functions.stack(functions.separate(embed, axis = 1)[:i + 1], axis = 1))
                score = self.decode_one_step(target_embed_states, encoder_hidden_states, batch_size) #bl*v
                correct_words_1d, batch_size, sentence_length = self.stack_variable_list_to_batch_axis_for_1d(sentence[1 : i + 2]) #bl
                loss += functions.softmax_cross_entropy(score, correct_words_1d, ignore_label = -1)
            predicts = list(self.separate_variable_list_from_batch_axis_for_1d(functions.argmax(score, axis = 1), batch_size, sentence_length))
            predicts.insert(0, sentence[0])
            return loss, predicts
    
    def decode_one_step(self, target_embed_states, encoder_hidden_states, batch_size):
        hidden_states = list()
        for i, layer in enumerate(self.children()):
            if i > 0 and i <= self.layer_size:
                hidden = layer(hidden if i != 1 else target_embed_states, encoder_hidden_states[i - 1], batch_size)
                hidden_states.append(hidden)
            elif i != 0:
                return layer(hidden)

########################################################
#GAN Networks

class RNNDiscriminatorYang(Chain, NetworkFunctions):
    def __init__(self, bilstm_method, layer_size, embed_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.bilstm_method = bilstm_method
        self.layer_size = layer_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        super(RNNDiscriminatorYang, self).__init__(
            source_lstm = self.select_bilstm(),
            target_lstm = self.select_bilstm(),
            final_weight_matrix = links.Linear(2 * hidden_size, 1),
        )

    def __call__(self, source_embed_states, true_embed_states, generate_embed_states):
        predict_true = self.forward(source_embed_states, true_embed_states)
        predict_generate = self.forward(source_embed_states, generate_embed_states)
        
        loss_discriminator = functions.sigmoid_cross_entropy(predict_true, Variable(self.library.ones(predict_true.shape, dtype=self.library.int32)))
        loss_generator = functions.sigmoid_cross_entropy(predict_generate, Variable(self.library.ones(predict_generate.shape, dtype=self.library.int32)))
        loss_discriminator += functions.sigmoid_cross_entropy(predict_generate, Variable(self.library.zeros(predict_generate.shape, dtype=self.library.int32)))

        return loss_generator, loss_discriminator, functions.sigmoid(predict_true), functions.sigmoid(predict_generate)

    def train_fixed_discriminator(self, source_embed_states, generate_embed_states):
        predict_generate = self.forward(source_embed_states, generate_embed_states)
        
        loss_generator = functions.sigmoid_cross_entropy(predict_generate, Variable(self.library.ones(predict_generate.shape, dtype=self.library.int32)))

        return loss_generator, None, None, functions.sigmoid(predict_generate)

    def train_with_score(self, source_embed_states, target_embed_states, correct_score):
        predict_score = functions.sigmoid(self.forward(source_embed_states, target_embed_states))
        loss = functions.mean_squared_error(predict_score, correct_score)
        return loss, predict_score 

    def evaluation(self, source_embed_states, target_embed_states):
        predict_score = self.forward(source_embed_states, target_embed_states)
        return functions.sigmoid(predict_score)

    def forward(self, source_embed_states, target_embed_states):
        self.reset_state()
        source_hidden_states = self.source_lstm(source_embed_states)
        target_hidden_states = self.target_lstm(target_embed_states)
        
        source_average = functions.average(functions.dstack(source_hidden_states), axis = 2)
        target_average = functions.average(functions.dstack(target_hidden_states), axis = 2)

        predict_score = functions.reshape(self.final_weight_matrix(functions.concat((source_average, target_average))), (source_average.shape[0],))
        
        return predict_score

    def get_state(self):
        return self.source_lstm.get_state(), self.target_lstm.get_state()

    def reset_state(self):
        self.source_lstm.reset_state()
        self.target_lstm.reset_state()
    
    def select_bilstm(self):
        if self.bilstm_method == "FinalConcat":
            return NlayerFinalConcatBiLSTM(self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        elif self.bilstm_method == "FinalAdd":
            return NlayerFinalAddBiLSTM(self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        
        else:
            return NlayerBiLSTM(self.bilstm_method, self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)

########################################################
#TBA
#class CNNDiscriminatorYang(Chain, NetworkFunctions):
########################################################

class RNNDiscriminatorYukio(Chain, NetworkFunctions):
    def __init__(self, compression_method, bilstm_method, layer_size, embed_size, hidden_size, matrix_size, activation_method, use_dropout, dropout_rate, use_residual, library):
        self.compression_method = compression_method
        self.bilstm_method = bilstm_method
        self.layer_size = layer_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.activation_method = activation_method
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.library = library
        super(RNNDiscriminatorYukio, self).__init__(
            source_lstm = self.select_bilstm(),
            target_lstm = self.select_bilstm(),
        )
        if compression_method == "Attention":
            self.add_link("source_attention", SelfAttentionLin(hidden_size, matrix_size))
            self.add_link("target_attention", SelfAttentionLin(hidden_size, matrix_size))

    def __call__(self, source_embed_states, true_embed_states, generate_embed_states):
        predict_true = self.forward(source_embed_states, true_embed_states)
        predict_generate = self.forward(source_embed_states, generate_embed_states)
        
        loss_discriminator = functions.sigmoid_cross_entropy(predict_true, Variable(self.library.ones(predict_true.shape, dtype=self.library.int32)))
        loss_generator = functions.sigmoid_cross_entropy(predict_generate, Variable(self.library.ones(predict_generate.shape, dtype=self.library.int32)))
        loss_discriminator += functions.sigmoid_cross_entropy(predict_generate, Variable(self.library.zeros(predict_generate.shape, dtype=self.library.int32)))

        return loss_generator, loss_discriminator, functions.sigmoid(predict_true), functions.sigmoid(predict_generate)

    def train_with_score(self, source_embed_states, target_embed_states, correct_score):
        predict_score = functions.sigmoid(self.forward(source_embed_states, target_embed_states))
        loss = functions.mean_squared_error(predict_score, correct_score)
        for layer in self.source_lstm.forwardLSTM.children():
            print(layer.uniLSTM.upward.W)
            sys.stdout.flush()
        return loss, predict_score 

    def evaluation(self, source_embed_states, target_embed_states):
        predict_score = self.forward(source_embed_states, target_embed_states)
        return functions.sigmoid(predict_score)

    def forward(self, source_embed_states, target_embed_states):
        self.reset_state()
        source_hidden_states = self.source_lstm(source_embed_states)
        target_hidden_states = self.target_lstm(target_embed_states)
        
        source_vector = self.compression(source_hidden_states, "source")
        target_vector = self.compression(target_hidden_states, "target")

        predict_score = functions.reshape(functions.batch_matmul(source_vector, target_vector, transa = True), (source_vector.shape[0],))
        
        return predict_score

    def get_state(self):
        return self.source_lstm.get_state(), self.target_lstm.get_state()

    def reset_state(self):
        self.source_lstm.reset_state()
        self.target_lstm.reset_state()

    def compression(self, hidden_states, side):
        if self.compression_method == "Average":
            return functions.average(functions.dstack(hidden_states), axis = 2)
        elif self.compression_method == "BothEndHidden":
            if side == "source":
                return self.get_state()[0][-1][1]
            else:
                return self.get_state()[1][-1][1]
        elif self.compression_method == "Attention":
            sentence_matrix = self.source_attention(hidden_states) if side == "source" else self.target_attention(hidden_states)
            return functions.reshape(sentence_matrix, (sentence_matrix.shape[0], sentence_matrix.shape[1] * sentence_matrix.shape[2]))

    def select_bilstm(self):
        if self.bilstm_method == "FinalConcat":
            return NlayerFinalConcatBiLSTM(self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        elif self.bilstm_method == "FinalAdd":
            return NlayerFinalAddBiLSTM(self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        
        else:
            return NlayerBiLSTM(self.bilstm_method, self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)

