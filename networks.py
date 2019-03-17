from chainer import *
import sys

class NetworkFunctions():
    def activate(self, x):
        if self.activation_method == 'tanh':
            return functions.tanh(x)
        if self.activation_method == 'sigmoid':
            return functions.sigmoid(x)
        if self.activation_method == 'relu':
            return functions.relu(x)
        elif self.activation_method == 'None':
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
        if self.bilstm_method == 'Concat':
            return ConcatBiLSTM(input_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        elif self.bilstm_method == 'Add':
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
        if self.bilstm_method == 'FinalConcat':
            return NlayerFinalConcatBiLSTM(self.layer_size, self.embed_size, self.hidden_size, self.activation_method, self.use_dropout, self.dropout_rate, self.use_residual, self.library)
        elif self.bilstm_method == 'FinalAdd':
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
            output = FeedForwardNetwork(hidden_size, target_vocabulary_size, 'None', False, 0.0, False, False, library),
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
            predict = Variable(self.library.array([self.vocabulary.word2id['<s>']] * batch_size, dtype = self.library.int32)) 
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
                if batch_size == 1 and predict.data[0] == self.vocabulary.word2id['</s>']:
                    break
            predict_embed_states.append(self.embedding(predict))

            if batch_size == 1 and predict.data[0] != self.vocabulary.word2id['</s>']:
                eos = Variable(self.library.array([self.vocabulary.word2id['</s>']], dtype = self.library.int32))
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

    def beam_search_normalization(self, score, length):
        return score / length

    def select_attention(self):
        if self.attention_method == 'Bahdanau':
            return AttentionBahdanau(self.hidden_size)
        elif self.attention_method == 'LuongDot':
            return AttentionLuongDot()
        elif self.attention_method == 'LuongGeneral':
            return AttentionLuongGeneral(self.hidden_size)
        elif self.attention_method == 'LuongConcat':
            return AttentionLuongConcat(self.hidden_size)

    def get_state(self):
        return self.lstm.get_state(), self.attention.get_state()

    def set_state(self, encoder_hidden_states, ch_list, attention_chc):
        self.lstm.set_state(ch_list)
        self.attention.add_encoder_hidden_states(encoder_hidden_states)

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
