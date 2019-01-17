class SharedTargetCNMT(Chain):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, layer_size, embed_size, hidden_size, bilstm_method, attention_method, activation_method, use_dropout, dropout_rate, use_residual, generation_limit, use_beamsearch, beam_size, library, source_vocabulary, target_vocabulary, source_word2vec, target_word2vec):
        super(SharedTargetCNMT, self).__init__(
            encoder = RNNEncoder(bilstm_method, layer_size, source_vocabulary_size, embed_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, library, source_vocabulary, source_word2vec),
            decoder = CNMTDecoder(layer_size, target_vocabulary_size, embed_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, attention_method, generation_limit, use_beamsearch, beam_size, library, target_vocabulary, target_word2vec),
		)
        self.library = library
        self.hidden_size = hidden_size

    def __call__(self, source_sentence, target_sentence):
        self.reset_states()
        encoder_hidden_states, encoder_states_list = self.encoder(source_sentence)
        self.decoder.set_history(encoder_hidden_states[0].shape[0])
        loss, predicts = self.decoder(encoder_hidden_states, encoder_states_list, target_sentence)
        return loss, predicts

    def forward(self, source_sentence, target_sentence):
        self.reset_states()
        encoder_hidden_states, encoder_states_list = self.encoder(source_sentence)
        self.decoder.set_history(encoder_hidden_states[0].shape[0])
        loss, predicts, source_attention_weights, target_attention_weights = self.decoder.forward(encoder_hidden_states, encoder_states_list, target_sentence)
        return loss, predicts, source_attention_weights, target_attention_weights
    
    def reset_states(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    def init_previous_state(self, batch_size):
        self.decoder.init_previous_state_shared(batch_size)


class CNMTDecoder(Chain, NetworkFunctions):
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
        super(CNMTDecoder, self).__init__(
            embedding = WordEmbedding(target_vocabulary_size, embed_size, activation_method, use_dropout, dropout_rate, target_vocabulary, target_word2vec),
            lstm = NlayerUniLSTM(layer_size, embed_size + hidden_size, hidden_size, activation_method, use_dropout, dropout_rate, use_residual, self.library),
            tilde = FeedForwardNetwork(3 * hidden_size, hidden_size, activation_method, use_dropout, dropout_rate, False, False, library),
            output = FeedForwardNetwork(hidden_size, target_vocabulary_size, "None", False, 0.0, False, False, library),
            source_attention = self.select_attention(),
            target_attention = self.select_attention(),
        )

    def __call__(self, encoder_hidden_states, encoder_lstm_states, sentence, past_encoder_states=None):
        return self.forward(encoder_hidden_states, encoder_lstm_states, sentence)[:2]

    def forward(self, encoder_hidden_states, encoder_lstm_states, sentence, past_encoder_states=None):
        predicts = list()
        target_embed_states = list()
        hidden_states = list()

        if sentence is not None:
            loss = Variable(self.library.zeros((), dtype = self.library.float32))
            predicts.append(sentence[0])
            hidden_tilde = Variable(self.library.zeros((sentence[0].shape[0], self.hidden_size), dtype = self.library.float32))
            xs, batch_size, sentence_length = self.stack_variable_list_to_batch_axis_for_1d(sentence)
            target_embed_states = self.separate_variable_list_from_batch_axis_for_2d(self.embedding(xs), batch_size, sentence_length)
            self.set_state(self.stack_list_to_axis_1(encoder_hidden_states), encoder_lstm_states)
            
            for i, (previous_embed, correct_word) in enumerate(zip(target_embed_states, sentence[1:])):
                score, hidden, hidden_tilde = self.decode_one_step(previous_embed, hidden_tilde)
                hidden_states.append(hidden.data)
                predict = functions.argmax(score, axis = 1)
                loss += functions.softmax_cross_entropy(score, correct_word, ignore_label = -1)
                predicts.append(functions.where(correct_word.data != -1, predict, correct_word))
            xs, batch_size, sentence_length = self.stack_variable_list_to_batch_axis_for_1d(predicts)
            
            self.previous_states = hidden_states
            
            return loss, predicts

        elif not self.use_beamsearch:
            batch_size = encoder_hidden_states[0].shape[0]
            predict = Variable(self.library.array([self.vocabulary.word2id["<s>"]] * batch_size, dtype = self.library.int32)) 
            predicts.append(predict)
            hidden_tilde = Variable(self.library.zeros((batch_size, self.hidden_size), dtype = self.library.float32))
            self.set_state(self.stack_list_to_axis_1(encoder_hidden_states), encoder_lstm_states)

            while len(predicts) - 1 < self.generation_limit:
                previous_embed = self.embedding(predict)
                score, hidden, hidden_tilde = self.decode_one_step(previous_embed, hidden_tilde)
                hidden_states.append(hidden.data)

                predict = functions.argmax(score, axis = 1)
                predicts.append(predict)
                if batch_size == 1 and predict.data[0] == self.vocabulary.word2id["</s>"]:
                    break

            if batch_size == 1 and predict.data[0] != self.vocabulary.word2id["</s>"]:
                eos = Variable(self.library.array([self.vocabulary.word2id["</s>"]], dtype = self.library.int32))
                predicts.append(eos)

            self.previous_states = hidden_states    

            return None, predicts
        
        else:
            bos = Variable(self.library.array([1], dtype = self.library.int32))
            hidden_tilde = Variable(self.library.zeros((1, self.hidden_size), dtype = self.library.float32))
            #initial_beam = [log_probability, predict, decoder_hiddens, decoder_tildes, decoder_lstm_states, source_attention_weights, target_attention_weights, finished]
            initial_beam = [(0, [bos], list(), [hidden_tilde], None, list())]
            encoder_hidden_states = functions.stack(encoder_hidden_states, axis=1)

            _, predicts, decoder_hiddens, _, _, _ = self.beam_search(initial_beam, encoder_hidden_states, encoder_lstm_states)
            self.previous_states = decoder_hiddens
            
            return None, predicts, source_attention_weights, target_attention_weights
            
    def decode_one_step(self, previous_embed, previous_hidden_tilde):
        hidden = self.lstm(functions.concat((previous_embed, previous_hidden_tilde)))
        attention, _ = self.source_attention(hidden)
        target_attention, _ = self.target_attention(hidden)
        hidden_tilde = self.tilde(functions.concat((hidden, attention, target_attention)))
        score = self.output(hidden_tilde)
        return score, hidden, hidden_tilde

    def beam_search(self, initial_beam, encoder_hidden_states, encoder_lstm_states):
        beam = [0] * self.generation_limit
        for i in range(self.generation_limit):
            beam[i] = list()
            if i == 0:
                new_beam = list()
                for logprob, predicts, decoder_hiddens, decoder_tildes, decoder_lstm_states, finished in initial_beam:
                    self.set_state(encoder_hidden_states, encoder_lstm_states)
                    previous_embed = self.embedding(predicts[-1])
                    score, hidden, hidden_tilde = self.decode_one_step(previous_embed, decoder_tildes[-1])
                    prob = functions.softmax(score)
                    lstm_state = self.get_state()
                    for predict in self.library.argsort(prob.data[0])[-1:-self.beam_size-1:-1]:
                        predict_variable = Variable(self.library.array([predict], dtype = self.library.int32))
                        new_beam.append((logprob + functions.log(prob[0][predict]), predicts + [predict_variable], decoder_hiddens + [hidden.data], decoder_tildes + [hidden_tilde], lstm_state, True if predict == 2 else False)) 
            
            else:
                new_beam = list()
                for logprob, predicts, decoder_hiddens, decoder_tildes, decoder_lstm_states, finished in beam[i - 1]:
                    if finished is not True:
                        self.set_state(encoder_hidden_states, decoder_lstm_states)
                        previous_embed = self.embedding(predicts[-1])
                        score, hidden, hidden_tilde = self.decode_one_step(previous_embed, decoder_tildes[-1])
                        prob = functions.softmax(score)
                        lstm_state = self.get_state()
                        for predict in self.library.argsort(prob.data[0])[-1:-self.beam_size-1:-1]:
                            predict_variable = Variable(self.library.array([predict], dtype = self.library.int32))
                            new_beam.append((logprob + functions.log(prob[0][predict]), predicts + [predict_variable], decoder_hiddens + [hidden.data], decoder_tildes + [hidden_tilde], lstm_state, True if predict == 2 else False)) 
                    else:
                        new_beam.append((logprob, predicts, decoder_hiddens, decoder_tildes, decoder_lstm_states, finished))
            
            for _, items in zip(range(self.beam_size), sorted(new_beam, key = lambda x: self.beam_search_normalization(x[0].data, len(x[1])), reverse = True)):
                beam[i].append(items)
        
        return beam[-1]

    def beam_search_normalization(self, score, length):
        return score / length

    def select_attention(self):
        if self.attention_method == "Bahdanau":
            return AttentionBahdanau(self.hidden_size)
        elif self.attention_method == "LuongDot":
            return AttentionLuongDot()
        elif self.attention_method == "LuongGeneral":
            return AttentionLuongGeneral(self.hidden_size)
        elif self.attention_method == "LuongConcat":
            return AttentionLuongConcat(self.hidden_size)

    def get_state(self):
        return self.lstm.get_state()

    def set_state(self, encoder_hidden_states, ch_list):
        self.lstm.set_state(ch_list)
        self.source_attention.add_encoder_hidden_states(encoder_hidden_states)
        self.target_attention.add_encoder_hidden_states(self.decoder_hidden_states)

    def reset_state(self):
        self.lstm.reset_state()
        self.source_attention.reset_state()
        self.target_attention.reset_state()    

    def set_history(self, batch_size):
        pad = self.previous_states[:batch_size]
        self.previous_states += pad
        self.decoder_hidden_states = self.stack_list_to_axis_1([Variable(state) for state in self.previous_states])

    def set_encoding_history(self, batch_size):
        pad = self.previous_states[:batch_size]
        self.previous_states += pad
        self.decoder_hidden_states = self.stack_list_to_axis_1(self.previous_states)

    def init_previous_state_shared(self, batch_size):
        self.previous_state = [self.library.zeros((batch_size, self.hidden_size), dtype=self.library.float32)]

    def init_previous_state_separated(self, batch_size):
        self.previous_sentence = [Variable(config.library.array([-1]*batch_size, dtype=config.library.int32)]

