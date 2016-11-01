import tensorflow as tf


class RNNTrainer:
    # TODO split to separate functions
    # TODO support truncated BPTT
    def __init__(self, batch_size, sentence_size, vocabulary_size, hidden_size, layer_size,
                 cell_type, optimizer_type, clip_norm, keep_prob, max_keep):

        with tf.Graph().as_default():
            # Placeholders for training data
            self.input = tf.placeholder(tf.int32, [batch_size, sentence_size])
            self.output = tf.placeholder(tf.int32, [batch_size, sentence_size])

            # Lookup word embedding
            embedding = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_size], stddev=0.01), name='embedding')
            inputs = tf.nn.embedding_lookup(embedding, self.input)
            inputs = tf.nn.dropout(inputs, keep_prob)

            # Create and connect RNN cells
            if cell_type == 'lstm':
                cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            elif cell_type == 'rnn':
                cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
            else:
                cell = tf.nn.rnn_cell.GRUCell(hidden_size)

            # Dropout output of the RNN cell
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            # Stack multiple RNN cells
            if layer_size > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_size)

            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, sentence_size, inputs)]
            outputs, _ = tf.nn.rnn(cell, rnn_inputs, dtype=tf.float32)

            # Predict distribution over next word
            output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
            softmax_w = tf.Variable(tf.truncated_normal([hidden_size, vocabulary_size], stddev=0.01), name='softmax_w')
            softmax_b = tf.Variable(tf.zeros(shape=[vocabulary_size]), name='softmax_b')
            logits = tf.matmul(output, softmax_w) + softmax_b

            # Define loss function and optimizer
            self.loss = tf.nn.seq2seq.sequence_loss(
                [logits],
                [tf.reshape(self.output, [-1])],
                [tf.ones([batch_size * sentence_size], dtype=tf.float32)])

            # Apply gradient clipping to address gradient explosion
            trainable_variables = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_variables)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, clip_norm)
            if optimizer_type == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(0.01)
            elif optimizer_type == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(0.01)
            elif optimizer_type == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(0.01)
            else:
                optimizer = tf.train.AdamOptimizer()
            self.train_step = optimizer.apply_gradients(zip(clipped_gradients, trainable_variables))

            # Keep latest max_keep checkpoints
            self.saver = tf.train.Saver(trainable_variables, max_to_keep=max_keep)
            self.session = tf.Session()
            self.session.run(tf.initialize_all_variables())

    def train(self, input_, output_):
        _, loss, gradient_norm = self.session.run([self.train_step, self.loss, self.gradient_norm],
                                                  {self.input: input_, self.output: output_})
        return loss, gradient_norm

    def save(self, model_path, epoch):
        self.saver.save(self.session, model_path, global_step=epoch)

    def close(self):
        self.session.close()
