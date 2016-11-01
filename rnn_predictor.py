import tensorflow as tf


class RNNPredictor:
    # TODO: Load meta graph from file and use it instead to support various internal structures
    def __init__(self, vocabulary_size, hidden_size, layer_size, cell_type):
        with tf.Graph().as_default():
            # Placeholder for test data
            self.input = tf.placeholder(tf.int32, [None])

            # Lookup word embedding
            embedding = tf.Variable(tf.zeros([vocabulary_size, hidden_size]), name='embedding')
            rnn_inputs = tf.nn.embedding_lookup(embedding, self.input)

            # Create RNN cell
            if cell_type == 'lstm':
                cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            elif cell_type == 'rnn':
                cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
            else:
                cell = tf.nn.rnn_cell.GRUCell(hidden_size)

            # Stack multiple RNN cells
            if layer_size > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_size)

            self.initial_state = cell.zero_state(1, dtype=tf.float32)

            # Call the RNN cell
            with tf.variable_scope('RNN'):
                rnn_output, self.next_state = cell(rnn_inputs, self.initial_state)

            # Predict distribution over next word
            softmax_w = tf.Variable(tf.zeros([hidden_size, vocabulary_size]), name='softmax_w')
            softmax_b = tf.Variable(tf.zeros([vocabulary_size]), name='softmax_b')
            logits = tf.matmul(rnn_output, softmax_w) + softmax_b

            # predictions is negative log probability of shape [vocabulary_size]
            self.predictions = -tf.nn.log_softmax(logits)

            self.saver = tf.train.Saver(tf.trainable_variables())
            self.session = tf.Session()

    def predict(self, input_value, state_value=None):
        if state_value is not None:
            feed_dict = {self.input: input_value, self.initial_state: state_value}
        else:
            feed_dict = {self.input: input_value}
        predictions, next_state = self.session.run([self.predictions, self.next_state], feed_dict)
        return predictions, next_state

    def restore_from_directory(self, model_directory):
        model_path = tf.train.latest_checkpoint(model_directory)
        self.saver.restore(self.session, model_path)

    def restore_from_file(self, model_path):
        self.saver.restore(self.session, model_path)

    def close(self):
        self.session.close()
