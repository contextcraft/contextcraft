import tensorflow as tf

class RNNWithCopyAttention(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(RNNWithCopyAttention, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.attention = tf.keras.layers.Attention()
        self.fc = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, hidden_state):
        x = self.embedding(inputs)
        rnn_output, rnn_state = self.rnn(x, initial_state=hidden_state)
        context_vector, attention_weights = self.attention([rnn_output, hidden_state], return_attention_scores=True)
        output = self.fc(context_vector)
        return output, rnn_state, attention_weights
