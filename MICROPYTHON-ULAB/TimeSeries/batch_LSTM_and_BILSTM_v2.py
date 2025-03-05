#
# Creating a Long Short-Term Memory (LSTM) and Bidirectional LSTM
# (BI-LSTM) network from scratch in pure Python can be quite complex
# and is generally not recommended due to the inefficiency and lack of
# features compared to using libraries like TensorFlow or
# PyTorch. However, I can provide you with a simplified version of
# these models using NumPy for matrix operations to give you a better
# understanding of how they work.
#
# Let's start with a basic LSTM implementation. Note that this is a
# simplified version and does not include optimizations or advanced
# features.
#
# Basic LSTM Implementation
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        outputs = []

        for x in inputs:
            x = x.reshape(-1, 1)
            xc = np.vstack((h_prev, x))

            f = sigmoid(self.Wf @ xc + self.bf)
            i = sigmoid(self.Wi @ xc + self.bi)
            c_hat = tanh(self.Wc @ xc + self.bc)
            c = f * c_prev + i * c_hat
            o = sigmoid(self.Wo @ xc + self.bo)
            h = o * tanh(c)

            y = softmax(self.Wy @ h + self.by)
            outputs.append(y)

            h_prev = h
            c_prev = c

        return outputs

# Basic Bidirectional LSTM Implementation
# A Bidirectional LSTM (BI-LSTM) processes the input sequence in both
# forward and backward directions and combines the results.

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Initialize weights
        self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim)
        self.b_f = np.zeros((hidden_dim, 1))
        self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim)
        self.b_i = np.zeros((hidden_dim, 1))
        self.W_c = np.random.randn(hidden_dim, input_dim + hidden_dim)
        self.b_c = np.zeros((hidden_dim, 1))
        self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim)
        self.b_o = np.zeros((hidden_dim, 1))

    def forward(self, x, h_prev, c_prev):
        # Concatenate input and hidden state
        concat = np.vstack((h_prev, x))
        
        # Forget gate
        f_t = sigmoid(np.dot(self.W_f, concat) + self.b_f)
        # Input gate
        i_t = sigmoid(np.dot(self.W_i, concat) + self.b_i)
        # Candidate memory cell
        c_tilda = tanh(np.dot(self.W_c, concat) + self.b_c)
        # Output gate
        o_t = sigmoid(np.dot(self.W_o, concat) + self.b_o)
        
        # New cell state
        c_t = f_t * c_prev + i_t * c_tilda
        # New hidden state
        h_t = o_t * tanh(c_t)
        
        return h_t, c_t

class BiLSTM:
    def __init__(self, input_dim, hidden_dim):
        self.forward_lstm = LSTMCell(input_dim, hidden_dim)
        self.backward_lstm = LSTMCell(input_dim, hidden_dim)

    def forward(self, inputs):
        T = len(inputs)
        h_forward = [np.zeros((self.forward_lstm.hidden_dim, 1)) for _ in range(T + 1)]
        c_forward = [np.zeros((self.forward_lstm.hidden_dim, 1)) for _ in range(T + 1)]
        
        h_backward = [np.zeros((self.backward_lstm.hidden_dim, 1)) for _ in range(T + 1)]
        c_backward = [np.zeros((self.backward_lstm.hidden_dim, 1)) for _ in range(T + 1)]

        # Forward pass
        for t in range(T):
            h_forward[t + 1], c_forward[t + 1] = self.forward_lstm.forward(inputs[t], h_forward[t], c_forward[t])

        # Backward pass
        for t in reversed(range(T)):
            h_backward[t], c_backward[t] = self.backward_lstm.forward(inputs[t], h_backward[t + 1], c_backward[t + 1])

        # Concatenate forward and backward hidden states
        h_bi = [np.vstack((h_forward[t + 1], h_backward[t])) for t in range(T)]
        return h_bi

if __name__ == '__main__':
    # Example usage LSTM:
    lstm = LSTM(input_size=10, hidden_size=20, output_size=5)
    inputs = [np.random.randn(10) for _ in range(5)] # Example input sequence
    outputs = lstm.forward(inputs)
    print('LSTM:',outputs)
    # Example usage BI-LSTM:
    input_dim = 4  # example input dimension
    hidden_dim = 3  # example hidden dimension
    sequence_length = 5  # example sequence length
    inputs = [np.random.randn(input_dim, 1) for _ in range(sequence_length)]
    bi_lstm = BiLSTM(input_dim, hidden_dim)
    outputs = bi_lstm.forward(inputs)
    for t, output in enumerate(outputs):
        print(f"Time step {t}: {output.ravel()}")
