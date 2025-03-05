#
# Below are simple implementations of LSTM (Long Short-Term Memory)
# and BI-LSTM (Bidirectional Long Short-Term Memory) networks using
# pure Python with NumPy. These implementations are simplified and do
# not include advanced features like gradient clipping, batch
# normalization, or efficient optimizations found in deep learning
# libraries like TensorFlow or PyTorch. Moreover, these
# implementations are very basic and do not include backpropagation,
# which is required for training.
#
# Explanation
# LSTMCell: Represents a single LSTM cell which performs
#           the forward pass.
# LSTM: Manages a sequence of LSTM cells and computes the
#           output for a sequence of inputs.
# BILSTM: Similar to LSTM, but it processes the input sequence
#           in both forward and backward directions.
# True_LSTM: add a supplementary parameter in the class for the input
#           comming from a dataset. The 3 first classes implement
#           random data generation for the input.
# LSTM_fb: adaptation of
#          https://github.com/CallMeTwitch/Neural-Network-Zoo/blob/main/LongShortTermMemoryNetwork.py
#          to read the Tour Perret dataset and not the text
#          dataset.
# Other functions after the class definitions: almost not used!
#

import numpy as np
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


##### Activation Functions #####
# from math import exp, tanh, exp
import math


def ssigmoid(input, derivative=False):
    if derivative:
        return input * (1 - input)
    # return 1 / (1 + math.exp(-input))
    res = np.array(1 / (1 + np.array([np.exp(i) for i in -input])))
    return res


def ttanh(input, derivative=False):
    if derivative:
        return 1 - input**2
    # return math.tanh(input)
    res = np.array([np.tanh(i) for i in input])
    # print(res.reshape(len(input),1))
    return res.reshape(len(input), 1)


def softmax(input):
    res = np.array([np.exp(i) for i in input])
    s = np.sum(res)
    if s == 0.0:
        return res / 100000000
    else:
        return res / s


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))
        self.Wh = np.random.randn(input_size, hidden_size) * 0.01
        self.bh = np.zeros((input_size, 1))

    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        xh = np.vstack((x, h_prev))
        # Forget gate
        f = sigmoid(np.dot(self.Wf, xh) + self.bf)
        # Input gate
        i = sigmoid(np.dot(self.Wi, xh) + self.bi)
        # Candidate cell state
        c_cand = tanh(np.dot(self.Wc, xh) + self.bc)
        # Cell state
        c = f * c_prev + i * c_cand
        # Output gate
        o = sigmoid(np.dot(self.Wo, xh) + self.bo)
        # Hidden state
        h = o * tanh(c)
        return h, c


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def forward(self, xs):
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        hs = {}
        cs = {}
        ys = {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        for t in range(len(xs)):
            hs[t], cs[t] = self.cell.forward(xs[t], hs[t - 1], cs[t - 1])
            ys[t] = np.dot(self.Wy, hs[t]) + self.by
        return hs, cs, ys

    def predict(self, xs):
        hs, cs, ys = self.forward(xs)
        return ys


class True_LSTM:
    def __init__(self, xxs, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
        # We bypass the random data and replace it by 'real' data
        # from our dataset
        # self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.Wy = xxs.reshape(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, xs):
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        hs = {}
        cs = {}
        ys = {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        for t in range(len(xs)):
            hs[t], cs[t] = self.cell.forward(xs[t], hs[t - 1], cs[t - 1])
            ys[t] = np.dot(self.Wy, hs[t]) + self.by
        return hs, cs, ys

    def predict(self, xs):
        hs, cs, ys = self.forward(xs)
        return ys


# BI-LSTM Implementation
class BILSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.forward_cell = LSTMCell(input_size, hidden_size)
        self.backward_cell = LSTMCell(input_size, hidden_size)
        self.Wy = np.random.randn(output_size, 2 * hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def forward(self, xs):
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        forward_hs = {}
        forward_cs = {}
        forward_hs[-1] = np.copy(h_prev)
        forward_cs[-1] = np.copy(c_prev)
        for t in range(len(xs)):
            forward_hs[t], forward_cs[t] = self.forward_cell.forward(
                xs[t], forward_hs[t - 1], forward_cs[t - 1]
            )

        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))
        backward_hs = {}
        backward_cs = {}
        backward_hs[len(xs)] = np.copy(h_prev)
        backward_cs[len(xs)] = np.copy(c_prev)
        for t in reversed(range(len(xs))):
            backward_hs[t], backward_cs[t] = self.backward_cell.forward(
                xs[t], backward_hs[t + 1], backward_cs[t + 1]
            )

        ys = {}
        for t in range(len(xs)):
            h = np.vstack((forward_hs[t], backward_hs[t]))
            ys[t] = np.dot(self.Wy, h) + self.by
        return forward_hs, forward_cs, backward_hs, backward_cs, ys

    def predict(self, xs):
        forward_hs, forward_cs, backward_hs, backward_cs, ys = self.forward(xs)
        return ys


# Normalize the data
def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


# Split the dataset into training and testing sets
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    # print('indices:',indices)
    # np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    # train_indices, test_indices = indices[:split], indices[split:]
    # print(split)
    # X_train, X_test = X[:train_indices], X[test_indices:]
    # y_train, y_test = y[:train_indices], y[test_indices:]
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test


def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
    b_f = np.zeros((hidden_size, 1))
    W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
    b_i = np.zeros((hidden_size, 1))
    W_c = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
    b_c = np.zeros((hidden_size, 1))
    W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
    b_o = np.zeros((hidden_size, 1))
    W_y = np.random.randn(output_size, hidden_size) * 0.01
    b_y = np.zeros((output_size, 1))
    return W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W_y, b_y


def lstm_forward(x, h_prev, c_prev, W_f, b_f, W_i, b_i, W_c, b_c, W_o, b_o, W_y, b_y):
    h_prev = h_prev.reshape(-1, 1)
    c_prev = c_prev.reshape(-1, 1)
    x = x.reshape(-1, 1)

    concat = np.vstack((x, h_prev))

    f_t = sigmoid(np.dot(W_f, concat) + b_f)
    i_t = sigmoid(np.dot(W_i, concat) + b_i)
    c_t_hat = tanh(np.dot(W_c, concat) + b_c)
    c_t = f_t * c_prev + i_t * c_t_hat
    o_t = sigmoid(np.dot(W_o, concat) + b_o)
    h_t = o_t * tanh(c_t)

    y_t = np.dot(W_y, h_t) + b_y

    cache = (
        f_t,
        i_t,
        c_t_hat,
        c_t,
        o_t,
        h_t,
        h_prev,
        c_prev,
        x,
        W_f,
        W_i,
        W_c,
        W_o,
        W_y,
    )
    return h_t, c_t, y_t, cache


def lstm_backward(dy, cache):
    f_t, i_t, c_t_hat, c_t, o_t, h_t, h_prev, c_prev, x, W_f, W_i, W_c, W_o, W_y = cache

    dy = dy.reshape(-1, 1)
    dh_t = np.dot(W_y.T, dy)
    do_t = dh_t * tanh(c_t) * sigmoid_derivative(o_t)
    dc_t = dh_t * o_t * tanh_derivative(tanh(c_t))
    dc_t += dc_t_next * f_t_next

    df_t = dc_t * c_prev * sigmoid_derivative(f_t)
    di_t = dc_t * c_t_hat * sigmoid_derivative(i_t)
    dc_t_hat = dc_t * i_t * tanh_derivative(c_t_hat)

    concat = np.vstack((x, h_prev))
    dW_f = np.dot(df_t, concat.T)
    db_f = df_t
    dW_i = np.dot(di_t, concat.T)
    db_i = di_t
    dW_c = np.dot(dc_t_hat, concat.T)
    db_c = dc_t_hat
    dW_o = np.dot(do_t, concat.T)
    db_o = do_t

    dx = (
        np.dot(W_f.T, df_t)
        + np.dot(W_i.T, di_t)
        + np.dot(W_c.T, dc_t_hat)
        + np.dot(W_o.T, do_t)
    )[:features, :]
    dh_prev = (
        np.dot(W_f.T, df_t)
        + np.dot(W_i.T, di_t)
        + np.dot(W_c.T, dc_t_hat)
        + np.dot(W_o.T, do_t)
    )[features:, :]
    dc_prev = dc_t * f_t

    return dx, dh_prev, dc_prev, dW_f, db_f, dW_i, db_i, dW_c, db_c, dW_o, db_o, dy


##### Long Short-Term Memory Network Class #####
##### https://medium.com/@CallMeTwitch/building-a-neural-network-zoo-from-scratch-the-long-short-term-memory-network-1cec5cf31b7


##### Helper Functions #####
def oneHotEncode(text):
    output = np.zeros((char_size, 1))
    output[char_to_idx[text]] = 1

    return output


# Xavier Normalized Initialization
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(
        6 / (input_size + output_size)
    )


# As is standard with this series, the only package used for the math
# behind this network is NumPy. TQDM has also been imported to add a
# progress bar to the training phase. If you’d prefer to keep this a
# NumPy only project, you can easily remove TQDM.
from tqdm import tqdm


class LSTM_fb:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        # Forget Gate
        self.wf = initWeights(input_size, hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        # Input Gate
        self.wi = initWeights(input_size, hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        # Candidate Gate
        self.wc = initWeights(input_size, hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        # Output Gate
        self.wo = initWeights(input_size, hidden_size)
        self.bo = np.zeros((hidden_size, 1))

        # Final Gate
        self.wy = initWeights(hidden_size, output_size)
        self.by = np.zeros((output_size, 1))

    # Reset Network Memory
    def reset(self):
        self.concat_inputs = {}

        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1: np.zeros((self.hidden_size, 1))}

        self.activation_outputs = {}
        self.candidate_gates = {}
        self.output_gates = {}
        self.forget_gates = {}
        self.input_gates = {}
        self.outputs = {}

    # Forward Propogation
    def forward(self, inputs):
        self.reset()

        outputs = []
        for q in range(len(inputs)):
            # print(len(self.hidden_states[q - 1]),'q:',q,self.hidden_states[q - 1])
            # print(len(self.hidden_states[q - 1]),'q:',q,self.hidden_states[q - 1])
            # print(len(inputs[q]),'q:',q,inputs[q])
            # sys.exit(0)
            # print(len(inputs),q)
            # print(len(inputs[q]),inputs[q])
            # self.concat_inputs[q] = np.concatenate((self.hidden_states[q - 1], inputs[q]))
            # print('q:',q)
            self.concat_inputs[q] = np.concatenate(
                (self.hidden_states[q - 1], inputs[q])
            )

            self.forget_gates[q] = ssigmoid(
                np.dot(self.wf, self.concat_inputs[q]) + self.bf
            )
            self.input_gates[q] = ssigmoid(
                np.dot(self.wi, self.concat_inputs[q]) + self.bi
            )
            self.candidate_gates[q] = ttanh(
                np.dot(self.wc, self.concat_inputs[q]) + self.bc
            )
            self.output_gates[q] = ssigmoid(
                np.dot(self.wo, self.concat_inputs[q]) + self.bo
            )

            self.cell_states[q] = (
                self.forget_gates[q] * self.cell_states[q - 1]
                + self.input_gates[q] * self.candidate_gates[q]
            )
            self.hidden_states[q] = self.output_gates[q] * ttanh(self.cell_states[q])

            outputs = outputs + [np.dot(self.wy, self.hidden_states[q]) + self.by]

        return outputs

    # Backward Propogation
    def backward(self, errors, inputs):
        d_wf, d_bf = 0, 0
        d_wi, d_bi = 0, 0
        d_wc, d_bc = 0, 0
        d_wo, d_bo = 0, 0
        d_wy, d_by = 0, 0
        dh_next, dc_next = np.zeros_like(self.hidden_states[0]), np.zeros_like(
            self.cell_states[0]
        )
        # dh_next, dc_next = np.zeros_like(self.hidden_states), np.zeros_like(self.cell_states)
        for q in reversed(range(len(inputs))):
            error = errors[q]

            # Final Gate Weights and Biases Errors
            d_wy += np.dot(error, self.hidden_states[q].T)
            d_by += error

            # Hidden State Error
            d_hs = np.dot(self.wy.T, error) + dh_next

            # Output Gate Weights and Biases Errors
            d_o = (
                ttanh(self.cell_states[q])
                * d_hs
                * ssigmoid(self.output_gates[q], derivative=True)
            )
            # d_o = tanh(self.cell_states[q]) * d_hs * sigmoid(self.output_gates[q])
            d_wo += np.dot(d_o, inputs[q].T)
            d_bo += d_o

            # Cell State Error
            d_cs = (
                ttanh(ttanh(self.cell_states[q]), derivative=True)
                * self.output_gates[q]
                * d_hs
                + dc_next
            )
            # d_cs = tanh(tanh(self.cell_states[q])) * self.output_gates[q] * d_hs + dc_next

            # Forget Gate Weights and Biases Errors
            d_f = (
                d_cs
                * self.cell_states[q - 1]
                * ssigmoid(self.forget_gates[q], derivative=True)
            )
            # d_f = d_cs * self.cell_states[q - 1] * sigmoid(self.forget_gates[q])
            d_wf += np.dot(d_f, inputs[q].T)
            d_bf += d_f

            # Input Gate Weights and Biases Errors
            d_i = (
                d_cs
                * self.candidate_gates[q]
                * ssigmoid(self.input_gates[q], derivative=True)
            )
            # d_i = d_cs * self.candidate_gates[q] * sigmoid(self.input_gates[q])
            d_wi += np.dot(d_i, inputs[q].T)
            d_bi += d_i

            # Candidate Gate Weights and Biases Errors
            d_c = (
                d_cs
                * self.input_gates[q]
                * ttanh(self.candidate_gates[q], derivative=True)
            )
            # d_c = d_cs * self.input_gates[q] * tanh(self.candidate_gates[q])
            d_wc += np.dot(d_c, inputs[q].T)
            d_bc += d_c

            # Concatenated Input Error (Sum of Error at Each Gate!)
            d_z = (
                np.dot(self.wf.T, d_f)
                + np.dot(self.wi.T, d_i)
                + np.dot(self.wc.T, d_c)
                + np.dot(self.wo.T, d_o)
            )

            # Error of Hidden State and Cell State at Next Time Step
            dh_next = d_z[: self.hidden_size, :]
            dc_next = self.forget_gates[q] * d_cs

        for d_ in (d_wf, d_bf, d_wi, d_bi, d_wc, d_bc, d_wo, d_bo, d_wy, d_by):
            # np.clip(d_, -1, 1, out = d_)
            d_ = np.clip(d_, -1, 1)

        self.wf = self.wf + d_wf * self.learning_rate
        self.bf = self.bf + d_bf * self.learning_rate

        self.wi = self.wi + d_wi * self.learning_rate
        self.bi = self.bi + d_bi * self.learning_rate

        self.wc = self.wc + d_wc * self.learning_rate
        self.bc = self.bc + d_bc * self.learning_rate

        self.wo = self.wo + d_wo * self.learning_rate
        self.bo = self.bo + d_bo * self.learning_rate

        self.wy = self.wy + d_wy * self.learning_rate
        self.by = self.by + d_by * self.learning_rate

    # Train
    def train(self, inputs, labels):
        inputs = [oneHotEncode(input) for input in inputs]

        for _ in tqdm(range(self.num_epochs)):
            predictions = self.forward(inputs)

            errors = []
            for q in range(len(predictions)):
                # print('predictions[q]',predictions[q])
                errors = errors + [-softmax(predictions[q])]
                errors[-1][char_to_idx[labels[q]]] = (
                    errors[-1][char_to_idx[labels[q]]] + 1
                )

            self.backward(errors, self.concat_inputs)

    # Test
    def test(self, inputs, labels):

        accuracy = 0
        probabilities = self.forward([oneHotEncode(input) for input in inputs])
        # print(type(probabilities),probabilities[:5])
        # my_l = [i for i, j in enumerate(probabilities) if j == 'np.nan']
        # if len(my_l) > 0 :
        #    print('NaN')
        #    print(probabilities[:10])
        #    sys.exit()
        output = []
        for q in range(len(labels)):
            prediction = idx_to_char[
                np.random.choice(
                    [*range(char_size)], p=softmax(probabilities[q].reshape(-1))
                )
            ]

            output = output + [prediction]

            if prediction == labels[q]:
                accuracy = accuracy + 1

        print(f"Ground Truth:\nt{labels}\n")
        # print(f'Predictions:\nt{"".join(output)}\n')
        s = ["{:.6f}".format(x) for x in output]

        print(f'Predictions:\nt{" ".join(s)}\n')

        print(f"Accuracy: {round(accuracy * 100 / len(inputs), 2)}%")

        return s


if __name__ == "__main__":
    # Example usage LSTM
    input_size = 10
    hidden_size = 20
    output_size = 5
    lstm = LSTM(input_size, hidden_size, output_size)
    xs = [np.random.randn(input_size, 1) for _ in range(5)]
    ys = lstm.predict(xs)
    print("LSTM:", ys)
    # Example usage BI-LSTM
    input_size = 10
    hidden_size = 20
    output_size = 5
    bilstm = BILSTM(input_size, hidden_size, output_size)
    xs = [np.random.randn(input_size, 1) for _ in range(5)]
    ys = bilstm.predict(xs)
    print("BI-LSTM", ys)
    # Example with Keras and Tensorflow
    # import tensorflow as tf
    # from tensorflow.keras.datasets import imdb
    # Load dataset
    # print('Load imdb Keras dataset. Please, wait...')
    # num_distinct_words = 5000
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_distinct_words)

    # Assume the last column is the target variable
    from numpy import genfromtxt

    data = genfromtxt("TourPerretNoHeader.csv", delimiter=";", comments="#")
    my_data = data[:, 3:4]  #
    # skip the first element which corresponds to the header
    # and flatten the input for compatibility with ACAMP_1
    # my_data = my_data[1:].reshape(-1)

    #
    # Normalize data to fit in [0:1]
    #
    input_size = 20
    hidden_size = 20
    output_size = 15

    XX = (my_data - np.min(my_data)) / (np.max(my_data) - np.min(my_data))
    X = XX[0 : output_size * hidden_size].flatten()

    lstm = True_LSTM(X, input_size, hidden_size, output_size)
    # xs = [np.random.randn(input_size, 1) for _ in range(5)]
    # We predict the first input_size values of the data set
    # that partially serve for training.
    XX = X[0:input_size]
    xs = [XX.reshape(input_size, 1) for _ in range(output_size)]
    ys = lstm.predict(xs)
    print("LSTM:", ys)

    import matplotlib.pyplot as plt
    from scipy.interpolate import BSpline
    from scipy.ndimage.filters import gaussian_filter1d

    labels = ["", "Input", "train_X", "train_y", "Prediction"]
    colors = ["y", "r", "g", "b", "gray"]

    # fig, axs = plt.subplots(5, 1, layout="constrained")
    fig = plt.figure(layout="constrained")
    axs = fig.subplots(5, 1, squeeze=False)

    axs[0, 0].plot(XX[0:output_size] * 0.01, label="Input")
    for j in range(output_size):
        ysmoothed = gaussian_filter1d([i[0] for i in ys[j]], sigma=2)
        axs[0, 0].plot(ysmoothed, label="pred smoothed")

        # 300 represents number of points to make between T.min and T.max
        # xnew = np.linspace(ys[j].min(), ys[j].max(), 300)
        # power_smooth = BSpline(ys[j], power, xnew)
        # plt.plot(xnew,power_smooth)

        axs[0, 0].plot([i[0] for i in ys[j]], label="pred not smoothed")

        break
    plt.legend()

    #
    # Other method
    #
    XX = (my_data - np.min(my_data)) / (np.max(my_data) - np.min(my_data))

    # Alternate 1:
    # X = XX[:6630].flatten()
    # train_X, X_test, train_y, y_test = train_test_split(
    #    X, X, test_size=0.33, random_state=137
    # )

    X = XX[:660].flatten()
    train_X, X_test, train_y, y_test = train_test_split(
        X, X, test_size=0.33, random_state=137
    )

    # Othewize, alternate 2:
    # X = XX[:150].flatten()
    # X1 = XX[150:300].flatten()
    # train_X, X_test, train_y, y_test = train_test_split(
    #    X, X1, test_size=0.33, random_state=137
    # )

    # print('train_x:',train_X)
    # print('train_y:',train_y)

    chars = set(np.array(X).flatten())

    data_size, char_size = len(X), len(chars)

    print(f"Data size: {data_size}, Char Size: {char_size}")

    char_to_idx = {c: i for i, c in enumerate(chars)}
    # print('char_to_idx:',char_to_idx)
    idx_to_char = {i: c for i, c in enumerate(chars)}
    # print('idx_to_char:',idx_to_char)

    # Initialize Network
    # hidden_size = 55 # Accuracy: 13.57%
    hidden_size = 5  # Accuracy: 94.8%
    # hidden_size = 20 # Accuracy: 15.84%
    # hidden_size = 15 # Accuracy: 92.31%
    # hidden_size = 1  # Accuracy: 9.05%

    lstm = LSTM_fb(
        input_size=char_size + hidden_size,
        hidden_size=hidden_size,
        output_size=char_size,
        num_epochs=1_000,
        learning_rate=0.05,
    )

    ##### Training #####
    lstm.train(train_X, train_y)

    ##### Testing #####
    myres = lstm.test(X_test, y_test)

    # xticks = np.arange(0, 700, 50)

    axs[1, 0].plot(X, color=colors[1], label=labels[1])
    axs[1, 0].legend(loc="upper right")

    foo = data_size - len(train_X.tolist())

    axs[2, 0].plot(
        train_X.tolist() + np.zeros(foo).tolist(), color=colors[2], label=labels[2]
    )
    axs[2, 0].legend(loc="upper right")

    axs[3, 0].plot(
        train_y.tolist() + np.zeros(foo).tolist(), color=colors[3], label=labels[3]
    )
    axs[3, 0].legend(loc="upper right")

    values = np.arange(0.2, 0.6, 0.1)
    value_increment = 1.0
    # print('values:',values)

    foo = data_size - len(myres)
    axs[4, 0].plot(np.zeros(foo).tolist() + myres, color=colors[4], label=labels[4])
    axs[4, 0].legend(loc="upper left")
    axs[4, 0].set_yticks([])

    plt.show()
