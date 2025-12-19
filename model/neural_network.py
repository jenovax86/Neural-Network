import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_scale = 0.01

        self.input_hidden_weights = (
            np.random.randn(self.input_size, self.hidden_size) * self.weight_scale
        )
        self.hidden_output_weights = (
            np.random.randn(self.hidden_size, self.output_size) * self.weight_scale
        )

        self.hidden_layer_bias = np.zeros(self.hidden_size)
        self.output_layer_bias = np.zeros(self.output_size)

    def feed_forward(self, input_array):
        self.hidden_layer_output = self.sigmoid(
            np.dot(np.array(input_array, dtype=float), self.input_hidden_weights)
            + self.hidden_layer_bias
        )
        self.predicted_output = self.sigmoid(
            np.dot(self.hidden_layer_output, self.hidden_output_weights)
            + self.output_layer_bias
        )
        return self.predicted_output

    def backpropagation(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2)
        targets = np.array(targets, ndmin=2)
        output_delta = (self.predicted_output - targets) * self.sigmoid_derivative(
            self.predicted_output
        )
        hidden_delta = np.dot(
            output_delta, self.hidden_output_weights.T
        ) * self.sigmoid_derivative(self.hidden_layer_output)

        self.input_hidden_weights -= self.learning_rate * np.dot(inputs.T, hidden_delta)
        self.hidden_output_weights -= self.learning_rate * np.dot(
            self.hidden_layer_output.T, output_delta
        )

        self.hidden_layer_bias -= self.learning_rate * hidden_delta.sum(axis=0)
        self.output_layer_bias -= self.learning_rate * output_delta.sum(axis=0)

    def create_mini_batches(self, input, target, batch_size):
        mini_batches = []
        num_rows_in_input = input.shape[0]
        indices = np.random.permutation(num_rows_in_input)
        input_shuffled = input[indices]
        target_shuffled = target[indices]

        for index in range(0, num_rows_in_input, batch_size):
            input_batch = input_shuffled[index : index + batch_size]
            target_batch = target_shuffled[index : index + batch_size]
            mini_batches.append((input_batch, target_batch))
        return mini_batches

    def stochastic_gradient_descent(self, inputs, targets, epochs, batch_size=128):
        limit_size = 1024
        for epoch in range(epochs):
            mini_batches = self.create_mini_batches(inputs, targets, batch_size)
            for input_batch, target_batch in mini_batches:
                self.feed_forward(input_batch)
                self.backpropagation(input_batch, target_batch)

            if epoch % 10 == 0:
                prediction = self.feed_forward(inputs[:limit_size])
                print(
                    f" Epoch: {epoch}, Loss: {self.compute_loss(prediction, targets[:limit_size])}"
                )

    def train_model(self, inputs, targets, epochs):
        for epoch in range(epochs):
            prediction = self.feed_forward(inputs)
            self.backpropagation(inputs, targets)
            if epoch % 10 == 0:
                print(
                    f" Epoch: {epoch}, Loss: {self.compute_loss(prediction, targets)}"
                )

    def predict_number(self, input):
        return np.argmax(input, axis=1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, sigmoid_function):
        return sigmoid_function * (1 - sigmoid_function)

    def compute_loss(self, predicted_output, target_output):
        return np.mean(np.square(predicted_output - target_output))
