import numpy as np

class NeuralNetwork:
    def __init__(self, input):
        self.input = np.array(input)
        self.size = len(input)
        self.weight = np.random.rand(self.size) 
        self.bias = np.random.rand(self.size)


    def feed_forward(self):
        formula = self.sigmoid(np.dot(self.input, self.weight) + self.bias)
        return formula

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))


neural_network = NeuralNetwork([0, 2 , 3 ])
