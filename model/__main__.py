import os
import pickle


from model.data_loader import DataLoader
from model.neural_network import NeuralNetwork


def main():
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_images = os.path.join(base_directory, "data", "t10k-images.idx3-ubyte")
    test_labels = os.path.join(base_directory, "data", "t10k-labels.idx1-ubyte")
    train_images = os.path.join(base_directory, "data", "train-images.idx3-ubyte")
    train_labels = os.path.join(base_directory, "data", "train-labels.idx1-ubyte")
    loader = DataLoader(test_images, test_labels, train_images, train_labels)
    x_train, y_train = loader.load_training_data()
    size_of_input_array = x_train.shape[1]
    neural_network = NeuralNetwork(
        input_size=size_of_input_array, hidden_size=25, output_size=10
    )
    neural_network.stochastic_gradient_descent(
        inputs=x_train, targets=y_train, epochs=40
    )

    with open("neural_network.pkl", "wb") as file:
        pickle.dump(neural_network, file)


if __name__ == "__main__":
    main()
