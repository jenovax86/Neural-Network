import os
import pickle

from model.data_loader import DataLoader


def main():
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pickle_file_path = os.path.join(base_directory, "neural_network.pkl")
    test_images = os.path.join(base_directory, "data", "t10k-images.idx3-ubyte")
    test_labels = os.path.join(base_directory, "data", "t10k-labels.idx1-ubyte")
    train_images = os.path.join(base_directory, "data", "train-images.idx3-ubyte")
    train_labels = os.path.join(base_directory, "data", "train-labels.idx1-ubyte")
    loader = DataLoader(test_images, test_labels, train_images, train_labels)
    x_test, y_test = loader.load_test_data()
    with open(pickle_file_path, "rb") as file:
        network_load = pickle.load(file)
    output_number = network_load.feed_forward(x_test[:1024])
    number = network_load.predict_number(output_number)
    print(number)


if __name__ == "__main__":
    main()
