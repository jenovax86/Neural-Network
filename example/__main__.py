import os
import pickle

from model.data_loader import DataLoader


def show_prediction_vs_actual_number(predicted_output, real_output):
    for predicted, actual in zip(predicted_output, real_output):
        print(f"predicted output: {predicted}, real output: {actual}")


def main():
    LIMIT_SIZE = 1024
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

    output_number = network_load.feed_forward(x_test[:LIMIT_SIZE])
    predicted_numbers = network_load.predict_number(output_number)
    actual_numbers = network_load.predict_number(y_test)
    show_prediction_vs_actual_number(predicted_numbers, actual_numbers)


if __name__ == "__main__":
    main()
