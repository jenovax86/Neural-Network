import struct
import numpy as np


class DataLoader:
    def __init__(
        self,
        test_images_filepath,
        test_labels_filepath,
        train_image_filepath,
        train_label_filepath,
    ):
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
        self.train_image_filepath = train_image_filepath
        self.train_label_filepath = train_label_filepath

    def read_images(self, filepath):
        with open(filepath, "rb") as file:
            _, number_images, rows, columns = struct.unpack(">IIII", file.read(16))
            file_data = np.frombuffer(file.read(), dtype=np.uint8)
            data = file_data.reshape(number_images, rows * columns)
            return data

    def read_labels(self, filepath):
        with open(filepath, "rb") as file:
            _, _ = struct.unpack(">II", file.read(8))
            label = np.frombuffer(file.read(), dtype=np.uint8)

        return label

    def load_training_data(self):
        input_train = self.read_images(self.train_image_filepath)
        output_train = self.read_labels(self.train_label_filepath)
        return input_train, output_train

    def load_test_data(self):
        input_test = self.read_images(self.test_images_filepath)
        output_test = self.read_labels(self.test_labels_filepath)
        return input_test, output_test
