"""
Used to build the vocabulary and split the dataset into training and testing sets
"""

import json

import numpy
import pandas
from DatasetReader import DatasetReader
from Preprocessor import Preprocessor

VOCABULARY_FILE_PATH = "./data/vocabulary.json"
TRAIN_FILE_PATH = "./data/dataset_train.csv"
TEST_FILE_PATH = "./data/dataset_test.csv"


class Dataset:
    def __init__(self):
        self._reader = DatasetReader()
        self._preprocessor = Preprocessor()
        self.volcabulary_file_path = VOCABULARY_FILE_PATH

    def build_vocabulary(self):
        """
        build vocabulary from the dataset
        iterate through the dataset and build a vocabulary
        saves the vocabulary to a file
        """
        vocabulary = {}
        for i in range(self._reader.num_rows):
            print(f"Processing row {i + 1} out of {self._reader.num_rows}")
            email_text, label = self._reader.get_row_from_train(i)
            processed_text = self._preprocessor.process(email_text)
            for word in processed_text:
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)

        with open(self.volcabulary_file_path, "w") as file:
            json.dump(vocabulary, file)

    def split_dataset(self):
        """
        split the dataset into training and testing sets
        saves the splits to separate files
        """
        dataset = pandas.read_csv(self._reader.file_path)
        train_size = int(len(dataset) * 0.8)
        train_set = dataset[:train_size]
        test_set = dataset[train_size:]

        print(f"Saving training set with {len(train_set)} rows to {TRAIN_FILE_PATH}")
        train_set.to_csv(TRAIN_FILE_PATH, index=False)
        print(f"Saving testing set with {len(test_set)} rows to {TEST_FILE_PATH}")
        test_set.to_csv(TEST_FILE_PATH, index=False)


if __name__ == "__main__":
    dataset = Dataset()
    # dataset.build_vocabulary()
    # dataset.split_dataset()
