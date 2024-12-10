import json

import numpy
from DatasetReader import DatasetReader
from Preprocessor import Preprocessor

VOCABULARY_FILE_PATH = "./data/vocabulary.json"


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
            email_text, label = self._reader.get_row(i)
            processed_text = self._preprocessor.process(email_text)
            for word in processed_text:
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)

        with open(self.volcabulary_file_path, "w") as file:
            json.dump(vocabulary, file)


if __name__ == "__main__":
    dataset = Dataset()
    dataset.build_vocabulary()
    print("Vocabulary built successfully")
