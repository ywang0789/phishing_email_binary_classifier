"""ONLY USED FOR TESTING becausee real dataset very large"""

import csv

REAL_DATASET_FILE_PATH = "./data/Phishing_Email.csv"


class Sampler:
    def __init__(self, input_file_path):
        self._input_file_path = input_file_path

    def create_sample(self, output_file, sample_size):
        """
        reads a number of rows from the input file and writes them to the output file
        """
        try:
            with open(self._input_file_path, "r") as input_csv:
                reader = csv.reader(input_csv)
                with open(output_file, "w") as output_csv:
                    writer = csv.writer(output_csv)
                    for i, row in enumerate(reader):
                        writer.writerow(row)
                        if i == sample_size:
                            break
        except Exception as e:
            print(f"Error creating sample: {e}")


if __name__ == "__main__":
    output_csv = "./test/sample_dataset.csv"
    sample_size = 10

    sampler = Sampler(REAL_DATASET_FILE_PATH)
    sampler.create_sample(output_csv, sample_size)
