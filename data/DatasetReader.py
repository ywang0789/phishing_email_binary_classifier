"""
helper to read csv files
"""

import pandas

DATASET_FILE_PATH = "./data/Phishing_Email.csv"
DATASET_TRAIN_FILE_PATH = "./data/dataset_train.csv"
DATASET_TEST_FILE_PATH = "./data/dataset_test.csv"


class DatasetReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.num_rows = self._get_total_rows()

    def get_row(self, row_index: int) -> tuple:
        """
        get a row from the dataset file by index
        returns the 'Email Text' and 'Email Type' of the row
        """

        try:
            dataset = pandas.read_csv(self.file_path)
            email_text = dataset.iloc[row_index]["Email Text"]
            email_type = dataset.iloc[row_index]["Email Type"]
            return email_text, email_type
        except Exception as e:
            print(f"Error get_row: {e}")
            return "", ""

    def _get_total_rows(self) -> int:
        """
        get total number of rows in the dataset file
        """
        try:
            dataset = pandas.read_csv(self.file_path)
            return len(dataset)
        except Exception as e:
            print(f"Error _get_total_rows: {e}")


if __name__ == "__main__":
    reader = DatasetReader()
    print(reader.num_rows)
    print(reader.get_row(1000)[1])
