import json
import string

import numpy

STOP_WORDS_FILE_PATH = "./data/stop_words_english.json"


class Preprocessor:
    def __init__(self):

        self.stop_words_file_path = STOP_WORDS_FILE_PATH
        self.stop_words = self._load_stop_words()

    def process(self, text: str) -> numpy.ndarray:
        """
        process text and return a numpy array
        """
        processed_text = self._clean(text)
        return numpy.array(processed_text)

    def _clean(self, text: str) -> list:
        """
        clean text and tokenize text. lowercase, remove special characters, remove numbers
        """
        if not isinstance(text, str):
            text = str(text)
        text = self._remove_specials(text)
        text = text.lower()
        tokens = self._tokenize(text)
        tokens = self._remove_stop_words(tokens)
        tokens = self._remove_numbers(tokens)
        return tokens

    def _remove_specials(self, text: str) -> str:
        """
        remove special characters/puncuations from text
        """
        for char in string.punctuation:
            text = text.replace(char, "")
        return text

    def _tokenize(self, text: str) -> list:
        """
        turn string text into a list of words/tokens
        """
        return text.split()

    def _remove_numbers(self, tokens: list) -> list:
        """
        remove numbers/digits from tokens
        """
        for token in tokens:
            if token.isdigit():
                tokens.remove(token)
        return tokens

    def _load_stop_words(self) -> set:
        """
        load stop words from the JSON file
        """
        with open(self.stop_words_file_path, "r", encoding="utf-8") as file:
            stop_words = json.load(file)
        return set(stop_words)

    def _remove_stop_words(self, tokens: list) -> list:
        """
        remove stop words from tokens
        """
        for token in tokens:
            if token in self.stop_words:
                tokens.remove(token)

        return tokens


if __name__ == "__main__":
    preprocessor = Preprocessor()
    text = "Hello, I am a string with numbers 1234 and special characters !@#"
    print(preprocessor.process(text))
