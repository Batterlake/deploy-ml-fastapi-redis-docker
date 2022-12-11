from typing import List

import cv2
import numpy as np

from .base import OCR


class LangWrapper:
    def __init__(self, ocr: OCR):
        self.model = ocr
        self.model.load()

    def predict_images(self, images: List[np.ndarray]) -> List[str]:
        prep = self.model.preprocess(images)
        return self.model.predict(prep)

    def predict_filenames(self, filenames: List[str]) -> List[str]:
        images = [cv2.imread(fn, cv2.IMREAD_COLOR) for fn in filenames]
        return self.predict_images(images)


class Eu(OCR):
    def __init__(self) -> None:
        OCR.__init__(self)
        self.letters = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]
        self.max_text_len = 9
        self.max_plate_length = 9
        self.letters_max = len(self.letters) + 1

        self.init_label_converter()


class Ru(OCR):
    def __init__(self):
        OCR.__init__(self)
        self.letters = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "E",
            "H",
            "K",
            "M",
            "O",
            "P",
            "T",
            "X",
            "Y",
        ]
        self.max_text_len = 9
        self.max_plate_length = 9
        self.letters_max = len(self.letters) + 1
        self.init_label_converter()
