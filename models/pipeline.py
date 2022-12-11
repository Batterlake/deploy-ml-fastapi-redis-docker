from typing import List
from PIL import Image
import numpy as np
import cv2

from detection.yolo import Detector
from recognition.faces import get_face_embeddings
from recognition.plates import recognize_plates


class Pipeline:
    def __init__(self, device: str='cpu') -> None:
        self._detector = Detector(device=device)
    
    def run(self, image: np.ndarray)
        pass


if __name__ == "__main__":
    d = Detector()
    image = Image.open('test.jpg')
    pred = d.predict(image)
    faces = pred['faces']
    plates = pred['plates']

    embeddings = get_face_embeddings(faces)
    plate_texts = recognize_plates(plates)

    print(len(faces), len(plates))
    print(plate_texts)