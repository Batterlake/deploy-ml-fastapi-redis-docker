from typing import List
import numpy as np

from .ocr.langs import Ru, Eu, LangWrapper


_model = LangWrapper(Ru())

def recognize_plates(images: List[np.ndarray]) -> List[str]:
    return _model.predict_images(images)