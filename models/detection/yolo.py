from pathlib import Path
from typing import Dict, List

import numpy as np
import yolov5

WEIGHTS = Path(__file__).parent / "weights.pt"


class Detector:
    def __init__(self, weights: str = WEIGHTS, device="cpu", thres=0.3) -> None:
        self._model = yolov5.load(weights, device=device)
        self._model = self._model.eval()
        self._model.conf = thres
        self._labels = {0.0: "plates", 1.0: "faces"}

    def predict(self, image: np.ndarray) -> Dict[str, List[np.ndarray]]:
        preds = self._model(image)
        crops = preds.crop(save=False)
        crops_by_classes = {cls_name: [] for cls_name in self._labels.values()}

        for crop in crops:
            crops_by_classes[self._labels[crop["cls"].item()]].append(crop["im"])

        return crops_by_classes
