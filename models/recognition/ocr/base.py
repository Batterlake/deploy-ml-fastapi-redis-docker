import os
from typing import Any, List

import cv2
import numpy as np
import torch

from .config import device_torch, modelhub
from .npocrnet import NPOcrNet
from .ocr_tools import StrLabelConverter, decode_batch
from .utils import normalize_img


class OCRError(Exception):
    ...


class OCR(object):
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self) -> None:
        # model
        self.dm = None
        self.model = None
        self.trainer = None
        self.letters = []
        self.max_text_len = 0

        # Input parameters
        self.max_plate_length = 0
        self.height = 50
        self.width = 200
        self.color_channels = 3
        self.label_length = 13

        # Train hyperparameters
        self.batch_size = 32
        self.epochs = 1
        self.gpus = 1

        self.label_converter = None
        self.path_to_model = None

    def init_label_converter(self):
        self.label_converter = StrLabelConverter(
            "".join(self.letters), self.max_text_len
        )

    def create_model(self) -> NPOcrNet:
        """
        TODO: describe method
        """
        self.model = NPOcrNet(
            self.letters,
            letters_max=len(self.letters) + 1,
            label_converter=self.label_converter,
            max_plate_length=self.max_plate_length,
        )
        # self.model.apply(weights_init)
        self.model = self.model.to(device_torch)
        return self.model

    def recreate_model(self) -> torch.nn.Module:
        if self.model is None:
            self.create_model()

    def preprocess(self, imgs):
        xs = []
        for img in imgs:
            x = normalize_img(img, width=self.width, height=self.height)
            xs.append(x)
        xs = np.moveaxis(np.array(xs), 3, 1)
        xs = torch.tensor(xs)
        xs = xs.to(device_torch)
        return xs

    def forward(self, xs):
        return self.model(xs)

    def postprocess(self, net_out_value):
        net_out_value = [p.cpu().numpy() for p in net_out_value]
        pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        return pred_texts

    @torch.no_grad()
    def predict(self, xs: List or torch.Tensor, return_acc: bool = False) -> Any:
        net_out_value = self.model(xs)
        net_out_value = [p.cpu().numpy() for p in net_out_value]
        pred_texts = decode_batch(torch.Tensor(net_out_value), self.label_converter)
        pred_texts = [pred_text.upper() for pred_text in pred_texts]
        if return_acc:
            if len(net_out_value):
                net_out_value = np.array(net_out_value)
                net_out_value = net_out_value.reshape(
                    (
                        net_out_value.shape[1],
                        net_out_value.shape[0],
                        net_out_value.shape[2],
                    )
                )
            return pred_texts, net_out_value
        return pred_texts

    def load_model(self, path_to_model, nn_class=NPOcrNet):
        self.path_to_model = path_to_model
        self.recreate_model()
        self.model.load_state_dict(
            torch.load(path_to_model, map_location=torch.device("cpu"))["state_dict"],
        )
        self.model = self.model.to(device_torch)  # type: ignore
        self.model.eval()
        return self.model

    def load(self, path_to_model: str = "latest", nn_class=NPOcrNet) -> NPOcrNet:
        """
        TODO: describe method
        """
        self.create_model()
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.get_classname())
            path_to_model = model_info["path"]
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(
                path_to_model, self.get_classname(), self.get_classname()
            )
            path_to_model = model_info["path"]

        return self.load_model(path_to_model, nn_class=nn_class)


if __name__ == "__main__":
    det = OCR()
    det.get_classname = lambda: "Eu"
    det.letters = [
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
    det.max_text_len = 9
    det.max_plate_length = 9
    det.letters_max = len(det.letters) + 1
    det.init_label_converter()
    det.load()

    image_path = os.path.join(
        os.getcwd(), "./data/examples/numberplate_zone_images/JJF509.png"
    )
    img = cv2.imread(image_path)
    xs = det.preprocess([img])
    y = det.predict(xs)
    print("y", y)

    image_path = os.path.join(
        os.getcwd(), "./data/examples/numberplate_zone_images/RP70012.png"
    )
    img = cv2.imread(image_path)
    xs = det.preprocess([img])
    y = det.predict(xs)
    print("y", y)
