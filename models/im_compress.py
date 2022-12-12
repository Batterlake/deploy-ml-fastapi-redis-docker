import base64
import json

import cv2
import numpy as np


def resize_image(image, desired_width):
    current_width = image.shape[1]
    scale_percent = desired_width / current_width
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def compress_image(image, grayscale=True, desired_width=480):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = resize_image(image, desired_width)
    return image


def image_from_base64(string: str):
    bytes_ = base64.b64decode(string.encode("utf-8"))
    return image_from_bytes(bytes_)


def image_from_bytes(byte_im: bytes):
    nparr = np.frombuffer(byte_im, np.uint8)
    gsimg = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_np = np.stack([gsimg for _ in range(3)], axis=-1)
    return img_np


def image_to_jpeg_nparray(image, quality=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
    is_success, im_buf_arr = cv2.imencode(".jpg", image, quality)
    return im_buf_arr


def image_to_jpeg_bytes(image, quality=[int(cv2.IMWRITE_JPEG_QUALITY), 95]):
    buf = image_to_jpeg_nparray(image, quality)
    byte_im = buf.tobytes()
    return byte_im


def optimize_to_send(image, decode=True):
    reduced = compress_image(image)
    byte_im = image_to_jpeg_bytes(reduced)
    # encode image
    img_enc = base64.b64encode(byte_im)
    if decode:
        img_enc = img_enc.decode("utf-8")
    img_dump = json.dumps({"img": img_enc})
    return img_dump
