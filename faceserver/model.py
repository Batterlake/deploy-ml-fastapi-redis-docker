# before poetry
import cv2
import numpy as np

from deepface import DeepFace
from deepface.commons import functions
from deepface.detectors import FaceDetector


def preprocess_face_image(
    img,
    target_size=(
        112,
        112,
    ),
    grayscale=False,
):
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    # Then pad the other side to the target size by adding black pixels
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    if grayscale == False:
        # Put the base image in the middle of the padded image
        img = np.pad(
            img,
            (
                (diff_0 // 2, diff_0 - diff_0 // 2),
                (diff_1 // 2, diff_1 - diff_1 // 2),
                (0, 0),
            ),
            "constant",
        )
    else:
        img = np.pad(
            img,
            ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)),
            "constant",
        )

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    img_pixels = np.expand_dims(img, axis=0).astype(np.float32)
    img_pixels /= 255  # normalize input in [0, 1]

    return img_pixels


def detect_mul_faces(face_detector, detector_backend, img, align=True):

    obj = FaceDetector.detect_faces(face_detector, detector_backend, img, align)
    faces = []
    regions = []
    for o in obj:
        faces.append(o[0])
        regions.append(o[1])

    return faces, regions


def detect_face(img, backend, align=True):

    # detector stored in a global variable in FaceDetector object.
    # this call should be completed very fast because it will return found in memory
    # it will not build face detector model in each call (consider for loops)
    face_detector = FaceDetector.build_model(backend)
    detected_faces = None
    img_regions = None
    detected_faces, img_regions = detect_mul_faces(face_detector, backend, img, align)
    return detected_faces, img_regions


def prepare_images(images, target_size=(112, 112)):
    tensors = np.zeros([len(images), *target_size, 3], dtype=np.float32)
    for i, im in enumerate(images):
        tensors[i] = preprocess_face_image(im, target_size=target_size)

    return tensors


def recognize_faces(images, backend):
    recognizer = DeepFace.build_model(backend)
    batch = prepare_images(images)
    tensors = functions.normalize_input(img=batch, normalization="base")
    embedding = recognizer(tensors).numpy()
    return embedding


def detect_embedd(image, detector_backend, recognizer_backend, align=False):
    faces, locations = detect_face(image, detector_backend, align=align)

    embeddings = None
    result = image.copy()
    if faces is not None and len(faces) != 0:
        result = faces.copy()
        embeddings = recognize_faces(faces, recognizer_backend)

    return result, locations, embeddings
