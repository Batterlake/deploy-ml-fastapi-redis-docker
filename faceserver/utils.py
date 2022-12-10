import base64
import io
import sys

import numpy as np
from PIL import Image


def decode_image(image_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(image_bytes)))


def base64_decode_image(a, dtype, shape):
    """
    image = base64_decode_image(
                q["image"],
                os.environ.get("IMAGE_DTYPE"),
                (
                    1,
                    int(os.environ.get("IMAGE_HEIGHT")),
                    int(os.environ.get("IMAGE_WIDTH")),
                    int(os.environ.get("IMAGE_CHANS")),
                ),
            )

    """
    # If this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    # Convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # Return the decoded image
    return a