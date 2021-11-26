import cv2
from flask.wrappers import Request
import numpy as np


def decode_image_request_to_ndarray(request: Request) -> np.ndarray:
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img
