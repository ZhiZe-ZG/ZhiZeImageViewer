import numpy as np
import cv2 as cv


def RGB2BGR(image: np.ndarray) -> np.ndarray:
    """
    convert RGB image to bgr image
    """
    return cv.cvtColor(image, cv.COLOR_RGB2BGR)


def RGB2GRAY(image: np.ndarray) -> np.ndarray:
    """
    convert RGB to gray image
    """
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)
