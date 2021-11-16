import cv2
import numpy as np

from constants import INDICIES, SEGMENTS

class Drawing:
    def __init__(self) -> None:
        pass

    def drawStickman(self, kps, image):
        flipped = np.int32(np.flip(kps, axis=1))
        cv2.fillPoly(image, [flipped[SEGMENTS["torso"]]], (0,0,0))

        cv2.polylines(image, [flipped[SEGMENTS["leftArm"]], 
                            flipped[SEGMENTS["rightArm"]],
                            flipped[SEGMENTS["leftLeg"]],
                            flipped[SEGMENTS["rightLeg"]]], 0, (0,0,0), 7)
