from os import listdir
from os.path import isfile, join

import cv2
import numpy as np

from utils import overlay_transparent, rotate_image, drawDebugPoints, getAngle
from constants import INDICIES

class BodyPart:
    def __init__(self, image, originalImage, xpos, ypos) -> None:
        self.image = image
        self.originalImage = originalImage
        self.xpos, self.ypos = xpos, ypos

    def get(self):
        return self.image, self.originalImage, self.xpos, self.ypos

class Character:
    def __init__(self, filepath) -> None:
        self.filepath = filepath 
        self.bodyparts = self.readBodyparts(self.filepath)

    def readBodyparts(self, filepath):
        files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
        bodyparts = {}
        for f in files:
            key = "".join(f.split('-')[1:])[:-4]

            image = cv2.imread(join(filepath,f), cv2.IMREAD_UNCHANGED)
            longer = max(image.shape[0], image.shape[1])
            tmpImg = np.zeros((longer*2, longer*2, image.shape[2]))

            xpos = (tmpImg.shape[1] // 2) - image.shape[1] // 2
            ypos = (tmpImg.shape[0] // 2) - image.shape[0] // 2

            tmpImg[ypos:ypos+image.shape[0], xpos:xpos+image.shape[1]] = image

            bodyparts[key] = BodyPart(tmpImg, image, xpos, ypos)

        print(bodyparts)
        return bodyparts


    def drawCharacter(self, keypoints, frame):

        #LEFT ARM LOWER
        bp, _, xpos, ypos = self.bodyparts['armlower'].get()

        bp = drawDebugPoints(bp)

        p1 = keypoints[INDICIES["leftElbow"]]
        p2 = keypoints[INDICIES["leftWrist"]]
        diff = p2 - p1
        size = np.linalg.norm(diff)
        angle = getAngle(diff)



        

        bp = rotate_image(bp, angle)
        frame = overlay_transparent(frame, bp, int(p1[1])-ypos, int(p1[0])-xpos)

        #bp = cv2.resize(bp, (bp.shape[0]+self.angle, bp.shape[1]+self.angle))
