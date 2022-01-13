from os import listdir
from os.path import isfile, join
import json
from typing import OrderedDict

import cv2
import numpy as np

from utils import drawDebugPoints, getRotatedImageWithAnchor, overlay_transparent, getAngle, getRotatedImageWithAnchor2
from constants import INDICIES

class BodyPart:
    def __init__(self, image) -> None:
        self.image = image

class Character:
    def __init__(self, filepath) -> None:
        self.filepath = filepath 
        self.bodyparts = self.readBodyparts(self.filepath)
        self.angle=0

    def readBodyparts(self, filepath):
        files = [f for f in listdir(filepath) if isfile(join(filepath, f)) and ".png" in f]
        bodyparts = {}
        for f in files:
            key = "".join(f.split('-')[1:])[:-4]

            image = cv2.imread(join(filepath,f), cv2.IMREAD_UNCHANGED)
            bodyparts[key] = BodyPart(image)

        with open(join(filepath, "rigging.json"), 'r') as f:
            self.rigging = json.load(f)

        print(self.rigging)
        return bodyparts


    def drawCharacter(self, keypoints, frame):
        self.angle += 1
        #TORSO IS ANCHOR
        p1 = keypoints[INDICIES["leftShoulder"]]
        #torsopos = p1
        torsopos = np.array([200, 200])

        #LEFT ARM UPPER
        bp = self.bodyparts['armupper'].image

        anchor1 = np.array((20, bp.shape[1]//2))
        anchor2 = np.array((bp.shape[0]-20, bp.shape[1]//2))

        offset = np.array(self.rigging['left-arm-upper'])

        p2 = keypoints[INDICIES["leftElbow"]]
        p1 = keypoints[INDICIES["leftShoulder"]]
        diff = p1 - p2
        angle, anglerad = getAngle(diff)
        #angle, anglerad = self.angle, np.deg2rad(self.angle)

        leftArmUpper, anchor1, anchor2 = getRotatedImageWithAnchor2(bp, anchor1, anchor2, angle, anglerad)

        leftShoulder = [int(torsopos[0] + offset[0] - anchor1[0]), 
                        int(torsopos[1] + offset[1] - anchor1[1])]


        #LEFT ARM LOWER
        bp = self.bodyparts['armlower'].image

        anchor = np.array((20, bp.shape[1]//2))

        offset = np.array(self.rigging['left-arm-lower'])

        p2 = keypoints[INDICIES["leftElbow"]]
        p1 = keypoints[INDICIES["leftWrist"]]
        diff = p1 - p2
        angle, anglerad = getAngle(diff)

        bp, anchor = getRotatedImageWithAnchor(bp, anchor, angle, anglerad)

        leftElbow = [int(leftShoulder[0] + anchor2[0] - anchor[0]), 
                      int(leftShoulder[1] + anchor2[1] - anchor[1])]

        frame = overlay_transparent(frame, bp, leftElbow[1], leftElbow[0])
        frame = overlay_transparent(frame, leftArmUpper, leftShoulder[1], leftShoulder[0])

        #LOWER TORSO
        bp = self.bodyparts['torsolower'].image
        offset = np.array(self.rigging['torso-lower'])
        frame = overlay_transparent(frame, bp, int(torsopos[1] + offset[1]), int(torsopos[0] + offset[0]))

        #TORSO
        bp = self.bodyparts['torsoupper'].image
        frame = overlay_transparent(frame, bp, int(torsopos[1]), int(torsopos[0]))

        #RIGHT ARM UPPER
        bp = self.bodyparts['armupper'].image

        anchor1 = np.array((20, bp.shape[1]//2))
        anchor2 = np.array((bp.shape[0]-20, bp.shape[1]//2))

        offset = np.array(self.rigging['right-arm-upper'])

        p2 = keypoints[INDICIES["rightElbow"]]
        p1 = keypoints[INDICIES["rightShoulder"]]
        diff = p1 - p2
        angle, anglerad = getAngle(diff)
        #angle, anglerad = self.angle, np.deg2rad(self.angle)

        rightArmUpper, anchor1, anchor2 = getRotatedImageWithAnchor2(bp, anchor1, anchor2, angle, anglerad)

        rightShoulder = [int(torsopos[0] + offset[0] - anchor1[0]), 
                        int(torsopos[1] + offset[1] - anchor1[1])]

        #RIGHT ARM LOWER
        bp = self.bodyparts['armlower'].image

        anchor = np.array((20, bp.shape[1]//2))

        offset = np.array(self.rigging['right-arm-lower'])

        p2 = keypoints[INDICIES["rightElbow"]]
        p1 = keypoints[INDICIES["rightWrist"]]
        diff = p1 - p2
        angle, anglerad = getAngle(diff)

        bp, anchor = getRotatedImageWithAnchor(bp, anchor, angle, anglerad)

        rightElbow = [int(rightShoulder[0] + anchor2[0] - anchor[0]), 
                      int(rightShoulder[1] + anchor2[1] - anchor[1])]

        frame = overlay_transparent(frame, bp, rightElbow[1], rightElbow[0])

        #Draw right upper arm over lower
        frame = overlay_transparent(frame, rightArmUpper, rightShoulder[1], rightShoulder[0])


        #RIGHT LEG UPPER
        bp = cv2.flip(self.bodyparts['legupper'].image, 1)

        anchor1 = np.array((20, bp.shape[1]//2))
        anchor2 = np.array((bp.shape[0]-20, bp.shape[1]//2))

        offset = np.array(self.rigging['right-leg-upper'])

        p2 = keypoints[INDICIES["rightHip"]]
        p1 = keypoints[INDICIES["rightKnee"]]
        diff = p2 - p1
        angle, anglerad = getAngle(diff)
        #angle, anglerad = self.angle, np.deg2rad(self.angle)

        rightLegUpper, anchor1, anchor2 = getRotatedImageWithAnchor2(bp, anchor1, anchor2, angle, anglerad)

        rightHip = [int(torsopos[0] + offset[0] - anchor1[0]), 
                        int(torsopos[1] + offset[1] - anchor1[1])]


        #RIGHT LEG LOWER
        bp = self.bodyparts['leglower'].image

        anchor = np.array((20, bp.shape[1]//2))

        offset = np.array(self.rigging['right-leg-lower'])

        p2 = keypoints[INDICIES["rightAnkle"]]
        p1 = keypoints[INDICIES["rightKnee"]]
        diff = p1 - p2
        angle, anglerad = getAngle(diff)

        bp, anchor = getRotatedImageWithAnchor(bp, anchor, angle, anglerad)

        rightKnee = [int(rightLegUpper[0] + anchor2[0] - anchor[0]), 
                      int(rightLegUpper[1] + anchor2[1] - anchor[1])]

        frame = overlay_transparent(frame, bp, rightKnee[1], rightKnee[0])

        #Draw right upper leg over lower
        frame = overlay_transparent(frame, rightLegUpper, rightHip [1], rightHip[0])