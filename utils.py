import cv2 
import numpy as np

PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]



def drawKeypoints(keypoints, image, skel=False):
    for kps in keypoints:
        for k in kps:
            point = (int(k[1]), int(k[0]))
            image = cv2.circle(image, point, 2, (0, 200, 0), 2)
    if skel:
        drawSkel(keypoints, image)

def drawSkel(keypoints, image):
    for kp in keypoints:
        polygons = []

        for (p1Index, p2Index) in CONNECTED_PART_INDICES:
            polygons.append(
                np.array([ [int(x) for x in kp[p1Index][::-1]], [int(x) for x in kp[p2Index][::-1]] ])
            )

        image = cv2.polylines(image, polygons, isClosed=False, color=(255, 255, 255))

def drawSimilarityMetric(image, similarity):
    font = cv2.FONT_HERSHEY_COMPLEX
    image = cv2.putText(image, str(similarity), (50, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)




