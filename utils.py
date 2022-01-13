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

def getAngle(vec):
    angle = np.arctan2(vec[1], vec[0])
    return np.rad2deg(angle)

def drawDebugPoints(image):
    image = cv2.circle(image, (0,0), 3, (0,0,255,255), -1)
    image = cv2.circle(image, (image.shape[0],image.shape[1]), 3, (0,0,255,255), 3)
    return image

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

