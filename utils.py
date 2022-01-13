import cv2 
import numpy as np
from scipy import ndimage

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
    return np.rad2deg(angle), angle

def drawDebugPoints(image):
    image = cv2.circle(image, (0,0), 10, (0,0,255,255), -1)
    image = cv2.circle(image, (image.shape[1],image.shape[0]), 10, (0,0,255,255), 3)
    return image

def rotateVector(vector, theta):
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(rot, vector)

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def getRotatedImageWithAnchor(image, anchor, angle, anglerad):
        height, width = image.shape[:2]
        image_center = np.array((height/2, width/2))

        anchorVec = image_center - anchor

        image = rotate_image(image, angle)

        height, width = image.shape[:2]
        image_center = np.array((height/2, width/2))

        anchor = image_center - rotateVector(anchorVec, anglerad)
        return image, anchor

def getRotatedImageWithAnchor2(image, anchor1, anchor2, angle, anglerad):
        height, width = image.shape[:2]
        image_center = np.array((height/2, width/2))

        anchorVec1 = image_center - anchor1
        anchorVec2 = image_center - anchor2

        image = rotate_image(image, angle)

        height, width = image.shape[:2]
        image_center = np.array((height/2, width/2))

        anchor1 = image_center + rotateVector(anchorVec1, anglerad)
        anchor2 = image_center + rotateVector(anchorVec2, anglerad)

        return image, anchor1, anchor2

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    if x < 0 or y < 0:
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

