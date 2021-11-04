import argparse

import cv2
from Smoothing import Smoothing #pylint: disable=import-error
import tensorflow.compat.v1 as tf #pylint: disable=import-error
tf.disable_v2_behavior()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from Estimator import *
from utils import drawKeypoints


_WIN_NAME = "CharacterPose"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-draw_skel', action='store_true')
    args = parser.parse_args()

    cv2.namedWindow(_WIN_NAME)

    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(cap.get(3))
    height = int(cap.get(4))
    dim = (int(width), int(height))

    if cap is None or not cap.isOpened():
        print("Opening video capture failed. Exiting.")
        exit()


    with tf.Session() as sess:
        estimator = Estimator(sess) #load posenet model
        smoothing = Smoothing(1)

        #cv2.namedWindow(_WIN_NAME, cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty(_WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        #load image from file, get its normalized keypoints and set them to variable

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) #flip image so it looks like mirror

            keypoints = estimator.get_keypoints(frame)
            overlay_image = frame.copy()

            #if detection exists
            if 0 not in keypoints:

                if args.draw_skel:
                    drawKeypoints(smoothing.add_sample(keypoints), overlay_image, skel=True)

            #drawStrikePose(stitched_image)
            cv2.imshow(_WIN_NAME, overlay_image)

            char = cv2.waitKey(1)
            if char == ord('q'):
                break
            elif char == ord('n'):
                smoothing.sample_size += 1


        cap.release()
        cv2.destroyAllWindows()




