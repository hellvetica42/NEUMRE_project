import os
import time
import argparse

import cv2
import numpy as np

from Smoothing import Smoothing 
from Estimator import *
from utils import drawKeypoints
from Drawing import Drawing
from Character import Character

import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

_WIN_NAME = "CharacterPose"
_CHARACTER_PATH = "Characters/fighter"
_DRAW_WIN_NAME = "Drawing"
_FPS = 24
_SPF = 1.0/_FPS

#checks if file exists
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-draw_skel', action='store_true')
    parser.add_argument('-path', type=file_path)
    parser.add_argument('-draw_stick', action='store_true')
    args = parser.parse_args()

    cv2.namedWindow(_WIN_NAME)

    if args.draw_stick:
        cv2.namedWindow(_DRAW_WIN_NAME)

    cap = cv2.VideoCapture(args.path)
    if cap.isOpened() is False:
        print("Error opening file", args.path)
        exit()

    character = Character(_CHARACTER_PATH)
    draw = Drawing()

    with tf.Session() as sess:
        estimator = Estimator(sess) #load posenet model
        smoothing = Smoothing(3) #initialize smoothing function

        while True:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if ret is False:
                break

            #create empty white image for drawing
            if args.draw_stick:
                white_img = np.zeros(frame.shape, dtype='uint8')
                white_img[:] = 255

            #get keypoints for posenet
            keypoints = estimator.get_keypoints(frame)

            #if detection exists
            if 0 not in keypoints:

                #get smooth keypoints 
                smooth_kps = smoothing.getSample(keypoints[0])

                if args.draw_skel:
                    drawKeypoints([smooth_kps], frame, skel=True)

                if args.draw_stick:
                    draw.drawStickman(smooth_kps, white_img)

            if args.draw_stick:
                cv2.imshow(_DRAW_WIN_NAME, white_img)

            character.drawCharacter(smooth_kps, frame)

            cv2.imshow(_WIN_NAME, frame)

            end_time = time.perf_counter()

            #calculate time to wait to reach fps
            wait_time = int(max(_SPF - (end_time-start_time), 0) * 1000)
            wait_time = 1 if wait_time == 0 else wait_time

            char = cv2.waitKey(wait_time)

            if char == ord('q'):
                break
            elif char == ord('n'):
                smoothing.sample_size += 1

        cap.release()
        cv2.destroyAllWindows()




