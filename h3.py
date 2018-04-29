 


import sys
import os
import dlib
import glob
from skimage import io
import cv2
import time
from collections import OrderedDict
import numpy as np
import math
from math import *
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)


    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

(EyelStart, EyelEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(EyerStart, EyerEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

(NoseStart, NoseEnd) = FACIAL_LANDMARKS_IDXS["nose"]

(JawStart, JawEnd) = FACIAL_LANDMARKS_IDXS["jaw"]

(MouthStart, MouthEnd) = FACIAL_LANDMARKS_IDXS["mouth"]

predictor_path = "shape_predictor_68_face_landmarks.dat"

print FACIAL_LANDMARKS_IDXS["left_eye"]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


vs = cv2.VideoCapture(0)
time.sleep(1.0)

while 1:
    ret, frame = vs.read()
    
    
    frame_resized = frame #cv2.resize(frame, (height, width), cv2.INTER_AREA) #resize(frame, width=240)
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    rects = detector(frame_gray, 0)

    for rect in rects:
        shape = predictor(frame_gray, rect)
        
        #Key points...
        
        lip_l = (shape.part(48).x, shape.part(48).y)
        lip_r = (shape.part(54).x, shape.part(54).y)
        
        nose_tip = (shape.part(30).x, shape.part(30).y)
        
        eyl = (shape.part(36).x, shape.part(36).y)
        eyr = (shape.part(45).x, shape.part(45).y)
        chin = (shape.part(8).x, shape.part(8).y)
        
        
        
        shape = shape_to_np(shape)
        
        leftEye = shape[EyelStart:EyelEnd]
        rightEye = shape[EyerStart:EyerEnd]

        mouth = shape[MouthStart:MouthEnd]
        nose = shape[NoseStart:NoseEnd]
        jaw = shape[JawStart:JawEnd]
        

        
        for (x, y) in np.concatenate((leftEye, rightEye, mouth, nose, jaw), axis=0):
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
        
        #cv2.circle(frame, (eyl[0], eyl[1]), 2, (255, 255, 255), -1)
        #cv2.circle(frame, (eyr[0], eyr[1]), 2, (255, 255, 255), -1)
    
    cv2.imshow("win", frame)
        
    key = cv2.waitKey(1)
    
 #cv2.destroyAllWindows()
 
