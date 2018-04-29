

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
        
        #Distances...
        
        jaw_1 = (shape.part(0).x, shape.part(0).y)
        jaw_2 = (shape.part(16).x, shape.part(16).y)
        
        nose_tip = (shape.part(28).x, shape.part(28).y)
        
        eyl_1 = (shape.part(37).x, shape.part(37).y)
        eyl_2 = (shape.part(40).x, shape.part(40).y)
        
        eyl_x = int((eyl_1[0] + eyl_2[0])/2)
        eyl_y = int((eyl_1[1] + eyl_2[1])/2)
        eyl = (eyl_x, eyl_y)
        
        eyr_1 = (shape.part(43).x, shape.part(43).y)
        eyr_2 = (shape.part(46).x, shape.part(46).y)
        
        eyr_x = int((eyr_1[0] + eyr_2[0])/2)
        eyr_y = int((eyr_1[1] + eyr_2[1])/2)
        eyr = (eyr_x, eyr_y)
        
        a = int(sqrt((eyr[0]-jaw_2[0])**2 + (eyr[1]-jaw_2[1])**2))
        b = int(sqrt((nose_tip[0]-eyr[0])**2 + (nose_tip[1]-eyr[1])**2))
        c = int(sqrt((nose_tip[0]-eyl[0])**2 + (nose_tip[1]-eyl[1])**2))
        d = int(sqrt((eyl[0]-jaw_1[0])**2 + (eyl[1]-jaw_1[1])**2))
        print a, b, c, d
        
        if (((d-a)>0) and ((c-b)>10)):
            print "Left"
        
        if (((a-d)>20) and ((b-c)>10)):
            print "Right"
            
        n1 = (shape.part(27).x, shape.part(27).y)
        n2 = (shape.part(28).x, shape.part(28).y)
        n3 = (shape.part(29).x, shape.part(29).y)
        n4 = (shape.part(30).x, shape.part(30).y)
        
        
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
 
