 
import sys
import os
import dlib
import glob
#from skimage import io
import cv2
import time
from collections import OrderedDict
import numpy as np

from math import *
from scipy.spatial import distance as dist
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 35

# initialize the frame counters and the total number of blinks
COUNTER = 0

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
 
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
 
    # return the eye aspect ratio
    return ear


predictor_path = "shape_predictor_68_face_landmarks.dat"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

vs = cv2.VideoCapture(0)
time.sleep(1.0)

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                            ])
ret, frame = vs.read()
size = frame.shape
    
# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

while 1:
    ret, frame = vs.read()
    size = frame.shape
    
    frame_resized = frame #cv2.resize(frame, (height, width), cv2.INTER_AREA) #resize(frame, width=240)
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    t = time.time()
    rects = detector(frame_gray, 0)
    print('****************************')
    print('rects')
    print(t-time.time())
    print('len of rects', len(rects))

    for rect in rects:
        t = time.time()
        shape = predictor(frame_gray, rect)
        print('****************************')
        print('shape')
        print(t-time.time())
        
        nose_tip = (shape.part(30).x, shape.part(30).y)
        chin = (shape.part(8).x, shape.part(8).y)
        eyel = (shape.part(36).x, shape.part(36).y)
        eyer = (shape.part(45).x, shape.part(45).y)
        mol = (shape.part(48).x, shape.part(48).y)
        mor = (shape.part(54).x, shape.part(54).y)

        image_points = np.array([
                            nose_tip,     # Nose tip
                            chin,     # Chin
                            eyel,     # Left eye left corner
                            eyer,     # Right eye right corne
                            mol,     # Left Mouth corner
                            mor      # Right mouth corner
                        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, )
        
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
 
        cv2.line(frame, p1, p2, (255,0,0), 2)
        #Angle computation
        print degrees(atan2((p1[1]-p2[1]), (p1[0]-p2[0])))
        
        #fall asleep
        shape = shape_to_np(shape)
        
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = min([leftEAR,rightEAR])
        
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if 100 >= COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Warning! Seems he is trying to sleep", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if COUNTER > 100:
                cv2.putText(frame, "ALARM!!! ALARM!!! HE\'S SLEEPING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # reset the eye frame counter
            COUNTER = 0

        for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
        
    cv2.imshow("win", frame)
        
    key = cv2.waitKey(1)
    
cv2.destroyAllWindows()
vs.release()
