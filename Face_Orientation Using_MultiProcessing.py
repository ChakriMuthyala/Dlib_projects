#_________Multi Processing that developed with aim on to develop on raspberry pi
#Used Cores:    4
#developed threading like algorithm implementation to divide and conquer

import sys
import os


from functools import partial


import multiprocessing as mp
import cv2
import os
from collections import OrderedDict
import numpy as np
from math import atan2, degrees, sqrt
from scipy.spatial import distance as dist
import dlib
import time


#Predefinitions.........
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 35



#dlib stuff
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#CV Stuff

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
dist_coeffs = np.zeros((4,1))
FCOUNTER = 0
    
# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )


def get_feature_points(frame_gray):
    
    rects = detector(frame_gray, 0)
    
    if(len(rects) == 1):
        for rect in rects:
            shape = predictor(frame_gray, rect)
            image_points = np.array([
                            (shape.part(30).x, shape.part(30).y),     # Nose tip
                            (shape.part(8).x, shape.part(8).y),     # Chin
                            (shape.part(36).x, shape.part(36).y),     # Left eye left corner
                            (shape.part(45).x, shape.part(45).y),     # Right eye right corne
                            (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
                            (shape.part(54).x, shape.part(54).y)      # Right mouth corner
                        ], dtype="double")
    
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, )
        
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        return p1, p2
    else:
        return (2,1),(1,1)
        
def angle_computation(p1, p2):
    x = p1[0]-p2[0]
    y = p1[1]-p2[1]
    theta =  degrees(atan2(y,x))
    dis = sqrt(x**2+y**2)
    print(theta, dis)

    
    

    
    
    
    
#def drowsiness_detection():








if __name__ == '__main__':
    pool = mp.Pool( processes=4 )
            
    ret, frame = vs.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    r1 = pool.apply_async( get_feature_points, [ frame_gray ] )
    r2 = pool.apply_async( get_feature_points, [ frame_gray ] )
    r3 = pool.apply_async( get_feature_points, [ frame_gray ] )
    r4 = pool.apply_async( get_feature_points, [ frame_gray ] )
    
    r1p1, r1p2 = r1.get()
    r2p1, r2p2 = r2.get()
    r3p1, r3p2 = r3.get()
    r4p1, r4p2 = r4.get()
    
    
    while 1:
        ret, frame = vs.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if FCOUNTER == 1:
            r1 = pool.apply_async( get_feature_points, [ frame_gray ] )
            r2p1, r2p2 = r2.get()
            angle_computation(r2p1, r2p2)

        elif FCOUNTER == 5:
            r2 = pool.apply_async( get_feature_points, [ frame_gray ] )
            r3p1, r3p2 = r3.get()
            angle_computation(r3p1, r3p2)

        elif FCOUNTER == 9:
            r3 = pool.apply_async( get_feature_points, [ frame_gray ] )
            r4p1, r4p2 = r4.get()
            angle_computation(r4p1, r4p2)
            
        elif FCOUNTER == 13:
            r4 = pool.apply_async( get_feature_points, [ frame_gray ] )
            r1p1, r1p2 = r1.get()
            angle_computation(r1p1, r1p2)
            FCOUNTER = 0

        FCOUNTER += 1
            
