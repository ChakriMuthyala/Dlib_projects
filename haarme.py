#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io
import cv2
import time
#import imutils

predictor_path = "shape_predictor_68_face_landmarks.dat"
#faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

vs = cv2.VideoCapture(0)
time.sleep(1.0)

while 1:
    ret, img = vs.read()
    #img = imutils.resize(img, width=450)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        #print shape
        
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0).x,
         #                                         shape.part(1).x))
        #k= shape.part(36)
       # print format(k[1])
        #k = list(k)
        #print k(0)
        #p1 = (int(shape.part(36).x), int(shape.part(36).y)
        #cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        #cv2.circle(img, ((int(shape.part(37).x), int(shape.part(37).y))), 2, (255, 255, 255), -1)
        #cv2.circle(img, ((int(shape.part(46).x), int(shape.part(46).y))), 2, (255, 255, 255), -1)
        #cv2.circle(img, ((int(shape.part(34).x), int(shape.part(34).y))), 2, (255, 255, 255), -1)
        #cv2.circle(img, ((int(shape.part(49).x), int(shape.part(49).y))), 2, (255, 255, 255), -1)
        #cv2.circle(img, ((int(shape.part(55).x), int(shape.part(55).y))), 2, (255, 255, 255), -1)
        #cv2.circle(img, ((int(shape.part(9).x), int(shape.part(9).y))), 2, (255, 255, 255), -1)
        #cv2.circle(img, ((int(shape.part(1).x), int(shape.part(1).y))), 2, (255, 255, 255), -1)
        #cv2.circle(img, ((int(shape.part(17).x), int(shape.part(17).y))), 2, (255, 255, 255), -1)
    #cv2.imshow("win", img)
        
    #key = cv2.waitKey(1)
 
    # if the `q` key was pressed, break from the loop
   
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)
    #time.sleep(0.2)
    win.add_overlay(dets)
   # dlib.hit_enter_to_continue()
 #cv2.destroyAllWindows()
 
