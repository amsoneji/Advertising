#!/usr/bin/env python

import numpy as np
import cv2
import pdb
import time

# local modules
from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def get_face():
    import sys, getopt
    #print help_message

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=lena.jpg:noise=0.05')

    just_face = ''

    while True:
        #pdb.set_trace()
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()

        # Change to crop image to only show face
        if (just_face == ''):
            just_face = img.copy()

        param = (cv2.IMWRITE_PXM_BINARY, 1)
        if (len(rects) > 0):
            just_face = gray.copy()
            (x1,y1,x2,y2) = rects[0]
            just_face = just_face[y1:y2, x1:x2]
            cv2.imwrite('./test_face.pgm', just_face, param)
            return './test_face.pgm'

        vis_roi = vis
        draw_rects(vis, rects, (0, 255, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', just_face)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
