#!/usr/bin/env python

import numpy as np
import cv2
import ntpath
import glob
import sys
import os
from PIL import Image
sys.path.append('./python2')

# local modules
from video import create_capture
from common import clock, draw_str


help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <nested_fn>] [<video_source>]
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

if __name__ == '__main__':
    import sys, getopt
    print help_message
    
    path = '/home/ava/Dropbox/MultiMedia/Emotion-Detection-in-Videos/'
    video_path = '/home/ava/Dropbox/MultiMedia/Emotion-Detection-in-Videos/'
    files = glob.glob(video_path+ "/*.mp4")

    for fn in files:   
        #print fn
        args, video_src = getopt.getopt(fn, '', ['cascade=', 'nested-cascade='])
        args = dict(args)
        cascade_fn = args.get('--cascade', "./opencv/data/haarcascades/haarcascade_frontalface_alt2.xml")
        nested_fn  = args.get('--nested-cascade', "./opencv/data/haarcascades/haarcascade_eye.xml")

        cascade = cv2.CascadeClassifier(cascade_fn)
        nested = cv2.CascadeClassifier(nested_fn)
    
        cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
    
        name = ntpath.basename(fn)  

        #print name
        subfolder = os.path.join(path,'out_img/', name.split('.')[0])
        if not os.path.exists(subfolder):
            print subfolder
            print '--------path not exist---------'
            os.mkdir(subfolder)
            i=-1
            while True:
                i=i+1
            #rint i
                ret, img = cam.read()
            #print ret
                if not ret : break
                if i%10 ==0:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)       
                    t = clock()

                    rects = detect(gray, cascade)
            
                    vis = img.copy()
     
                    if not nested.empty():
                        
                        for x1, y1, x2, y2 in rects:
                            roi = gray[y1:y2, x1:x2]
                            vis_roi = vis[y1:y2, x1:x2]

                            res = cv2.resize(vis_roi,(48,48), interpolation = cv2.INTER_CUBIC)       

                            im = Image.fromarray(res).convert('L')
                            
                            image_name = os.path.join(subfolder ,'{:05d}.jpg'.format(i/10))
                            im.save(image_name)

        cv2.destroyAllWindows()



            
