# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
#code to get the key frame from the video and save it as a png file.

import cv2
import os
import numpy as np
#videopath : path of the video file
#frames_path: path of the directory to which the frames are saved
#count: to assign the video order to the frane.
def frameExtractorWithInvert(videopath,frames_path,count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no= int(video_length/2) 
    #print("Extracting frame..\n")
    cap.set(1,frame_no)
    ret,frame=cap.read()
    inverted = cv2.flip(frame,1)
    cv2.imwrite(frames_path + "/%#05d.png" % (count+1), inverted)
    frame_no = int(video_length*(1/4))
    return frame

def frameExtractor(videopath,frames_path,count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no= int(video_length/2)
    #print("Extracting frame..\n")
    cap.set(1,frame_no)
    ret,frame=cap.read()
    cv2.imwrite(frames_path + "/%#05d.png" % (count+1), frame)
    return frame
