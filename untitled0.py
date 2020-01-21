# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:51:41 2020

@author: Education
"""

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
_ = cap.set(3, 1920)
_ = cap.set(4, 1080)
_ = cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS,0)

fps = 30

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter('record3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             fps, size)
success, frame = cap.read()

# cv2.imwrite("one_refer.jpg",frame)

numFrame = 60 * fps -1

while success and numFrame > 0:
        videoWriter.write(frame)
        cv2.imshow('a',frame)
        cv2.waitKey(30)
        success, frame = cap.read()
        success1, frame1 = cap1.read()
        success2, frame2 = cap2.read()
        numFrame -= 1
        print(numFrame)
cap.release()