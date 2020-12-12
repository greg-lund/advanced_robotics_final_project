import cv2
import numpy as np

front = cv2.imread('stopsign_mask.png')
img = cv2.imread('mask.png')

mask = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
front_mask = cv2.cvtColor(front,cv2.COLOR_BGR2GRAY)

rows = mask.shape[0]
dp = 0.01
param1 = 12
param2 = 11
minRadius = 4
maxRadius = 8
circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,dp,20,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
front_circles = cv2.HoughCircles(front_mask,cv2.HOUGH_GRADIENT,dp,20,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)

circles = np.uint16(circles) if circles is not None else None
front_circles = np.uint16(front_circles) if front_circles is not None else None
print("circles:",circles)
print("front_circles:",front_circles)

if circles is not None:
    print("Sign is not none!")
    for c in circles[0,:]:
        cv2.circle(img,(c[0],c[1]),c[2],(0,200,0),2)
        cv2.circle(img,(c[0],c[1]),2,(0,0,255),3)
    cv2.imshow('image',img)
    cv2.waitKey(0)

if front_circles is not None:
    print("Front is not none!")
    for c in front_circles[0,:]:
        cv2.circle(front,(c[0],c[1]),c[2],(0,200,0),2)
        cv2.circle(front,(c[0],c[1]),2,(0,0,255),3)
    cv2.imshow('image',front)
    cv2.waitKey(0)

