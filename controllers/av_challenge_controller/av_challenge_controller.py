#!/usr/bin/python3

"""av_challenge_controller controller."""
from controller import Robot, Camera, Motor, Lidar
from vehicle import Driver, Car
import numpy as np
import cv2
import matplotlib.pyplot as plt

def detectStopSign(camera,init_brake=0.5):
    '''
    Given a camera, find stop sign via
    hsv masking, and depending on the size
    give a brake command
    '''

    img = cv2.flip(cv2.rotate(np.array(camera.getImageArray(), dtype='uint8'), cv2.ROTATE_90_CLOCKWISE), 1)
    cv2.imwrite('stop_sign_light.png',img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([110,17,168], dtype=np.uint8)
    upper = np.array([180,255,255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for c in contours:
        areas.append(cv2.contourArea(c))

    if len(areas) != 0:
        max_area = max(areas)
        if max_area > 30:
            return max(-init_brake/110 * (max_area - 130),0)
    return 1

def laneLineCmdVel(camera, white_sensitivity=60):
    '''
    Take in raw image from front_camera, track
    lane lines and return steering angle such that
    the car centers itself on the lines.
    Some inspiration taken from://towardsdatascience.com/deeppicar-part-4-lane-following-via-opencv-737dd9e47c96
    '''

    # Get image from camera in the correct orientation
    img = cv2.flip(cv2.rotate(np.array(camera.getImageArray(), dtype='uint8'), cv2.ROTATE_90_CLOCKWISE), 1)

    # Mask image to get just the (white) lane line
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 255-white_sensitivity], dtype=np.uint8)
    upper_white = np.array([255, white_sensitivity, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower_white, upper_white)

    # Get edges via canny edge detection
    edges = cv2.Canny(mask, 200, 400)
    (h, w) = edges.shape
    edges[0:int(2*h/3), :] = 0

    # Convert edges into lines via Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 1, np.array([]), minLineLength=8, maxLineGap=24)

    # If no lines return no steering command
    if lines is None:
        return None

    # Otherwise lets average, find the slope and displacement from center
    avg_center = 0
    avg_theta = 0
    n = 0
    for l in lines:
        l = l.flatten()
        if l[1] > l[3]:
            theta = np.pi - np.arctan2(l[1]-l[3], l[0]-l[2])
        else:
            theta = np.pi - np.arctan2(l[3]-l[1], l[2]-l[0])
        avg_theta += theta
        avg_center += (l[0] + l[2])/2
        n += 1

    avg_center /= n
    avg_theta /= n
    return float(avg_center - w/2)/float(w/2)


# Initialize our car and sensors
car = Car()
front_camera = car.getCamera("front_camera")
front_camera.enable(50)

car.setBrakeIntensity(0.75)
car.setCruisingSpeed(35)

# Do we want to stop at stop signs?
stop_sign = True

# Controller tuning and inits
alpha = 0.7
prev_cmd_angle = 0
low_pass_cmd = 0

while car.step() != -1:
    cmd_angle = laneLineCmdVel(front_camera, 40)
    if cmd_angle is None:
        cmd_angle = prev_cmd_angle

    diff = cmd_angle - low_pass_cmd
    p_total = 0.5 * cmd_angle

    diff = cmd_angle - prev_cmd_angle
    desired_speed = 40 / (abs(cmd_angle) + 1)
    desired_speed = 35 / (abs(cmd_angle) + 1)

    inc = max(min(diff, 0.02), -0.02)

    cmd_angle = prev_cmd_angle + inc
    low_pass_cmd = cmd_angle

    car.setCruisingSpeed(desired_speed)
    car.setSteeringAngle(p_total)

    prev_cmd_angle = low_pass_cmd
    
    # Stop sign detection
    if stop_sign:
        brake_cmd = detectStopSign(front_camera)
        car.setCruisingSpeed(desired_speed*brake_cmd)
