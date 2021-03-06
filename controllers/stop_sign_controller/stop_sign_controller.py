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
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([110,17,168], dtype=np.uint8)
    upper = np.array([180,255,255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=12,param2=11,minRadius=4,maxRadius=8)

    if circles is None:
        return False
    else:
        circles = np.uint16(circles)
        return True

    '''
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for c in contours:
        areas.append(cv2.contourArea(c))

    if len(areas) != 0:
        max_area = max(areas)
        if max_area > 30:
            return max(-init_brake/110 * (max_area - 130),0)
    return 1
    '''

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
    lower_white = np.array([0,0,215], dtype=np.uint8)
    upper_white = np.array([255,40,255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower_white, upper_white)

    '''
    out_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite('front_camera.png',out_img)
    cv2.imwrite('front_camera_bgr.png',img)
    cv2.imwrite('front_camera_hsw.png',img_hsv)
    cv2.imwrite('front_camera_mask.png',mask)
    '''

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


if __name__ == '__main__':
    # Initialize our car and sensors
    car = Car()
    front_camera = car.getCamera("front_camera")
    front_camera.enable(50)

    car.setBrakeIntensity(0.75)
    car.setCruisingSpeed(35)

    # Do we want to stop at stop signs?
    detect_stop_sign = True
    stop = False
    brakes = []
    brake_cmd = 1
    brake_k = 0.95

    # Controller tuning and inits
    max_speed = 39
    prev_cmd_angle = 0

    low_pass_cmd = 0

    while car.step() != -1:
        cmd_angle = laneLineCmdVel(front_camera)
        if cmd_angle is None:
            cmd_angle = prev_cmd_angle

        diff = cmd_angle - low_pass_cmd
        p_total = 0.55 * cmd_angle

        diff = cmd_angle - prev_cmd_angle
        desired_speed = max_speed / (abs(cmd_angle) + 1)
        # p_total = 0.5 * cmd_angle
        print(cmd_angle, low_pass_cmd)


        print('desired speed', desired_speed)
        print('diff', diff)

        inc = max(min(diff, 0.02), -0.02)

        cmd_angle = prev_cmd_angle + inc
        low_pass_cmd = cmd_angle

        car.setCruisingSpeed(desired_speed)
        car.setSteeringAngle(p_total)
        print('speed', car.getCurrentSpeed())
        prev_cmd_angle = low_pass_cmd

        # Stop sign detection
        if detect_stop_sign:
            brakes.append(detectStopSign(front_camera))
            brakes = brakes[-10:]
            if sum(brakes) >= 8:
                stop = True
                print("Detected stop sign! Braking...")
            if stop:
                brake_cmd *= brake_k
                car.setCruisingSpeed(desired_speed*brake_cmd)
