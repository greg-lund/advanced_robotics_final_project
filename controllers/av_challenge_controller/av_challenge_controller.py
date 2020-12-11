#!/usr/bin/python3

"""av_challenge_controller controller."""
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Motor, Lidar
from vehicle import Driver, Car
import numpy as np
import cv2
import matplotlib.pyplot as plt


def laneLineCmdVel(camera, white_sensitivity=60):
    '''
    Take in raw image from front_camera, track
    lane lines and return steering angle such that
    the car centers itself on the lines
    '''

    # Get image from camera in the correct orientation
    img = cv2.flip(cv2.rotate(np.array(camera.getImageArray(), dtype='uint8'), cv2.ROTATE_90_CLOCKWISE), 1)
    # plt.imshow(img)
    # plt.show()
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
        return 0
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


def getFrame(camera):
    img = np.array(camera.getImageArray(), dtype='uint8')
    img = cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)
    return img


# create the Robot instance.
car = Car()
front_camera = car.getCamera("front_camera")
rear_camera = car.getCamera("rear_camera")
lidar = car.getLidar("Sick LMS 291")

# get the time step of the current world.
timestep = int(car.getBasicTimeStep())

front_camera.enable(50)
rear_camera.enable(100)
lidar.enable(100)
lidar.enablePointCloud()
car.setBrakeIntensity(0.5)
car.setCruisingSpeed(35)
alpha = 0.1
prev_cmd_angle = 0
low_pass_cmd = 0
while car.step() != -1:
    cmd_angle = laneLineCmdVel(front_camera, 40)
    # angle needs to be between -1 and 1
    # cmd_angle = max(min(cmd_angle, 2), -2)
    print(cmd_angle, low_pass_cmd)
    # if cmd_angle - prev_cmd_angle > 0.5 and abs(prev_cmd_angle) > abs(cmd_angle):
    #     prev_cmd_angle += 0.01
    #     cmd_angle = prev_cmd_angle
    low_pass_cmd = (alpha * cmd_angle) + ((1 - alpha) * (low_pass_cmd))
    if abs(low_pass_cmd) > 0.1:
        car.setCruisingSpeed(20)
    else:
        car.setCruisingSpeed(35)

    car.setSteeringAngle(low_pass_cmd)
    print(low_pass_cmd)
    print('brake intensity', car.getBrakeIntensity())
    print('speed', car.getCurrentSpeed())
    prev_cmd_angle = low_pass_cmd
    wheel_speed = car.getWheelSpeed(0)

