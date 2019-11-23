import sys
import random
import cv2
import _pickle as cPickle
import os, glob
import shutil
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

img = cv2.imread('butterfly.png', 0)
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img, None)

img_with_keypoints = cv2.drawKeypoints(img, kp1, img)

cv2.imshow('test', img_with_keypoints)

cv2.waitKey()
