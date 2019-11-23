import sys
import random
import _pickle as cPickle
import os, glob
import cv2
import shutil
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter


def kplusporche(queryImage,index='keypoints.txt', names='./index/names', k=5):
    image = cv2.imread(queryImage, cv2.IMREAD_COLOR)

    ##histData = normalizeHistogram(image)

    #print(histData)

    with open(index, 'rb') as f:
        hist = cPickle.load(f)

    print(hist)

    exit(0)

    listDistance = []
    for histo in range(len(hist)):
        listDistance.append(distance.euclidean(histData, hist[histo]))
    kpp = []
    for i in range(k):
        p = float('inf')
        for j in range(len(hist)):
            if listDistance[j] != 0  and  listDistance[j] < p and j not in kpp:
                p = listDistance[j]
                indice = j
        kpp.append(indice)

    with open(names, 'rb') as fName:
        listName = cPickle.load(fName)
    print(listDistance)
    for i in range(len(kpp)):
        print(listDistance[kpp[i]])
        print(listName[kpp[i]])
    return (kpp)


kplusporche('android.png')
