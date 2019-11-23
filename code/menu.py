# !/usr/bin/env python
# -*- coding: utf-8 -*-

import copyreg
import _pickle as cPickle
import os,sys, glob
import shutil
from cv2 import cv2
from matplotlib import pyplot as plt
import numpy as np


# Main definition - constants
from pip._vendor.distlib.compat import raw_input

menu_actions = {}


# =======================
#     MENUS FUNCTIONS
# =======================

# Main menu
def main_menu():
    os.system('clear')
    print("Bienvenu à l'application pour la reconnaissance d'objets avec le descripteur SIFT,\n")
    print("Veuillez choisir un menu que vous voulez pour demarrer:")
    print("1. creation de la base teste et entrainement")
    print("2. creation du modèle")
    print("3. creation de la description d'une image ")
    print("4. coorespondance des points d'intérêts")
    print("5. affichage de la description d'une image")
    print("6. affichage de la densité de SIFT")
    print("\n0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)

    return

def createTestAndTrainningData(datasetFolder, test="./test/", training="./training/"):
    files = os.listdir(datasetFolder)
    print("Number of dataset : " + str(len(files)))

    # Create target Directory if don't exist
    if not os.path.exists(test) and not os.path.exists(training):
        os.mkdir(test)
        print("Directory ", test, " Created ")
        os.mkdir(training)
        print("Directory ", training, " Created ")
    else:
        print("Directories  already exists")

    count = 0
    dataset = os.path.dirname(datasetFolder) + "/" + os.path.basename(datasetFolder)
    for file in files:
        if count % 2 == 0:
            shutil.copy(dataset + '/' + file, test)
        else:
            shutil.copy(dataset + '/' + file, training)

        count = count + 1

    print("Total files copied : " + str(count))
    print("9. Back")
    print("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)

#cPickle
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)

copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
# Create SIFT descritor file
def saveSIFTDescriptorAndKeypointFile(imageFile, test="./training", model="./model"):
    image = os.path.join(test, imageFile)
    img = cv2.imread(image, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, None)

    keyDescriptor = {'kp': kp1, 'des': des1}
    image_w_ext = os.path.basename(image)
    filename, file_extension = os.path.splitext(image_w_ext)

    # Dump the keypoints
    f = open(filename, "wb+")
    f.write(cPickle.dumps(keyDescriptor))
    f.close()

    # Store file in model directory
    if not os.path.exists(model):
        os.mkdir(model)
        print("Directory ", model, " Created ")
    else:
        print("Directories  already exists")

    shutil.move(filename, model)
    print("9. Back")
    print("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)

def createSIFTDescriptorFile(imageFile, test="./training", model="./model"):
    image = os.path.join(test, imageFile)
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        index.append(temp)
    image_w_ext = os.path.basename(image)
    filename, file_extension = os.path.splitext(image_w_ext)

    # Dump the keypoints
    f = open(filename, "wb+")
    f.write(cPickle.dumps(index))
    f.close()

    # Store file in model directory
    if not os.path.exists(model):
        os.mkdir(model)
        print("Directory ", model, " Created ")
    else:
        print("Directories  already exists")

    shutil.move(filename, model)

# create SIFT file for all training set

def createTrainingSIFTFiles(trainingFolder="./training"):
    for file in os.listdir(trainingFolder):
        saveSIFTDescriptorAndKeypointFile(file)
    print("9. Back")
    print("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)

# read descriptor file

def readDescriptorFileAndDrawKp(filename, model="./model"):
    keyPointDescriptor = []
    '''
        for file in os.listdir(trainingFolder):
            filenameDir, file_extension = os.path.splitext(os.path.join(trainingFolder, file))

            f = filenameDir.split("/")

            if f[-1] == filename:
                image = filenameDir+file_extension
                print(image)
                break
        exit(0)
        img = cv2.imread(image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''

    filenameWithPath = os.path.join(model, filename)
    # print(filenameWithPath)
    index = cPickle.loads(open(filenameWithPath, "rb").read())
    '''
    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                            _response=point[3], _octave=point[4], _class_id=point[5])
        kp.append(temp)

    '''
    keyPointDescriptor.append(index.get('kp'))
    keyPointDescriptor.append(index.get('des'))
    # print(keyPointDescriptor)
    print("9. Back")
    print("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)
    return keyPointDescriptor

def keypointsMatcher(queryImage, testFolder="./test", modelFolder="./model"):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(os.path.join(testFolder, queryImage), 0)  # queryImage

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)

    # read through the model folder
    listOfModel = os.listdir(modelFolder)
    print("Number of models " + str(len(listOfModel)))

    for model in listOfModel:

        keyAndDescriptor = readDescriptorFileAndDrawKp(model)
        # print(keyAndDescriptor)

        kp2 = keyAndDescriptor
        des2 = keyAndDescriptor[1]

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        print(len(np.array(matches).shape))
        print(np.array(matches).shape[1])
        # Apply ratio test
        good = []
        # good_without_list = []

        if len(np.array(matches).shape) == 2 and np.array(matches).shape[1] == 2:
            print(model)
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    good.append([m])


def denseSIFT(img, step_size=20, feature_scale=40, img_bound=20):
    # Create a dense feature detector
    detector = cv2.FeatureDetector_create("Dense")

    # Initialize it with all the required parameters
    detector.setInt("initXyStep", step_size)
    detector.setInt("initFeatureScale", feature_scale)
    detector.setInt("initImgBound", img_bound)
    # Run feature detector on the input image
    print("9. Back")
    print("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)
    return detector.detect(img)

# Execute menu
def exec_menu(choice):
    os.system('clear')
    ch = choice.lower()
    if ch == '':
        menu_actions['main_menu']()
    else:
        try:
            menu_actions[ch]()
        except KeyError:
            print("Invalid selection, please try again.\n")
            menu_actions['main_menu']()
    print("9. Back")
    print("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)
    return

def main1():
    createTestAndTrainningData("./dataset")
def main2():
     createTrainingSIFTFiles("./training")

def main3():
    img = input("saisir le nom de l'objet correctement:")
    img='"'+img+'"'
    print(img)
    exit(0)

    createSIFTDescriptorFile(img)

def main4():
    keypointsMatcher("obj2__185.png")

def main5():
    print(readDescriptorFileAndDrawKp("obj3__125"))
def main6():
    print(denseSIFT("butterfly.jpg"))

# Back to main menu
def back():
    menu_actions['main_menu']()


# Exit program
def exit():
    sys.exit()


# =======================
#    MENUS DEFINITIONS
# =======================

# Menu definition
menu_actions = {
    'main_menu': main_menu,
    '1': main1,
    '2': main2,
    '3': main3,
    '4': main4,
    '5': main5,
    '6': main6,
    '9': back,
    '0': exit,
}

# =======================
#      MAIN PROGRAM
# =======================

# Main Program
if __name__ == "__main__":
    # Launch main menu
    main_menu()
