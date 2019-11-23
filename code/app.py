import _pickle as cPickle
import os
import random
import shutil
import sys
from collections import Counter

import cv2
import numpy as np
from matplotlib import pyplot as plt
# Main definition - constants
from pip._vendor.distlib.compat import raw_input
from sklearn.metrics import confusion_matrix

y_preD = []
y_truE = []
menu_actions = {}

"""
    How to install Opencv version containing SIFT and SURF image descriptor
    pip install opencv-python==3.4.2.16

    pip install opencv-contrib-python==3.4.2.16
"""


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
    print("3. coorespondance des points d'intérêts")
    print("4. affichage de la description d'une image")
    print("5. Mise en correspondance de deux images")
    print("6. Calcul matrice de confusion ")
    print("7. Dessiner les points d'interet")
    print("8. Dessiner les descripteurs")
    print("\n0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)

    return


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


# OK
def main2():
    createTrainingSIFTFiles("./training")


# OK
def main3():
    img = input("saisir le nom de l'objet correctement:")
    print(img)
    reponse = input("Voulez vous entrer le pourcnetage de distance ?  Y ou N : ")
    if str(reponse) == "Y" or str(reponse) == "y":
        pourcentage = input("saisir le pourcentage de la distance:")
        pourcentage = float(pourcentage)
        keypointsMatcher(img, distancePercentage=pourcentage)
    else:
        keypointsMatcher(img)


# OK
def main4():
    img = input("saisir le nom de l'objet(du dossier training) correctement:")

    print(readDescriptorFileAndDrawKp(img))


def main5():
    img1 = input("Entrez une image se trouvant dans la base de test: ")
    img2 = input("Entrez une deuxieme image se trouvant aussi dans la base test: ")
    drawCorrespondanceTowSameImage(img1, img2)
    # print(denseSIFT("butterfly.jpg"))


def main6():
    k = input("saisir le nombre d'images a tester par categorie: ")
    k = int(k)
    modelTest(k)


def main7():
    k = input("entrez une image se trouvant dans votre base de test: ")
    k = str(k)
    drawKeyPointsOnImage(k)


def main8():
    k = input("entrez une image se trouvant dans votre base de test: ")
    k = str(k)
    drawKeyDescriptorsOnImage(k)


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
    '7': main7,
    '8': main8,
    '9': back,
    '0': exit,
}


# Object detection

# 1 divide your data set into testold en trainning


def test():
    img = cv2.imread('butterfly.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    print(sift)


def drawKeyPointsOnImage(image):
    file = os.path.join('./test', image)
    img = cv2.imread(file)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, None)
    cv2.imshow('original', img)
    img_with_keypoints = cv2.drawKeypoints(img, kp1, img)
    cv2.imshow('keypoints', img_with_keypoints)
    cv2.waitKey()


def drawKeyDescriptorsOnImage(image):
    file = os.path.join('./test', image)
    img = cv2.imread(file)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, None)
    cv2.imshow('original', img)
    img_with_keypoints = cv2.drawKeypoints(img, kp1, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('descriptors', img_with_keypoints)
    cv2.waitKey()


def drawCorrespondanceTowSameImage(queryImage, searchImage, testFolder="./test"):
    img1 = cv2.imread(str(os.path.join(testFolder, queryImage)), 0)  # queryImage
    img2 = cv2.imread(str(os.path.join(testFolder, searchImage)), 0)  # trainImage

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio testold
    good_without_list = []

    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_without_list.append(m)

    img2 = cv2.imread(os.path.join(testFolder, queryImage), 0)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_without_list, None)

    plt.imshow(img3)
    plt.show()


def createTestAndTrainningData(datasetFolder, test="./test/", training="./training/", numberTest=1, numberTrain=2):
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

    dataset = os.path.dirname(datasetFolder) + "/" + os.path.basename(datasetFolder)
    fileCount = 0
    count = 0
    while fileCount < len(files):
        if count % 2 == 0:
            countTest = 0
            while countTest < numberTest and files[fileCount]:
                shutil.copy(dataset + '/' + files[fileCount], test)
                countTest = countTest + 1
                fileCount = fileCount + 1
        else:
            countTraining = 0
            while countTraining < numberTrain and files[fileCount]:
                shutil.copy(dataset + '/' + files[fileCount], training)
                countTraining = countTraining + 1
                fileCount = fileCount + 1

        count = count + 1


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


# create SIFT file for all trainingold set

def createTrainingSIFTFiles(trainingFolder="./training"):
    for file in os.listdir(trainingFolder):
        saveSIFTDescriptorAndKeypointFile(file)


# read descriptor from file with keypoint

def readDescriptorFileAndDrawKp(filename, model="./model"):
    keyPointDescriptor = []
    name = filename.split(".")
    filename = name[0]
    filenameWithPath = os.path.join(model, filename)
    index = cPickle.loads(open(filenameWithPath, "rb").read())
    keyPointDescriptor.append(index.get('kp'))
    keyPointDescriptor.append(index.get('des'))
    return keyPointDescriptor


'''
    Keypoints matcher, takes an image as queryImage and looks in the model folder descriptors tha match better 
    this image.
    queryImage should be taken from the ./testold folder
    distancePercentage = 0.75  is the distance that is selected for each image corresponding image in the model
'''


def keypointsMatcher(queryImage, testFolder="./test", modelFolder="./model", distancePercentage=0.65):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(os.path.join(testFolder, queryImage), 0)  # queryImage

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)

    # read through the model folder
    listOfModel = os.listdir(modelFolder)
    print("Number of models " + str(len(listOfModel)))
    print(len(des1))
    print(len(kp1))

    numberKpToSelect = len(kp1) // 3
    lilstOfSelectedModel = []

    for model in listOfModel:
        keyAndDescriptor = readDescriptorFileAndDrawKp(model)
        kp2 = keyAndDescriptor[0]
        print('kp2', len(kp2))
        print('borne', len(kp1) - numberKpToSelect, 'borne', len(kp1) + numberKpToSelect)

        if (len(kp2) >= len(kp1) - numberKpToSelect) and (len(kp2) <= len(kp1) + numberKpToSelect):
            des2 = keyAndDescriptor[1]
            print('first')
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des2, des1, k=2)

            # Apply ratio testold
            good = []

            # if len(np.array(matches).shape) == 2 and np.array(matches).shape[1] == 2:
            for m, n in matches:
                if (m.distance < distancePercentage * n.distance):
                    good.append([m])
            rapport = len(good) / len(kp1)
            if rapport >= 0.60:
                lilstOfSelectedModel.append({'model': model, 'numberDescriptor': len(good), 'score': round(rapport, 2),
                                             'ratio': round(m.distance / n.distance, 2)})
                print(model, " ", "selected  rapprot")

    lilstOfSelectedModel = sorted(lilstOfSelectedModel, key=lambda k: k['numberDescriptor'])
    ratio = 'Ratio: ' + str(
        round(lilstOfSelectedModel[0]['numberDescriptor'] / lilstOfSelectedModel[-1]['numberDescriptor'], 2))
    seen = []
    for ob in lilstOfSelectedModel:
        filename = ob['model'].split("__")[0]
        seen.append(filename)
    listSeen = Counter(seen)
    for element in listSeen:
        print('\n\n', element, ' ', listSeen[element], "  ", countCategoryElement(element),
              listSeen[element] / countCategoryElement(element), '\n\n')
    print(lilstOfSelectedModel)
    font = cv2.FONT_ITALIC
    # cv2.putText(img1, ratio, (5, 122), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow(queryImage, img1)
    for i in range(len(lilstOfSelectedModel)):
        nameImage = lilstOfSelectedModel[i]['model'] + '.png'
        fileImage = os.path.join('./training', nameImage)
        img = cv2.imread(fileImage)
        cv2.putText(img, 'Ratio:' + str(lilstOfSelectedModel[i]['ratio']), (5, 122), font, 0.5, (255, 0, 0), 2,
                    cv2.LINE_AA)
        # cv2.putText(img, 'score :'+str(lilstOfSelectedModel[i]['score']), (5, 10), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(lilstOfSelectedModel[i]['model'], img)
    cv2.waitKey()


def calculateCorrespond(desQuery, desModel, numberMatches):
    return numberMatches / (len(desQuery) + len(desModel))


def countCategoryElement(catname, modelFolder="./model"):
    # read through the model folder
    listOfModel = os.listdir(modelFolder)
    count = 0
    for filename in listOfModel:
        if catname + "__" in filename:
            count = count + 1
    return count


def denseSIFT(img, step_size=20, feature_scale=40, img_bound=20):
    # Create a dense feature detector
    detector = cv2.FeatureDetector_create("Dense")

    # Initialize it with all the required parameters
    detector.setInt("initXyStep", step_size)
    detector.setInt("initFeatureScale", feature_scale)
    detector.setInt("initImgBound", img_bound)
    # Run feature detector on the input image
    return detector.detect(img)


def keypointsMatcherSecond(queryImage, testFolder="./test", modelFolder="./model", distancePercentage=0.65):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(os.path.join(testFolder, queryImage), 0)  # queryImage

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)

    # read through the model folder
    listOfModel = os.listdir(modelFolder)
    numberKpToSelect = 2 * (len(kp1) // 3)
    lilstOfSelectedModel = []
    for model in listOfModel:

        keyAndDescriptor = readDescriptorFileAndDrawKp(model)
        des2 = keyAndDescriptor[1]

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des2, des1, k=2)

        # Apply ratio testold
        good = []

        if len(np.array(matches).shape) == 2 and np.array(matches).shape[1] == 2:
            for m, n in matches:
                if (m.distance / n.distance) > distancePercentage:
                    good.append([m])
            if len(good) < numberKpToSelect:
                lilstOfSelectedModel.append(model)

    # print(lilstOfSelectedModel)
    seen = []
    for ob in lilstOfSelectedModel:
        filename = ob.split("__")[0]
        seen.append(filename)
    listSeen = Counter(seen)
    trueList = []
    confusion = np.zeros(5)
    for element in listSeen:
        print(element, ' ', listSeen[element], "  ", countCategoryElement(element),
              listSeen[element] / countCategoryElement(element))
        if listSeen[element] / countCategoryElement(element) >= 0.9:
            trueList.append(element)
    y_truE.append(trueList)
    print(len(lilstOfSelectedModel))
    # print(y_true)
    # print(len(y_true))


def modelTest(k=5, testFolder="./test"):
    y_pred = []
    y_true = []

    i = 1
    while i <= 100:
        fileCount = 0
        listOfSelectedImage = []
        while fileCount < k:
            fileInCategory = random.randint(1, 1001)
            fileName = "obj" + str(i) + "__" + str(fileInCategory) + ".png"
            filePath = os.path.join(testFolder, fileName)
            if os.path.exists(filePath) and not fileName in listOfSelectedImage:
                print(filePath)
                listOfSelectedImage.append(fileName)

                myModel = fileName.split("__")
                y_preD.append(myModel[0])

                keypointsMatcherSecond(fileName)
                fileCount = fileCount + 1
        i = i + 1
    print("=======================================================================================")
    print(y_truE)
    print("***************************************************************************************")
    print(y_preD)
    label = y_preD
    print("lllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")
    print(label)
    for i in range(len(y_preD)):
        for j in range(len(y_truE[i])):
            truE = y_truE[i]
            y_pred.append(y_preD[i])
            y_true.append(truE[j])
    print("=======================================================================================")
    print(y_true)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(y_pred)
    results = confusion_matrix(y_true, y_pred)
    print(results)
    print("=======================================================================================")
    print("9. Back")
    print("0. Quit")
    choice = raw_input(" >>  ")
    exec_menu(choice)


def getSizesOfDataSet(folder):
    print(len(os.listdir(folder)))


def main():
    main_menu()
    # getSizesOfDataSet("./test")
    # testold()
    # createTestAndTrainningData("./dataset")
    # createTrainingSIFTFiles("./training")
    #createSIFTDescriptorFile("obj1__30.ppm")
    #keypointsMatcher("obj1__30.ppm", distancePercentage=0.65)
    # print(readDescriptorFileAndDrawKp("obj3__125"))
    # print(denseSIFT("butterfly.jpg"))
    # drawCorrespondanceTowSameImage("obj76__320.png", "obj76__250.png")
    # modelTest(k=1)


if __name__ == "__main__":
    main()
