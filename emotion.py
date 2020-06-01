import cv2
import glob
import random
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

emotions = ["neutral", "anger", "disgust", "contempt", "fear", "happy", "sadness", "surprise"] #Emotion list
fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
data = {}
fishface.read("emotion_model.yml")

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pred, conf = fishface.predict(gray)
print(emotions[pred])



