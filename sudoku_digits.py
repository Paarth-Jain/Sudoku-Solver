import cv2
import copy
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import model_from_json
import pickle

def nothing(x):
    pass

# cap = cv2.VideoCapture('sudoku.avi')
img = cv2.imread('Sudoku.jpg')
width = int(img.shape[1]/4)
height = int(img.shape[0]/4)

dsize = (width,height)

frame1 = cv2.resize(img,dsize)
frame = copy.deepcopy(frame1)

# pickle_in = open('model_trained.p','rb')
# model = pickle.load(pickle_in)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3), 0,0)

thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21 , 20)

dilate = cv2.dilate(thresh ,None,iterations= 2)

edges = cv2.Canny(dilate, 30, 200, apertureSize=3)

''' mask building'''

contours,heirarchy = cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(frame, contours,-1 , (0,255,0), 3)

max = 0
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > cv2.contourArea(contours[max]):
        max = i

epsilon = 0.01*cv2.arcLength(contours[max], True)
approx = cv2.approxPolyDP(contours[max], epsilon, True)

cv2.drawContours(frame,[approx],0,(0,0,255),3)
cv2.imshow('framec',frame)

mask = np.zeros_like(thresh)
out = np.zeros_like(thresh)
cv2.drawContours(mask,[approx],0,(255,255,255),-1)
out[mask == 255] = thresh[mask == 255]
# cv2.imshow('masked',out)

approx = np.reshape(approx,(4,2))
approx = approx.astype('float32')
out = out.astype('float32')

out_rect = np.array([[0,0],
                        [0,out.shape[1]],
                        [out.shape[0],out.shape[1]],
                        [out.shape[0],0]],
                        dtype="float32")


persp = cv2.getPerspectiveTransform(approx,out_rect)
warp = cv2.warpPerspective(frame,persp,(out.shape[0],out.shape[1]))
warp = cv2.resize(warp,(out.shape[0],out.shape[0]))
# cv2.imshow('persp',warp)
warp1 = copy.deepcopy(warp)
side = out.shape[0]
threshold = 0.9

print("Identifying Digits")
for i in range(9):
    for j in range(9):

        digit = warp[i*side//9:(i+1)*side//9,j*side//9:(j+1)*side//9]  #cutting out the corresponding digit square

        index_left = int((0.1*side)//9)
        index_right = int((0.9*side)//9)

        digit = digit[index_left:index_right,index_left:index_right]    #removing black borders
        digit = digit.astype("uint8")
        digit = cv2.cvtColor(digit,cv2.COLOR_BGR2GRAY)
        digit = cv2.adaptiveThreshold(digit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,10)
        digit = cv2.resize(digit,(32,32))
        ratio = sum(sum(digit==255))/1024
        threshold = 950/1024                #might vary for a different Image
        digit = digit.reshape(1,32,32,1)
    
        if ratio < threshold:
            classIndex = int(loaded_model.predict_classes(digit))
            prediction = loaded_model.predict(digit)
            # print(prediction)
            cv2.putText(warp1,str(classIndex),(j*side//9,(i+1)*side//9),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            # print(classIndex,i,j)
            

cv2.imshow('predicted',warp1)       #digits recognized!

k = cv2.waitKey(0)
if k == 27: 
    cv2.destroyAllWindows()

