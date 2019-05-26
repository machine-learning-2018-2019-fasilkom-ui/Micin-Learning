#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os 
import glob
import tensorflow as tf 
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import re


# # Read Data

# In[16]:


img_dir = "D:\Backup-Folder\ks\Micin-Learning-master\CaptchaData" # Enter Directory of all images 
img_Label = pd.read_csv("trainLabels.csv")
img_Label = list(img_Label['Labels'])
data_path = os.path.join(img_dir,'*.JPG')
files = glob.glob(data_path)
data = []

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

for f1 in sorted(files, key=numericalSort):
    img = cv.imread(f1)
    data.append(img)


# In[93]:


img_Label = img_label


# In[98]:


len(data), len(img_Label)


# In[19]:


Listimg = np.array(data)


# In[20]:


def showImage(row,col,Listimg):
    fig=plt.figure(figsize=(10, 10))
    columns = col
    rows = row
    for i in range(columns*rows):
        img = Listimg[i]
        fig.add_subplot(rows, columns, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
    plt.show()


# In[21]:


showImage(5,4,Listimg)


# In[37]:


gray = grayScale(Listimg)


# In[38]:


g = gray[104].copy()
b = BW(gray[104])
contours = find_contours(b)


# In[44]:


splitImg = []
for c in contours:
    x,y,w,h = cv.boundingRect(c) 
    if w/h > 1 :
        half = int(w//2)
        if half / h > 1: 
            halff = int(half//2)
            splitImg.append([x,y,halff,h])
            splitImg.append([x+halff,y,halff,h])
            splitImg.append([x+half,y,halff,h])
            splitImg.append([x+half+halff,y,halff,h])
            rect1 = np.array([[x+halff-1,y-1],
                         [x+halff-1,y+h-1],
                         [x+half-1,y+h-1],
                         [x+half-1,y-1]])
            rect2 = np.array([[x-1,y-1],
                         [x-1,y+h-1],
                         [x+halff-1,y+h-1],
                         [x+halff-1,y-1]])
            rect3 = np.array([[x+half+halff-1,y-1],
                         [x+half+halff-1,y+h-1],
                         [x+half+2*halff-1,y+h-1],
                         [x+half+2*halff-1,y-1]])
            rect4 = np.array([[x+half-1,y-1],
                         [x+half-1,y+h-1],
                         [x+half+halff-1,y+h-1],
                         [x+half+halff-1,y-1]])
            im = cv.drawContours(g,[rect1],0,1,0)
            im = cv.drawContours(g,[rect2],0,1,0)
            im = cv.drawContours(g,[rect3],0,1,0)
            im = cv.drawContours(g,[rect4],0,1,0)    
        else :     
            splitImg.append([x,y,half,h])
            splitImg.append([x+half,y,half,h])
            rect1 = np.array([[x+half-1,y-1],
                         [x+half-1,y+h-1],
                         [x+w-1,y+h-1],
                         [x+w-1,y-1]])
            rect2 = np.array([[x-1,y-1],
                         [x-1,y+h-1],
                         [x+half-1,y+h-1],
                         [x+half-1,y-1]])
            im = cv.drawContours(g,[rect1],0,1,0)
            im = cv.drawContours(g,[rect2],0,1,0)
    else :
        splitImg.append([x,y,w,h])
        rect1 = np.array([[x-1,y-1],
                     [x-1,y+h-1],
                     [x+w-1,y+h-1],
                     [x+w-1,y-1]])
        im = cv.drawContours(g,[rect1],0,1,0)
plt.imshow(g,cmap='gray')


# # Data Preprocessing 

# In[31]:


def grayScale(Listimg):
    grays = []
    for i in range(Listimg.shape[0]):
        grays.append(cv.cvtColor(Listimg[i], cv.COLOR_BGR2GRAY))   
    return np.array(grays)


# In[32]:


def BW(grayScaleImg):
    grayScaleImg = cv.copyMakeBorder(grayScaleImg, 4, 4, 4, 4, cv.BORDER_REPLICATE)
    #thresholding
    BW = cv.threshold(grayScaleImg, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    return BW


# In[33]:


def find_contours(BW):
    #find contours
    _, contours, _= cv.findContours(BW.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    return contours


# In[40]:


def splitImage(contours):
    splitImg = []
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
    # if ratio to high spplit into 2 image
        if w/h > 1 :   
            half = int(w//2)       
            if half / h > 1: 
                halff = int(half//2)
                splitImg.append([x,y,halff,h])
                splitImg.append([x+halff,y,halff,h])
                splitImg.append([x+half,y,halff,h])
                splitImg.append([x+half+halff,y,halff,h])
            else :
                splitImg.append([x,y,half,h])
                splitImg.append([x+half,y,half,h])
        else :
            splitImg.append([x,y,w,h])
    return splitImg


# In[41]:


def savePath(path,file):
    return os.path.join(path,file)

def LetterImage(grayImg,splitImg,img_Label,counts):
    splitImg = sorted(splitImg, key=lambda x: x[0])
    for boundary,label in zip(splitImg, img_Label) :
        print(label)
        x,y,w,h = boundary
        letter = grayImg[y-2:y+h+2, x-2:x+w+2]
        plt.imshow(letter,cmap='gray')
        
        save_path = savePath("Documents\\kuliahsss\\TrainCapt3",label)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        count = counts.get(label, 1)
        path = os.path.join(save_path, "{}.jpg".format(str(count).zfill(6)))
        cv.imwrite(path, letter)  
        
        counts[label] = count + 1
        


# In[42]:


def createNewSplit(Listimg,img_Label):
    counts = {}
    #convert to gray
    gray = grayScale(Listimg)
    # for all image 
    for i in range(gray.shape[0]):
        bw = BW(gray[i])
        contours = find_contours(bw)
        splitImg = splitImage(contours)
        LetterImage(gray[i],splitImg,img_Label[i],counts)


# In[45]:


gray = grayScale(Listimg)


# In[177]:


createNewSplit(Listimg,img_Label)


# In[99]:


img_dir = "D:\Backup-Folder\ks\Micin-Learning-master\letterSplit"
subdirs = [x[0] for x in os.walk(img_dir)]  
class_num = -1
y = []
trainData = []
for subdir in subdirs :
    data_path = os.path.join(subdir,"*.JPG")
    files = glob.glob(data_path)
    class_num += 1   
    for img in files :
        try :
            img_array = cv.imread(img,cv.IMREAD_GRAYSCALE)
            new_array = cv.resize(img_array,(50,50))
            trainData.append(new_array)
            y.append(class_num)
        except Exception as e:
            pass


# In[100]:


trainData = np.array(trainData)


# In[101]:


trainData.shape


# In[102]:


trainData  = trainData.reshape(trainData.shape[0],50,50,1)


# In[103]:


trainData.shape[1:]


# In[104]:


classNum = len(set(y))


# In[105]:


label = np.copy(y)


# In[106]:


label.shape


# # One hot encoding the Label

# In[107]:


label = np.int64(label)
nvalues = np.max(label) 
label = np.eye(nvalues)[label-1]


# In[108]:


label = np.float64(label)
label


# # Create Models

# In[129]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
from time import time


# In[132]:


def cnn_model(X,y,epoch, board):
    classNum = y.shape[1]

    model = Sequential()

    conv2d = Conv2D(64,(2,2),strides=(1,1),input_shape = X.shape[1:],padding='valid')
    
    model.add(conv2d)
    model.add(MaxPooling2D(pool_size=(2,2)))

    conv2d = Conv2D(64,(2,2),strides=(1,1),padding = 'valid',activation='relu')
    model.add(conv2d)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(conv2d)
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(100,activation='relu'))
    
    model.add(Dense(classNum))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'SGD',
                  metrics=['accuracy'])
    
    model.fit(X,y,epochs = epoch,batch_size = 16,validation_split=0.2, callbacks = [board])
    return model 


# In[137]:


classNum = label.shape[1]

model = Sequential()

conv2d = Conv2D(64,(2,2),strides=(1,1),input_shape = trainData.shape[1:],padding='valid')

model.add(conv2d)
model.add(MaxPooling2D(pool_size=(2,2)))

conv2d = Conv2D(64,(2,2),strides=(1,1),padding = 'valid',activation='relu')
model.add(conv2d)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(conv2d)
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))

model.add(Dense(classNum))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'SGD',
              metrics=['accuracy'])


# # Train Model

# In[112]:


s = list(zip(trainData, label))
np.random.shuffle(s)
xTrain, yTrain = zip(*s)
xTrain = np.array(xTrain)
yTrain = np.array(yTrain)


# In[113]:


xTrain.shape, yTrain.shape


# In[140]:


name = "wordcls-{}".format(int(time()))
tensorboard = TensorBoard(log_dir="logs{}".format(name))


# In[115]:


model = cnn_model(xTrain,yTrain,10,tensorboard)


# In[116]:


model2 = cnn_model(xTrain,yTrain,20,tensorboard)


# In[117]:


model3 = cnn_model(xTrain,yTrain,30,tensorboard)


# In[139]:


model4 = cnn_model(xTrain,yTrain,40,tensorboard)


# In[148]:


model4.save_weights("modelweight4.h5")


# # Open Test Data

# In[159]:


# Test directory with label
img_dir = "D:\\Backup-Folder\\ks\\Micin-Learning-master\\testCaptchawithLabel" 
data_path = os.path.join(img_dir,'*.JPG')
files = glob.glob(data_path)
data = []

for f1 in sorted(files, key=numericalSort):
    img = cv.imread(f1)
    data.append(img)


# In[161]:


# Test directory without label
img_dir = "D:\\Backup-Folder\\ks\\Micin-Learning-master\\testcaptchaNoLabel" 
data_path = os.path.join(img_dir,'*.JPG')
files = glob.glob(data_path)
datanoLabel = []

for f1 in sorted(files, key=numericalSort):
    img = cv.imread(f1)
    datanoLabel.append(img)


# # Preprocess Test Data

# In[163]:


classPath = "D:\\Backup-Folder\\ks\\Micin-Learning-master\\letterSplit"
classList = os.listdir(classPath)


# In[164]:


Listimg = np.array(data)
grayimg = grayScale(Listimg)


# # Load Model

# In[166]:


def model():
    classNum = yTrain.shape[1]

    model = Sequential()

    conv2d = Conv2D(64,(2,2),strides=(1,1),input_shape = (50,50,1),padding='valid')
    
    model.add(conv2d)
    model.add(MaxPooling2D(pool_size=(2,2)))

    conv2d = Conv2D(64,(2,2),strides=(1,1),padding = 'valid',activation='relu')
    model.add(conv2d)
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(conv2d)
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(100,activation='relu'))
    
    model.add(Dense(classNum))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'SGD',
                  metrics=['accuracy'])
    return model


# In[167]:


modelTest = model()


# In[169]:


modelTest.load_weights("modelweight4.h5")


# # Predict Test Data

# In[182]:


gray = grayimg
ls = []

for i in range(grayimg.shape[0]):
    bw = BW(gray[i])
    contours = find_contours(bw)
    splitImg = splitImage(contours)
    splitImg = sorted(splitImg, key=lambda x: x[0])
    predict = ""
    word = []
    count = 0 
    for boundary in splitImg:
        try :
            x,y,w,h = boundary
            y1 = y-2
            y2 = y+h+2
            x1 = x-2
            x2 = x+w+2
            if y1 < 0 :
                y1 = 0 
            if x1 < 0 :
                x1 = 0
            if y2 > gray.shape[1] :
                y2 = gray.shape[1]
            if x2 > gray.shape[2] :
                x2 = gray.shape[2]   
            letter = gray[i][y1:y2, x1:x2]
            letter = cv.resize(letter,(50,50))
            letter = letter.reshape(50,50,1)
            word.append(letter)  
        except:
            continue
    word = np.array(word)
    pred = modelTest.predict_classes(word)
    for c in pred : 
        predict += classList[c]
    ls.append(predict)


# In[183]:


labelTest = pd.read_csv("trainLabels.csv")
labelTest = list(labelTest.loc[150:]['Labels'])


# In[184]:


print(labelTest)


# In[185]:


trueprd = 0
for i in range(len(labelTest)) :
    if labelTest[i] == ls[i]:
        trueprd +=1


# In[186]:


wordList =  "".join(labelTest)


# In[187]:


wordListPred = "".join(ls)


# In[188]:


trueprdword = 0
for i in range(len(wordList)) :
    if wordList[i] == wordListPred[i] :
        trueprdword +=1


# In[189]:


print("Word Accuracy : " + str(trueprdword / len(labelTest)))


# In[190]:


print("Accuracy : "+ str(trueprd / len(labelTest)))


# In[191]:


showImage(5,5,Listimg[:25])
print(ls)


# In[195]:


def generate_captcha(gray):
    i = np.random.randint(50)
    plt.imshow(gray[i], cmap = 'gray')


# In[197]:


def solve_captcha(gray, i, modelTest):
    bw = BW(gray[i])
    contours = find_contours(bw)
    splitImg = splitImage(contours)
    splitImg = sorted(splitImg, key=lambda x: x[0])
    predict = ""
    word = []
    count = 0 
    for boundary in splitImg:
        try :
            x,y,w,h = boundary
            y1 = y-2
            y2 = y+h+2
            x1 = x-2
            x2 = x+w+2
            if y1 < 0 :
                y1 = 0 
            if x1 < 0 :
                x1 = 0
            if y2 > gray.shape[1] :
                y2 = gray.shape[1]
            if x2 > gray.shape[2] :
                x2 = gray.shape[2]   
            letter = gray[i][y1:y2, x1:x2]
            letter = cv.resize(letter,(50,50))
            letter = letter.reshape(50,50,1)
            word.append(letter)  
        except:
            continue
    word = np.array(word)
    pred = modelTest.predict_classes(word)
    for c in pred : 
        predict += classList[c]
    ls.append(predict)
    return ls

