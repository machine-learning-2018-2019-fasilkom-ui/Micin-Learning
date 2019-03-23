#!/usr/bin/env python
# coding: utf-8

# In[134]:


import os 
import glob
import tensorflow as tf 
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import re
import imutils


# # Read Data

# In[99]:


img_dir = "Documents\kuliahsss\samp" # Enter Directory of all images 
img_Label = pd.read_csv("trainLabels.csv")
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


# In[100]:


Listimg = np.array(data)


# In[101]:


def showImage(row,col,Listimg):
    fig=plt.figure(figsize=(10, 10))
    columns = col
    rows = row
    for i in range(columns*rows):
        img = Listimg[i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()


# In[102]:


showImage(5,4,Listimg)


# In[103]:


plt.imshow(Listimg[2])


# In[110]:


img_Label.tail()


# In[ ]:





# # Data Preprocessing 

# In[249]:


def grayScale(Listimg):
    grays = []
    for i in range(Listimg.shape[0]):
        grays.append(cv.cvtColor(Listimg[i], cv.COLOR_BGR2GRAY))   
    return np.array(grays)


# In[250]:


def find_contours(grayScaleImg):
    grayScaleImg = cv.copyMakeBorder(grayScaleImg, 4, 4, 4, 4, cv.BORDER_REPLICATE)
    #thresholding
    BW = cv.threshold(grayScaleImg, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    #find contours
    contours, hierarchy  = cv.findContours(BW.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    return contours


# In[251]:


def splitImage(contours):
    splitImg = []
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
    # if ratio to high spplit into 2 image
    if w/h > 1.25 :
        half = int(w//2)
        splitImg.append([x,y,half,h])
        splitImg.append([x+half,y,half,h])
    else :
        splitImg.append([x,y,w,h])
    return splitImg


# In[257]:


def savePath(path,file):
    return os.path.join(path,file)

# def makeFolder(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

def LetterImage(grayImg,splitImg,img_Label,counts):
    splitImg = sorted(splitImg, key=lambda x: x[0])
    for boundary,label in zip(splitImg, img_Label) :
        print(label)
        x,y,w,h = boundary
        letter = grayImg[y-2:y+h+2, x-2:x+w+2]
        plt.imshow(letter,cmap='gray')
        
        save_path = savePath("Documents\\kuliahsss\\newTrain",label)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        count = counts.get(label, 1)
        path = os.path.join(save_path, "{}.jpg".format(str(count).zfill(6)))
        cv.imwrite(path, letter)  
        
        counts[label] = count + 1
        


# In[259]:


def createNewSplit(Listimg,img_Label):
    counts = {}
    #convert to gray
    gray = grayScale(Listimg)
    # for all image 
    for i in range(gray.shape[0]):
        contours = find_contours(gray[i])
        splitImg = splitImage(contours)
        LetterImage(gray[i],splitImg,img_Label[i],counts)


# In[261]:


createNewSplit(Listimg,img_Label["Labels"])


# In[ ]:




