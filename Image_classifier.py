import cv2
from PIL import Image
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

Images = []

imdogs = []
for file in os.walk("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Panda/dogs"):
    for p in file[2]:
        imd = Image.open("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Panda/dogs/" + p)
        imd = imd.resize((32, 32))
        imd = np.ravel(imd)
        imdogs.append(imd)
imdogs = np.array(imdogs)
yd = np.full((1000, 1), 0)
imdogs1 = np.concatenate([imdogs, yd], axis = 1)

Images.extend(imdogs1)

imcats = []
for file in os.walk("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Panda/cats"):
    for p in file[2]:
        imd = Image.open("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Panda/cats/" + p)
        imd = imd.resize((32, 32))
        imd = np.ravel(imd)
        imcats.append(imd)
imcats = np.array(imcats)
yc = np.full((1000, 1), 1)
imcats1 = np.concatenate([imcats, yc], axis = 1)

Images.extend(imcats1)

impanda = []
for file in os.walk("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Panda/panda"):
    for p in file[2]:
        imd = Image.open("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Panda/panda/" + p)
        imd = imd.resize((32, 32))
        imd = np.ravel(imd)
        impanda.append(imd)
impanda = np.array(impanda)
yp = np.full((1000, 1), 2)
impanda1 = np.concatenate([impanda, yp], axis = 1)

Images.extend(impanda1)
Images = np.array(Images)

X, y = Images[:, :-1], Images[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# error_rate = []
# for i in range(1, 500):
#     knn = KNeighborsClassifier(n_neighbors = i)
#     knn.fit(X_train, y_train)
#     predictions = knn.predict(X_test)
#     error_rate.append(np.mean(predictions != y_test))

# print(error_rate)
# plt.figure(figsize = (10, 6))
# plt.plot(range(1, 500), error_rate, linestyle = '--', marker = 'o',
#          markerfacecolor = 'red', markersize = 10)
# plt.show()

knn = KNeighborsClassifier(n_neighbors = 260)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

pickle.dump(knn, open("Image_classifier.pkl", "wb"))