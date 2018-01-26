from svm_ import *
import numpy as np
import os



def getdata(path):
    fileList = os.listdir(path)
    samples = []
    labels = []
    for file in fileList:
        label = file.split(' ')[0][-1]
        img = cv2.imread(path + file)
        sample = getfeatures(img)
        samples.append(sample)
        labels.append(label)
    return samples, labels

s = SVM()
path = "e://train/"
samples1, labels1 = getdata(path)
samples = np.array(samples1, dtype=np.float32).reshape((1354, 172))
print samples.shape

y_train = np.array(labels1, dtype=np.float32)

s.train_auto(samples, y_train, None, None)
s.save("123.xml")