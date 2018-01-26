from svm_ import *
import cv2

def judge(result):
    data = getfeatures(result)
    y_val = clf.predict(data)
    return int(y_val)

def platejudge(resultrects):
    resultlist=[]
    for rr in resultrects:
        if 1 == judge(rr):
            resultlist.append(rr)
        else:
            w = rr.shape[1]
            h = rr.shape[0]
            rr_new = rr[int(h*0.1):int(h*0.8), int(w*0.05):int(w*0.9)]
            rr_new = cv2.resize(rr_new, (rr.shape[1], rr.shape[0]))
            if judge(rr_new) == 1:
                resultlist.append(rr)
    return resultlist




clf = SVM()
clf.load("123.xml")