import timeit
from PIL import Image
import sys
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
start=timeit.default_timer()

#import train
stop=timeit.default_timer()

def facecrop(image):
	face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	face_cascade2=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	print "Image size",image.shape
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,1.3,5,minSize=(70,70))
	if len(faces)==0:
            faces=face_cascade2.detectMultiScale(gray,1.3,5)
	print "Faces position",faces
	face=[]
	position=[]
	for (x,y,w,h) in faces:
            face.append(gray[y:y+h, x:x+w])
            position.append((x,y,w,h))
        return (face,position)

print "import time ",stop-start

def extractfeature(path):
	image=cv2.imread(path)
        sift= cv2.xfeatures2d.SIFT_create()
	image,position=facecrop(image)
        if image==None:
            return None
        gray=image
        for i,pos in zip(range(len(image)),position):
            print type(pos)
            xx=image[i]
            kp,dsc=sift.detectAndCompute(image[i],None)
            xx=cv2.drawKeypoints(xx,kp,None)
            cv2.imshow('notcropped',xx)
            cv2.waitKey(0)
            image[i]=image[i][0.1*pos[3] :pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3]]
            kp,dsc=sift.detectAndCompute(image[i],None)
            img=cv2.drawKeypoints(image[i],kp,None)
            cv2.imshow('face',img)
            cv2.waitKey(0)
        features=[]
        for i in range(len(image)):
            features.append((bowdict.compute(image[i],sift.detect(image[i]))))
	return (features,position)
#filee=open('bow','r')
#filee.read(bowdict)
#filee.close()

imagefile=sys.argv[1]

sift2 = cv2.xfeatures2d.SIFT_create()
bowdict = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
start=timeit.default_timer()
dictionary=joblib.load('trained/bow/bow.pkl')
stop=timeit.default_timer()
print "dict load time",stop-start

start=timeit.default_timer()
bowdict.setVocabulary(dictionary)
stop=timeit.default_timer()
print "set vocab time",stop-start
start=timeit.default_timer()
clf=joblib.load('trained/svm/svm.pkl')
stop=timeit.default_timer()
print "svm load time",stop-start
#print bowdict
start=timeit.default_timer()
test,position=extractfeature(imagefile)
stop=timeit.default_timer()
print "extract features time",stop-start
start=timeit.default_timer()
answer=[]
for i in range(len(test)):
    print clf.predict(test[i])
    answer.append(int(clf.predict(test[i])[0][3]))
stop=timeit.default_timer()
print "predict time",stop-start	


im=cv2.imread(imagefile)

angry=cv2.imread('anger.png')
happy=cv2.imread('happy.png')
sad=cv2.imread('sadness.png')
neutral=cv2.imread('neutral.png')

for ans,pos in zip(answer,position):
	if ans==1:
	    emotionimage=angry
	elif ans==5:
	    emotionimage=happy
	elif ans==4:
	    emotionimage=sad
	elif ans==0:
		emotionimage=neutral

	ww=(im[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3]].shape[0]
	hh=(im[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3]].shape[1]
	resized=cv2.resize(emotionimage,(hh,ww),Image.ANTIALIAS)
	print "Shape of resized emoticon",resized.shape
	print pos
	cv2.imshow('checkcheck',im[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])
	cv2.waitKey(0)
	print "shape of face",im[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3]].shape
        print resized.shape
        print im[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]].shape

        print (im[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3]].shape

        for c in range(3):
            (im[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3],c]=resized[:,:,c]*(resized[:,:,2]/255.0) +(im[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3],c] * (1.0 -resized[:,:,2]/255.0)
cv2.imshow('final',im)
cv2.waitKey(0)

cv2.imwrite(imagefile+' emotion',im)
