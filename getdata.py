import cv2
import numpy as np
import os
from sklearn import svm
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

def getdata():
	cnt=0
	image_paths=[]
	imagespath = "dataset/ck+/cohn-kanade-images/"
	labelspath="dataset/ck+/Emotion/"
	allsessions=os.listdir(imagespath)#folders of each session
	#print allsessions
	finalimages=[]
	finallabels=[]
	for session in allsessions:
		cur_images=imagespath+session
		cur_labels=labelspath+session
		#print cur_images,cur_labels
		#print os.listdir(cur_images)
		#print os.listdir(cur_labels)
		x=os.listdir(cur_images)
		y=os.listdir(cur_labels)
		images=[]
		labels=[]
		for i in x:
                    if i[0]=='.':
                        pass
                    else:
                        images.append(os.listdir(cur_images+'/'+i))
		for i in y:
                    if i[0]=='.':
                        pass
                    else:
                        labels.append(os.listdir(cur_labels+'/'+i))
		finalimages.append(images)
		finallabels.append(labels)
		#print len(images)
		#print labels,len(labels)
	#print len(finalimages),len(finallabels)
	#print finalimages
	#print finallabels
	#for i,l in zip(finalimages,finallabels):
	#    for ii,ll in zip(i,l):
        #        if ii[0][:8]!=ll[0][:8]:
        #            print ii,ll
        #            cnt+=1
        #print cnt
	return finalimages,finallabels

#getdata()


#finalimages - [ [ subject 1 [sess 1 (multiple images) (emotion 1)] [sess 2](emotion 2)] , [subject 2] , ..]

#finallabels - [ [ subject 1 [sess 1 ( single text file) ] [sess 2] ] [subject 2] ... ]


#training_paths=[]#every file
#names_paths=[]
#images=[]
#labels=[]
#for p in training_names:
#	folderInEachSession=os.listdir("dataset/ck+/cohn-kanade-images/S010/"+p)
#	for j in folderInEachSession:
#		imagesInEachSession.append("dataset/ck+/cohn-kanade-images/S010/"+p+"/"+j)
#print images
#print len(images)
