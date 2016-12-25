import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from getdata import getdata
import timeit

start=timeit.default_timer()
images,labels=getdata()
imagespath = "dataset/ck+/cohn-kanade-images/"
labelspath="dataset/ck+/Emotion/"
def facecrop(image):
	face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	face_cascade2=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        #print image.shape
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray,1.3,5)
        if len(faces)==0:
            faces=face_cascade2.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            face= gray[y:y+h, x:x+w]
            return face[0.1*h :h-0.1*h,0.15*h:w-0.15*h]
def createBOW():
    cntsubjects=0
    totalimages=0
    cnt=0
    cnt2=0
    dsize=1000
    bof=cv2.BOWKMeansTrainer(dsize)
    sessions=os.listdir(imagespath)
    print "Processing dataset"
    time1=timeit.default_timer()
    for subject,lsub in zip(images,labels):
        #print subject
        if len(subject)!=0:
                sub=subject[0][0][:4]
        #print sub
        for session,ll in zip(subject,lsub):
            #print session
            sess=session[0][5:8]
            #print sess,len(session)
            no_of_images=len(session)
            labelfile=ll[0]
            data=open(labelspath+sub+'/'+sess+'/'+labelfile)
            emotion=data.read()
            emotion=int(emotion[3])
            no_of_images_sofar=[0]*8
            max_images=168
            for image in session:
                #sub=image[:4]
                #sess=image[5:8]
                #print image,image[-6:-4]
                imageno=int(image[-6:-4])
                if emotion==1 or emotion==5 or emotion==6:
                    if imageno<=no_of_images and imageno>no_of_images-6:
                        #print image
                        if no_of_images_sofar[emotion]>max_images:
                            continue
                        totalimages+=1
                        no_of_images_sofar[emotion]+=1
                        gray=cv2.imread(imagespath+sub+'/'+sess+'/'+image)
                        #cv2.imshow('full',gray)
                        #cv2.waitKey(0)
                        if gray is not None:
                            justface=facecrop(gray) 
                        #cv2.imshow('face',gray)
                        #cv2.waitKey(0)
                        if justface is None:
                            print "no face",imagespath+sub+'/'+sess+'/'+image
                            cnt+=1
                            gray=cv2.imread(imagespath+sub+'/'+sess+'/'+image)
                            if gray is not None:
                                kp,dsc=sift.detectAndCompute(gray,None)
                                bof.add(dsc)
                            else:
                                cnt2+=1
                                print "no image",imagespath+sub+'/'+sess+'/'+image
                        else:
                                kp,dsc=sift.detectAndCompute(justface,None)
                                bof.add(dsc)
                if imageno<=3:
                    if no_of_images_sofar[0]>max_images:
                        continue
                    totalimages+=1
                    no_of_images_sofar[0]+=1
                    gray=cv2.imread(imagespath+sub+'/'+sess+'/'+image)
                    #cv2.imshow('full',gray)
                    #cv2.waitKey(0)
                    if gray is not None:
                        justface=facecrop(gray) 
                    #cv2.imshow('face',gray)
                    #cv2.waitKey(0)
                    if justface is None:
                        print "no face",imagespath+sub+'/'+sess+'/'+image
                        cnt+=1
                        gray=cv2.imread(imagespath+sub+'/'+sess+'/'+image)
                        if gray is not None:
                            kp,dsc=sift.detectAndCompute(gray,None)
                            bof.add(dsc)
                        else:
                            cnt2+=1
                            print "no image",imagespath+sub+'/'+sess+'/'+image
                    else:
                            kp,dsc=sift.detectAndCompute(justface,None)
                            bof.add(dsc)
            
        cntsubjects+=1
        print "Subjects processed",cntsubjects

    print "Total images used",totalimages
    print "Image for each emotion",no_of_images_sofar
    #extractfeature(image)
    #print 'hahah'
    time2=timeit.default_timer()
    print "Processing ended, Time taken",time2-time1
    print "Clustering begins"
    clus=timeit.default_timer()
    dictionary=bof.cluster()
    clus2=timeit.default_timer()
    print "Clustering ends, Time taken",clus2-clus
    print "no face",cnt
    print "no image",cnt2
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    print type(dictionary)
    bowdict.setVocabulary(dictionary)
    joblib.dump(dictionary,'trained/bow/bow.pkl')
    #filee=open('bow','w')
    #filee.write(bowdict)
    #filee.close()

def extractfeature(path):
    image=cv2.imread(path)
    image=facecrop(image)
    return bowdict.compute(image,sift.detect(image))

def trainSVM():
    train_desc=[]
    train_labels=[]
    print "Acquiring training descriptors and labels"
    ttt=timeit.default_timer()
    for i,l in zip(images,labels):
        #print i,l
        if len(i)!=0:
            subject=i[0][0][:4]
        #print subject
        for ii,ll in zip(i,l):
            #print ii,ll
            session=ii[0][5:8]
            cnt=0
            #print session,len(ii)
            no_of_images=len(ii)
            labelfile=ll[0]
            data=open(labelspath+subject+'/'+session+'/'+labelfile)
            emotion=data.read()
            max_images=168
            no_of_images_sofar=[0]*8
            for image in ii:
                imageno=int(image[-6:-4])
                if int(emotion[3])==1 or int(emotion[3])==5 or int(emotion[3])==6 :
                    #print imagespath+subject+'/'+session+'/'+image
                    #print image,imageno
                    if imageno<=no_of_images and imageno>no_of_images-6:
                        if no_of_images_sofar[int(emotion[3])]>max_images:
                            continue
                        no_of_images_sofar[int(emotion[3])]+=1
                        feature=extractfeature(imagespath+subject+'/'+session+'/'+image)
                        #print feature
                        train_desc.extend(feature)
                        cnt+=1
                        train_labels.append(emotion)
                if imageno<=3:
                    if no_of_images_sofar[0]>max_images:
                        continue
                    no_of_images_sofar[0]+=1
                    feature=extractfeature(imagespath+subject+'/'+session+'/'+image)
                    train_desc.extend(feature)
                    train_labels.append(emotion)
                    cnt+=1

                #for _ in xrange(6):
                #    train_labels.append(emotion)
    print "Sanity check, no of images of each emotion added to svm"
    print no_of_images_sofar
    tttt=timeit.default_timer()
    print "Training descriptors and labels acquired", tttt-ttt
    print "SVM training begins"
    svmtime=timeit.default_timer()
    clf=OneVsRestClassifier(svm.SVC(kernel='rbf'))
    clf.fit(np.array(train_desc),np.array(train_labels))
    svmtime2=timeit.default_timer()
    print "SVM training ended, Time taken",svmtime2-svmtime
    joblib.dump(clf,'trained/svm/svm.pkl') 
sift=cv2.xfeatures2d.SIFT_create()
sift2 = cv2.xfeatures2d.SIFT_create()
bowdict = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
##cv2.imshow('haha',facecrop(cv2.imread('abcd.jpg')))
##cv2.waitKey(0)
createBOW()
##print bowdict
trainSVM()
stop=timeit.default_timer()
#
print "Total time ",stop-start
