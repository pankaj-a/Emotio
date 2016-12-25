import cv2
import sys
import numpy as np
from sklearn.externals import joblib
import timeit

def facecrop(image):
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade2=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    print image.shape
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5,minSize=(70,70))
    if len(faces) == 0:
        faces = face_cascade2.detectMultiScale(gray, 1.3, 5)
    if len(faces)==0:
        print "no faces"
        return (None,None)
    face=[]
    position=[]
    for (x,y,w,h) in faces:
        face.append(gray[y:y+h, x:x+w])
        position.append((x,y,w,h))

    return (face,position)

def extractfeature(image):
    sift= cv2.xfeatures2d.SIFT_create()
    image,position=facecrop(image)
    if image==None:
        return (None,None)
    gray=image
    print len(image),len(position)
    print type(image),type(image[0])
    for i,pos in zip(range(len(image)),position):
        print type(pos)
        image[i]=image[i][0.1*pos[3] :pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3]]
        kp,dsc=sift.detectAndCompute(image[i],None)
        img=cv2.drawKeypoints(image[i],kp,None)
        cv2.imshow('face',img)
    #cv2.waitKey(0)
    features=[];
    for i in range(len(image)):
        features.append((bowdict.compute(image[i],sift.detect(image[i]))))
    return (features,position)

def start():
    #videofile=sys.argv[1]
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    fourcc=cv2.VideoWriter_fourcc(*'X264')
    print frame.shape
    dim=(frame.shape[1],frame.shape[0])
    out=cv2.VideoWriter('out.mp4',fourcc,20.0,dim)
    while True:
        ret,frame=cap.read()
        test=frame
        if test==None:
            #out.write(frame)
            #cv2.imshow('frame',frame)
            print 'No video, Exiting'
            break
        test,position=extractfeature(test)
        if test==None:
            print 'no face'
            out.write(frame)
            cv2.imshow('frame',frame)
            continue
        if len(test)==0:
            print 'no face'
            out.write(frame)
            cv2.imshow('frame',frame)
            continue
        start1=timeit.default_timer()
        answer=[]
        for i,pos in zip(range(len(test)),position):
            answer.append(int(clf.predict(test[i])[0][3]))
        #ans=int(clf.predict(test)[0][3])
        print answer
        stop=timeit.default_timer()
        #print stop-start1
        for ans,pos in zip(answer,position):
            if ans==1:
                emotionimage=angry
            elif ans==6:
                emotionimage=sad
            elif ans==5:
                emotionimage=happy
            elif ans==0:
                emotionimage=neutral
	    ww=(frame[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3]].shape[0]
	    hh=(frame[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3]].shape[1]
	    resized=cv2.resize(emotionimage,(hh,ww),interpolation = cv2.INTER_AREA)
	    print "Shape of resized emoticon",resized.shape
            for c in range(3):
                (frame[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3],c]=resized[:,:,c]*(resized[:,:,2]/255.0) +(frame[pos[1]:pos[1]+pos[3],pos[0]:pos[0]+pos[2]])[0.1*pos[3]:pos[3]-0.1*pos[3],0.15*pos[3]:pos[2]-0.15*pos[3],c] * (1.0 -resized[:,:,2]/255.0)
        cv2.imshow('frame',frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

sift2 = cv2.xfeatures2d.SIFT_create()
bowdict = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))

dictionary=joblib.load('trained/bow/bow.pkl')
bowdict.setVocabulary(dictionary)
clf=joblib.load('trained/svm/svm.pkl')
#emotions=['angry','contempt','disgust','fear','happy','sadness ','surprise']
#emoticons= load_emoticons(emotions)
angry=cv2.imread('anger.png')
happy=cv2.imread('happy.png')
sad=cv2.imread('sadness.png')
neutral=cv2.imread('neutral.png')

start()
