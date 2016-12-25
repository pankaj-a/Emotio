from getdata import getdata
from sklearn.externals import joblib
import cv2
import timeit

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
def extractfeature(path):
    image=cv2.imread(path)
    image=facecrop(image)
    return bowdict.compute(image,sift.detect(image))

imagespath = "dataset/ck+/cohn-kanade-images/"
labelspath="dataset/ck+/Emotion/"
images,labels=getdata()
sift = cv2.xfeatures2d.SIFT_create()
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
total=[0]*8
cnt=[0]*8
percent=[0]*8
start=timeit.default_timer()
for isub,lsub in zip(images,labels):
    if len(isub)!=0:
        subject=isub[0][0][:4]
    for ii,ll in zip(isub,lsub):
        session=ii[0][5:8]
        no_of_images=len(ii)
        labelfile=ll[0]
        data=open(labelspath+subject+'/'+session+'/'+labelfile)
        emotion=data.read()
        emotion=int(emotion[3])
        for image in ii:
            imageno=int(image[-6:-4])
            if emotion==1 or emotion==5 or emotion==6 :
                if imageno<=no_of_images and imageno>no_of_images-6:
                	total[emotion]+=1
                	feature=extractfeature(imagespath+subject+'/'+session+'/'+image)
                	ans=clf.predict(feature)[0]
                	ans=int(ans[0:len(ans)-1][3])
                	if emotion==ans:
                	    cnt[emotion]+=1
	    if imageno<=3:
		total[0]+=1
		feature=extractfeature(imagespath+subject+'/'+session+'/'+image)
                ans=clf.predict(feature)[0]
                ans=int(ans[0:len(ans)-1][3])
                if ans==0:
                    cnt[0]+=1
stop=timeit.default_timer()
print "Test time taken",stop-start
print "Predicted"
print cnt
print "Total Encountered"
print total
for i in range(8):
    if total[i]!=0:
        percent[i]=float(cnt[i])/float(total[i])
print "Percentage Accuracy"
print percent
