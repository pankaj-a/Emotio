Emotio: An emotion classifier built in Python using OpenCV and Scikit-learn.

What does it do:

	The classifier is a SVM oneVSrest classifier and was taught on the ck+
	(http://www.consortium.ri.cmu.edu/ckagree/) dataset. It takes an input image and extracts faces from it
	and tests the emotion as angry, happy or sad. The trained model is saved in the trained folder.

Project tested in Ubuntu 14.04 in the following environment:

	OpenCV 3.0.0
	
	Scikit-learn 0.17.1 (later versions may work too)
	
	Python 2.7.6
	
	
	How to Install:
	
	OpenCV: http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
	
	Scikit-learn: http://scikit-learn.org/stable/install.html
	
How to run:
	
	Get the Cohn-Kanade extended (ck+) dataset and put in the dataset folder.
	
	Run train.py. It processes the dataset and trains a model for the emotions happy, angry and sad.
	The trained model is stored in trained/bow and trained/svm.
	
	Run test.py with argument as the filename to be tested. It creates a file as 'filename+emotion' 
	and pastes the emoji on it.
	
	For a live test, run webcam.py. It takes the input from the webcam and displays the emoji directly
	on the live feed. It also saves an output file as out.mp4.
	

Reference:

	http://cs229.stanford.edu/proj2015/158_report.pdf
	
