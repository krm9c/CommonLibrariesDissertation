####################################################################
# Big Data Analysis
# Library  2.0 version
##########################################################################
import random
import os,sys
import numpy as np
import warnings
import tflearn
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	from   sklearn import preprocessing
from sklearn.datasets import  make_classification
############################################################################
# Set path for the data too
path = "/Users/krishnanraghavan/Documents/Data-case-study-1"
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Common_Libraries')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_1_codes')
sys.path.append('/Users/krishnanraghavan/Dropbox/Work/Research/Paper_2_codes')
from Library_Paper_one  import import_data, traditional_MTS
###################################################################################
# Global infinite loop of data
def Inf_Loop_data(par, That, Yhat, classes):
		return (That + 0.1*np.random.normal(0, 0.1, (That.shape[0], That.shape[1])) ), tflearn.data_utils.to_categorical(Yhat, classes)
###################################################################################
# Rolling Element Bearing Dataset
# Bootstrap sampling
def Bearing_Samples(path, sample_size, randfile):
	# Define Arrays
	CurrentFileNL=[]
	CurrentFileIR=[]
	CurrentFileOR=[]
	CurrentFilenorm=[]
	rand = [0 for i in xrange(sample_size)];
	File_1  = ['IR1.xls', 'OR1.xls','Normal_1.xls','NL1.xls'];
	File_2  = ['IR2.xls', 'OR2.xls','Normal_2.xls','NL2.xls'];
	File_3  = ['IR3.xls', 'OR3.xls','Normal_3.xls','NL3.xls'];
	if randfile == 1:
		File = File_1
	elif randfile == 2:
		File = File_2
	else:
		File = File_3

	for f in File:
		filename = os.path.join(path,f);
		if f.startswith('NL'):
			sheet='Test';
			temp=np.array(import_data(filename,sheet, 1));
			for i in range(0,sample_size):
				rand[i]= random.randint(0,(temp.shape[0]-1))
			CurrentFileNL = temp[rand,:];

		if f.startswith('IR'):
			sheet='Test'
			temp = import_data(filename,sheet, 1);
			for i in range(0,sample_size):
				rand[i]= random.randint(0,(temp.shape[0]-1))
			CurrentFileIR = temp[rand,:]

		if f.startswith('OR'):
			sheet='Test'
			temp = import_data(filename,sheet, 1);
			for i in range(0,sample_size):
				rand[i]= random.randint(0,(temp.shape[0]-1));
			CurrentFileOR = temp[rand,:];

		if f.startswith('Normal'):
			sheet='normal';
			temp = import_data(filename,sheet, 1);
			for i in range(0,sample_size):
				rand[i]= random.randint(0,(temp.shape[0]-1))
			CurrentFilenorm = temp[rand,:];
    #  We end this function by returning all the arrays and the fault centroids
	return (np.concatenate(( CurrentFileNL, CurrentFileIR, CurrentFileOR, CurrentFilenorm \
	))), tflearn.data_utils.to_categorical( np.concatenate(( (np.zeros(CurrentFilenorm.shape[0])),\
	  (np.zeros(CurrentFileNL.shape[0])+1), (np.zeros(CurrentFileIR.shape[0])+2),\
	   (np.zeros( CurrentFileOR.shape[0])+3)) ), 4)

# Time Based sampling
def Bearing_Non_Samples_Time(path, num, classes):
	sheet    = 'Test';
	f        = 'IR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	print filename
	Temp_IR  =  np.array(import_data(filename,sheet, 1));
	f        = 'OR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_OR  =  np.array(import_data(filename,sheet, 1));
	f        = 'NL'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_NL  =  np.array(import_data(filename,sheet, 1));
	sheet    = 'normal';
	f        = 'Normal_'+str(1)+'.xls'
	filename = os.path.join(path,f);
	Temp_Norm= np.array(import_data(filename,sheet, 1));
	T = np.concatenate((Temp_Norm, Temp_NL, Temp_IR, Temp_OR))
	Y = np.concatenate((np.zeros(Temp_Norm.shape[0]), np.zeros(Temp_NL.shape[0])+1, np.zeros(Temp_IR.shape[0])+2, np.zeros(Temp_OR.shape[0])+3))
	return T,tflearn.data_utils.to_categorical(Y, classes)
###################################################################################
# Artificial Dataset import
def DataImport_Artificial(n_sam, n_fea, n_inf, classes):
	X,y = make_classification(n_samples = n_sam, n_features = n_fea, n_informative = n_inf, n_redundant = (n_fea-n_inf), n_classes = classes, n_clusters_per_class = 1, class_sep = 1, hypercube = True, shuffle = True, random_state = 9000)
	index_1 = [i for i,v in enumerate(y) if v == 0 ]
	index_2 = [i for i,v in enumerate(y) if v == 1 ]
	index_3 = [i for i,v in enumerate(y) if v == 2 ]
	index_4 = [i for i,v in enumerate(y) if v == 3 ]
	Data_class_1 = X[index_1,:]
	L1 = y[index_1];
	L2 = y[index_2];
	L3 = y[index_3];
	L4 = y[index_4];
	Data_class_2 = X[index_2,:]
	Data_class_3 = X[index_3,:]
	Data_class_4 = X[index_4,:]
	T = np.concatenate((Data_class_1, Data_class_2, Data_class_3, Data_class_4))
	Y = np.concatenate((L1, L2, L3, L4))
	return T, tflearn.data_utils.to_categorical(Y, classes)
########################################################################################
def Data_MNIST():
	# Standard scientific Python imports
	import matplotlib.pyplot as plt
	# Import datasets, classifiers and performance metrics
	from sklearn import datasets, svm, metrics
	# Import numpy
	import numpy as np
	# The digits dataset
	digits = datasets.load_digits()
	# The data that we are interested in is made of 8x8 images of digits, let's
	# have a look at the first 4 images, stored in the `images` attribute of the
	# dataset.  If we were working from image files, we could load them using
	# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
	# images, we know which digit they represent: it is given in the 'target' of
	# the dataset.
	images = np.array(digits.images)
	labels = np.array(digits.target)
	# To apply a classifier on this data, we need to flatten the image, to
	# turn the data in a (samples, feature) matrix:
	T = images.reshape((images.shape[0], -1))
	Y = labels.reshape((labels.shape[0], -1))
	return T, tflearn.data_utils.to_categorical(Y, 10)

def DataImport(num, classes=4, file=0, sample_size = 1000, features = 100):
	if num ==0:
		# Import Artificial Data-set
		That, Yhat = DataImport_Artificial(n_sam = sample_size, n_fea=features, n_inf=2, classes= classes)
	elif num ==1:
		#  Import Rolling Element Data
		That, Yhat = Bearing_Non_Samples_Time(path, 1, classes)
	elif num ==2:
		That, Yhat = Bearing_Samples(path, sample_size, 1)
	elif num ==3:
		That, Yhat = Data_MNIST()
	return That, Yhat
