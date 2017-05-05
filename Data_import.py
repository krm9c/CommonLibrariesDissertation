###################################################################
# Big Data Analysis
###################################################################
import random
import os,sys
import numpy as np
import warnings
import tflearn
import math
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",\
	category=DeprecationWarning)
	from   sklearn import preprocessing
from sklearn.datasets import  make_classification
####################################################################
# Set path
path = '../data/'
sys.path.append('../CommonLibrariesDissertation')

####################################################################
# import an excel file
def returnxl_numpy(filename, sheet):
    import xlrd
    #Open a workbook
    workbook  = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_name(sheet)
    num_rows  = worksheet.nrows
    num_cells = worksheet.ncols
    # Declare a numpy array
    tempxl=np.zeros((num_rows,num_cells))
    #Store data on the numpy array
    for curr_row in range(0,num_rows):
        for curr_cell in range(0,num_cells):
            tempxl[curr_row,curr_cell] = worksheet.cell_value(curr_row, curr_cell)
    return tempxl


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
			temp=np.array(returnxl_numpy(filename,sheet));
			for i in range(0,sample_size):
				rand[i]= random.randint(0,(temp.shape[0]-1))
			CurrentFileNL = temp[rand,:];
		if f.startswith('IR'):
			sheet='Test'
			temp = returnxl_numpy(filename,sheet);
			for i in range(0,sample_size):
				rand[i]= random.randint(0,(temp.shape[0]-1))
			CurrentFileIR = temp[rand,:]
		if f.startswith('OR'):
			sheet='Test'
			temp = returnxl_numpy(filename,sheet);
			for i in range(0,sample_size):
				rand[i]= random.randint(0,(temp.shape[0]-1));
			CurrentFileOR = temp[rand,:];
		if f.startswith('Normal'):
			sheet='normal';
			temp = returnxl_numpy(filename,sheet);
			for i in range(0,sample_size):
				rand[i]= random.randint(0,(temp.shape[0]-1));
			CurrentFilenorm = temp[rand,:];
	T = np.concatenate((CurrentFileNL, CurrentFileIR, CurrentFileOR, CurrentFileNorm))
	Y = np.concatenate((np.zeros(CurrentFileNL.shape[0])+1, np.zeros(CurrentFileIR.shape[0])+2, np.zeros(CurrentFileOR.shape[0])+3, np.zeros(CurrentFileNorm.shape[0])+4))
	return T, y



def data_PHM08(path_phm08):
	temp_data_class_1 = []
	temp_data_class_2 = []
	Train_temp = np.loadtxt(path_phm08+'/train.txt')
	for i in xrange(1,219):
		index = [j for j,v in enumerate(Train_temp[:,0]) if v == i ]
		temp = Train_temp[index,:]
		temp_data_class_1.extend( temp[  0: int(math.ceil( (50/float(100))*temp.shape[0])) , 5:temp.shape[1] ])
		temp_data_class_2.extend( temp[  int(math.ceil( (50/float(100))*temp.shape[0]))+1:temp.shape[0] , 5:temp.shape[1] ])

	temp_data =[]
	temp_data.extend(temp_data_class_1)
	temp_data.extend(temp_data_class_2)
	y_train = np.concatenate( (np.zeros( len(temp_data_class_1) )+1, np.zeros( len(temp_data_class_2) )+2) )
	X_train = np.array(temp_data)

	temp_data_class_1 = []
	temp_data_class_2 = []
	Test_temp = np.loadtxt(path_phm08+'/test.txt')
	for i in xrange(1,219):
		index = [j for j,v in enumerate(Test_temp[:,0]) if v == i ]
		temp = Test_temp[index,:]
		temp_data_class_1.extend( temp[  0: int(math.ceil( (50/float(100))*temp.shape[0])) , 5:temp.shape[1] ])
		temp_data_class_2.extend( temp[  int(math.ceil( (50/float(100))*temp.shape[0]))+1: temp.shape[0] , 5:temp.shape[1] ])
	temp_data =[]
	temp_data.extend(temp_data_class_1)
	temp_data.extend(temp_data_class_2)
	y_test = np.concatenate( (np.zeros( len(temp_data_class_1) )+1, np.zeros( len(temp_data_class_2) )+2) )
	X_test = np.array(temp_data)
	T = np.concatenate((X_train, X_test))
	print T.shape
	Y = np.concatenate((y_train, y_test))
	print Y.shape
	P = list(np.random.permutation(T.shape[0]))
	y = Y[P]
	T = T[P,:]
	return T, y


def Bearing_Non_Samples_Time(path, num, classes):
	sheet    = 'Test';
	f        = 'IR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_IR  =  np.array(returnxl_numpy(filename,sheet));
	f        = 'OR'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_OR  =  np.array(returnxl_numpy(filename,sheet));
	f        = 'NL'+str(num)+'.xls'
	filename =  os.path.join(path,f);
	Temp_NL  =  np.array(returnxl_numpy(filename,sheet));
	sheet    = 'normal';
	f        = 'Normal_'+str(1)+'.xls'
	filename = os.path.join(path,f);
	Temp_Norm= np.array(returnxl_numpy(filename,sheet));
	T = np.concatenate((Temp_NL, Temp_IR, Temp_OR, Temp_Norm))
	Y = np.concatenate((np.zeros(Temp_NL.shape[0])+1, np.zeros(Temp_IR.shape[0])+2, np.zeros(Temp_OR.shape[0])+3, np.zeros(Temp_Norm.shape[0])+4))
	return T,Y


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
	return T, Y

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
	return T, Y


def sensorless(path_sensorless):
	from sklearn.datasets import load_svmlight_file
	data = load_svmlight_file(path_sensorless+"/Sensorless.scale")
	dense_vector = np.zeros((data[0].shape[0],data[0].shape[1]))
	data[0].toarray(out = dense_vector)
	return dense_vector, data[1]



def DataImport(num, classes=10, file=0, sample_size = 10000, features = 200):
	if num ==0:
		That, Yhat = DataImport_Artificial(n_sam = sample_size, n_fea=((2*classes)+features), n_inf=(2*classes), classes= classes)
	elif num ==11:
		path_roll = path + 'RollingElement'
		That, Yhat = Bearing_Non_Samples_Time(path_roll, 1, classes)
	elif num ==12:
		path_roll = path + 'RollingElement'
		That, Yhat = Bearing_Samples(path_roll, sample_size, 1)
	elif num ==2:
		That, Yhat = Data_MNIST()
	elif num ==3:
		path_sensorless= path+ 'Sensorless'
		That, Yhat = sensorless(path_sensorless)
	elif num ==4:
		path_phm08 = path+'PHM08'
		That, Yhat = data_PHM08(path_phm08)

	return That, Yhat
