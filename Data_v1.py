##############################################################################
# import any dat File and return values
# Function 5
###############################################################################
def import_generaldatfile(filename):
    contents=[]
    # Get data from a dat file
    # open a file
    for line in open(filename,'r'):
        listWords=line.split("\t")
        contents.append(listWords)

    # Declare a numpy array take data out of the contents  array and store it in the numpy array
    total_measurements=len(contents)+1
    total_attributes=len(contents[2])
    Data=np.zeros((total_measurements,total_attributes));
    i=0
    for element in contents:
        j=0
        i=i+1
        for number in element:
            if (number!='\n'):
                # print 'number is',number
                Data[i,j]=(float(number))
                j=j+1
    return total_measurements,total_attributes,Data

# Dataset 2
##########################################################################
# For the Case_study_2
# Function 6
###############################################################################
def GasSensorArrayImport(filename,factor):
	total_measurements,total_attributes,temp=import_generaldatfile(filename)
	index=[0,1,2,3,4,5,6,7,8,9,10,11,20,29,38,47,56,65,74,83]
	total_attributes=total_attributes-len(index)
	# print 'Length of the stuff',total_measurements,total_attributes
	values=np.zeros((total_measurements+1,total_attributes))
	i=0;
	maximus=factor;
	# print "Maximum of the array is", maximus
	for element in temp:
		j=0;
		i=i+1;
		y=0;
		for number in element:
			if j not in index:
				values[i,y]=(number/maximus);
				y=y+1
			j=j+1
	return values

def Gas_array_data_import():
    import numpy as np
    import random
    path_sensor = '/media/krm9c/My Book/Research_Krishnan/Data/Gas-Sensor-Array/'
    File_array = ['Acetone', 'Ammonia', 'Benzene', 'CO', 'CO_4000', 'Ethylene', 'Methane', 'Methanol', 'Toluene']
    import os, sys
    for element in File_array:
        path = path_sensor+'/'+element
        Temp = []
        count = 0
        for fname in os.listdir(path):
            P = GasSensorArrayImport(path+'/'+fname, 1)
            count = count+1
            Temp1 = []
            for i in xrange(1000):
                rand= [random.randint(0,(P.shape[0]-1)) for j in xrange(100)]
                X = np.mean(P[rand,0:72], axis = 0, dtype=np.float64).reshape((1, 72))
                Temp1.extend(X)
                # print np.array(Temp1).shape
            Temp.extend(Temp1)
        print element, np.array(Temp).shape, count
        np.savetxt(element+'.csv', np.array(Temp), delimiter = ',' )



def data_gas_array(path_gas):
    import numpy as np
    import random
    path_sensor = '/media/krm9c/My Book/Research_Krishnan/Data/Gas-Sensor-Array/'
    File_array = ['Acetone', 'Ammonia', 'CO', 'CO_4000', 'Ethylene', 'Methane', 'Methanol', 'Toluene']
    import os, sys
    Temp = []
    Temp_y = []
    i = 0
    for element in File_array:
        P = np.loadtxt(element+'.csv', delimiter = ',' )
        print P.shape
        Temp.extend(P)
        i = i+1
        Temp_y.extend(np.zeros((P.shape[0],1))+i)
    print 'data', np.array(Temp).shape
    print 'y', np.array(Temp_y).shape

data_gas_array('hola')
