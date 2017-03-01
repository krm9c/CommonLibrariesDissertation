####################################################################
# Big Data Analysis
# Library  1.0 (Paper One)
# Header File Definitions
from math import *
import os,sys
import numpy as np
#############################################################################
# Print a tree
# Function 1
def print_tree(self):
        i=0
        for element in self.tree_levels:
            print ("level =",i)
            i = i+1
            for element1 in element:
                print ("element.number=",element1.number)
                print  ("element.conencted_nodes=",element1.connected_nodes)
###############################################################################
# Import Data
# import an excel file
# Function 2
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
###########################################################################
# Call for the import of data
# Function 6
def import_data(filename,sheet, dataset):
    if dataset==1:
        values  =returnxl_numpy(filename,sheet);
    elif dataset==2:
        values  =Testing_newdataset(filename,1);
    return values
##########################################################################
# Traditional MTS Codes
# Function 7
def traditional_MTS(Norm, T, par):
    # Define the sizes of the normal and the test data set
    row_normal    = Norm.shape[0];
    column_normal = Norm.shape[1];
    row_test      = T.shape[0];
    column_test   = T.shape[1];
    scalar        =  preprocessing.StandardScaler(with_mean=True, with_std = True ).fit(Norm)
    N_scaled      =  scalar.transform(Norm)
    T_scaled      =  scalar.transform(T);
    # Since we have the parameters now today, lets start with the transformation..
    mean_test=[]
    for i in range(0 , column_test):
        a=np.array(T_scaled[:,i]);
        mean_test.append(a.mean());
    mean_normal=[]
    for i in range(0 , column_test):
        a=np.array(N_scaled[:,i]);
        mean_normal.append(a.mean());
    # Perform the final transformation
    # 1. if less than penultimate level (dimension reduction)
    # 2. If at penultimate level (distance calculation)
    mn           = np.array(mean_normal);
    mt           = np.array(mean_test);
    MD_test = np.zeros((T_scaled.shape[0],1))
    # print "The norm value", (np.dot((mt-mn),np.transpose((mt-mn))))
    if par ==0:
        # Correlation matrix
        Correlation_Matrix=np.corrcoef(np.transpose(N_scaled));
        Inverse_correlation=np.linalg.pinv(Correlation_Matrix);
        # Calculate the value of the MD
        for i in range (0,row_test):
            if ((np.dot(np.dot((T_scaled[i,:]-mn),Inverse_correlation),(T_scaled[i,:]-mn).transpose())) < 0):
                print ("problem")
            MD_test[i]=np.linalg.norm(np.dot((T_scaled[i,:]-mn),np.linalg.cholesky(Inverse_correlation)), 0.9);
    else:
        # Correlation matrix
        Correlation_Matrix=np.corrcoef(np.transpose(N_scaled));
        Inverse_correlation=np.linalg.inv(Correlation_Matrix);
        # Calculate the value of the MD
        for i in range (0,row_test):
                MD_test[i]=np.linalg.norm(np.dot((mt-mn),np.linalg.cholesky(Inverse_correlation)), 0.9);
    return MD_test