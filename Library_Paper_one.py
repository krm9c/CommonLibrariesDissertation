####################################################################
# Big Data Analysis
# Library  1.0 (Paper One)
# Header File Definitions
from math import *
import os,sys
import numpy as np
from sklearn import preprocessing
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
########################################################################

from scipy.spatial.distance  import pdist, squareform
## Distance Correlation calculation
def distance_covariance(x,y):
        x = np.reshape(x, [-1, 1])
        y = np.reshape(y, [-1, 1])

        # calculate distance matrix first
        A = squareform(pdist(x, 'euclidean'))
        N = A.shape[0]
        one_n = np.ones((N,1))/N
        A = A -A.dot(one_n) -np.transpose(one_n).dot(A) +(np.transpose(one_n).dot(A)).dot(one_n)
        np.fill_diagonal(A, 0)

        # Second distance matrix
        B = squareform(pdist(y, 'euclidean'))
        N = B.shape[0]
        one_n = np.ones((N,1))/N
        B = B -B.dot(one_n) -np.transpose(one_n).dot(B) +np.transpose(one_n).dot(B).dot(one_n)
        np.fill_diagonal(B, 0)

        Temp1 = np.multiply(A, B)
        Temp2 = np.multiply(A, A)
        Temp3 = np.multiply(B, B)

        nu_xy = (1/float(x.shape[0]*y.shape[0]))*np.sum(Temp1)
        nu_xx = (1/float(x.shape[0]*x.shape[0]))*np.sum(Temp2)
        nu_yy = (1/float(y.shape[0]*y.shape[0]))*np.sum(Temp3)
        if nu_xx*nu_yy < 1e-5:
            return 1e-3
        else:
            t_cor = nu_xy/float(sqrt(nu_xx*nu_yy))
            return t_cor

## Dependence Matrix calculation
def dependence_calculation(X):
    n  = X.shape[0];
    m  = X.shape[1];
    C  = np.zeros((m,m));
    rng = np.random.RandomState(0)
    idx = rng.randint(n, size=1000)
    P = X[idx, :]
    for i in xrange(0,m):
        x = P[:,i]
        for j in xrange(0,m):
            y = P[:,j]
            C[i][j] = distance_covariance(x, y);
    return C




# Traditional MTS Codes
# Function 7
def traditional_MTS(Norm, T, par):

    # Define the sizes of the normal and the test data set
    row_normal    = Norm.shape[0];
    column_normal = Norm.shape[1];
    row_test      = T.shape[0];
    column_test   = T.shape[1];

    # use the mean of both to transform each of the array
    scalar        =  preprocessing.StandardScaler(with_mean = True, with_std = True).fit(Norm)
    N_scaled      =  scalar.transform(Norm)
    T_scaled      =  scalar.transform(T)


    # print("Ref")
    # from scipy import stats
    # print(stats.describe(N_scaled))
    # print("Test")
    # print(stats.describe(T_scaled))

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
    Correlation_Matrix = dependence_calculation(N_scaled)
    Inverse_correlation=np.linalg.pinv(Correlation_Matrix);
    # print "The norm value", (np.dot((mt-mn),np.transpose((mt-mn))))
    if par ==0:
        # Calculate the value of the MD
        for i in range (0,row_test):
            if ((np.dot(np.dot((T_scaled[i,:]-mn),Inverse_correlation),(T_scaled[i,:]-mn).transpose())) < 0):
                print ("problem")
            MD_test[i] = np.linalg.norm(np.dot((T_scaled[i,:]-mn),np.linalg.cholesky(Inverse_correlation)), 2);
    else:
        # Calculate the value of the MD
        for i in range (0,row_test):
                MD_test[i]=np.linalg.norm(np.dot((mt-mn),np.linalg.cholesky(Inverse_correlation)),2);

    return MD_test
