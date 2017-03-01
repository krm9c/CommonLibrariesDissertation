####################################################################
# Big Data Analysis
# Library  2.0 version
##########################################################################
import random
import os,sys
from   math                            import *
import numpy                               as np
import warnings
from sklearn import preprocessing
from Library_Paper_one                 import import_data, traditional_MTS
from itertools import izip
import time
Paper = 'paper_2'

##########################################################################
# GDM-HDR The Hierarchical Dimension Reduction Technique
class level():
    # Default constructor for the class
        def __init__(self):
                self.total_level=0
                self.tree_levels=[]
#####################################################################
# Class definition for the nodes, so the representation is mdtree=(node,level)
        class my_tree_node():
                def __init__(self, row):
                    self.Transformed = np.zeros((row, 1));
                    self.par_trans = []
                    self.scaler = []
                    self.number = 0
                    self.connected_nodes = []
	###########################################################################
	# Resize the tree for working with the new data sizes (The columns in the data still remain the same)
        def resize(self, row):
                for level in self.tree_levels:
                    for node in level:
                        node.Transformed= np.resize(node.Transformed,(row,1))
    # The next three functions do the job of generating the tree as we want them to be
	##########################################################################
	# This function essentially generates all the groups in the tree
        def generate_group(self,level_list,gsize):
            temp  = []
            group_list = []
            node_count = -1
            if (len(level_list))==1:
                return temp;
            vargroup=0
            if (gsize == 1):
                print ("Size of the  level is", len(level_list))
                Gsize = input("Enter Group Size");
                vargroup=input("If you want to vary within a level, enter 1");
            else:
                Gsize=gsize;
            g=0;
            while (node_count < len(level_list)-1):
                g=0;
                group_list=[]
                if vargroup==1:
                    print ("Current groupings are ", temp)
                    print  ("Total number of elements left ", (len(level_list)-node_count));
                    Gsize= input("size of this group")
                while (g < Gsize) and (node_count < len(level_list)-1):
                    node_count=node_count+1;
                    group_list.append(level_list[node_count].number)
                    #print "The value of s in this loop is", s;
                    g=g+1;
                if ((len(level_list)-node_count-1)==1):
                    group_list.append(level_list[len(level_list)-1].number)
                    node_count = node_count+1;
                #print "The array that is being appended", s;
                temp.append(group_list);
            # print "The array going out is", temp;
            return temp
    ##########################################################################
	# Generate each level of the tree
	# Function 12
        def generate_level(self, total_variables, list_prev_level, row):
            temp_curr_level_list=[];
            # Run the loop for all the variables in the previous level
            for i in xrange(0,total_variables):
                tree_node = self.my_tree_node(row)
                tree_node.number = i
                tree_node.connected_nodes.append(list_prev_level[i])
                temp_curr_level_list.append(tree_node)
            # Return the list created for this level
            return temp_curr_level_list
	##########################################################################
	# Generate the overall tree.
        def tree_generate(self, row, column, gsize):
                curr_level_list=[]
                prev_level_list=[]
                tree=level()
                # For all the leaf nodes in the tree, let us initialize the connection as zero
                for i in xrange(0,column):
                    prev_level_list.append(0)
                # Start making connection upwards
                while len(prev_level_list)>=1:
                    curr_level_list= self.generate_level(len(prev_level_list),prev_level_list, row)
                    prev_level_list = []
                    prev_level_list= self.generate_group(curr_level_list,gsize)
                    tree.tree_levels.append(curr_level_list)
                return tree
    # The next three functions do the job of performing all th calculations.
    # All the tree generation part is done by now.
	##########################################################################
	# Calculating the MD value.
	# Function 13
        def calc_transformation_par(self, D_untransformed, level_flag ):
            try:
                # We calculate the parameters here.
                row_test =  D_untransformed.shape[0];
                column_test =  D_untransformed.shape[1];
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    temp_scalar = preprocessing.StandardScaler(with_mean = False, with_std = True).fit(D_untransformed)
                    D_scaled = temp_scalar.transform(D_untransformed)
                # Get the transformation parameters and store them
                if level_flag < 2:
                    if Paper =='paper_1':
                        dist_Corr = Pearson_Corr(D_scaled, 100)
                    else:
                        dist_Corr =  Dist_Corr_Pooled(D_scaled, 100)
                    Transform_array =  np.zeros((row_test, 1))
                    from scipy import linalg as LA
                    e_vals, e_vecs = LA.eig(dist_Corr)
                    eig_pairs = [(np.abs(e_vals[i]), e_vecs[:,i]) for i in xrange(len(e_vals))]
                    eig_pairs.sort(key=lambda x: x[0], reverse=True)
                    pc = eig_pairs[0][1].reshape(column_test,1)
                    Transform_array= np.array([np.dot(D_untransformed[i,:], pc) for i in xrange(row_test)])
                    return Transform_array, pc, temp_scalar
                elif level_flag ==2:
                    return D_untransformed, None, temp_scalar
            except np.linalg.linalg.LinAlgError as err:
                    print ("Linalg error occured, Error")
                    exit()
###############################################################################
        def calc_transformation(self, D_untransformed, level_flag, Transformer, scaler):
            try:
                # We calculate the parameters here.
                row_test =  D_untransformed.shape[0]
                if (level_flag == 2):
                    return D_untransformed
                else:
                    Transform_array = np.array([np.dot(scaler.transform(D_untransformed[i,:]\
                    .reshape(1,-1)), Transformer) for i in xrange(row_test)])
                    return Transform_array
            except np.linalg.linalg.LinAlgError as err:
                print ("Linalg error occured, Error")
                exit()
###########################################################################
# Traverse each node of the tree
# Function 15
        def traverse_tree(Tree, store_par):
            # Loop through all the levels in the tree first.
            for i in xrange(1,len(Tree.tree_levels)):
                if (i==1):
                    level_flag   = 0;
                elif i == (len(Tree.tree_levels)-1):
                    level_flag   = 2;
                else:
                    level_flag   = 1;
                for group in Tree.tree_levels[i]:
                    group_connectivity=group.connected_nodes
                    temp=group_connectivity[0]
                    D_untransformed = np.zeros((Tree.tree_levels[1][2].Transformed.shape[0] , len(temp)))
                    var=0
                    for element in group_connectivity:
                        for e in element:
                            D_untransformed[:,var]=(Tree.tree_levels[i-1][e].Transformed[:,0]).reshape(-1)
                            var=var+1
                    if store_par == 0:
                        group.Transformed, group.par_trans, group.scaler = Tree.calc_transformation_par(D_untransformed, level_flag );
                        if level_flag == 2:
                            Data_reduced = group.Transformed
                    elif store_par == 1:
                        group.Transformed  = Tree.calc_transformation(D_untransformed, level_flag, group.par_trans, group.scaler);
                        if level_flag == 2:
                            Data_reduced = group.Transformed
            return Data_reduced
###########################################################################
# This is where the tree generation and the transformation is called
def initialize_calculation(T, Data, gsize, par_train):
    # Part -- 1, Generation of the tree parameters
    row    = Data.shape[0];
    column = Data.shape[1];
    if par_train == 0 :
        # Generate the tree and groups
        T =  level()
        T =  T.tree_generate(row, column,  gsize)
    # Part -- 2, Calculate parameters at every level and store.
    i = 0
    T.resize(row)
    for element in T.tree_levels[0]:
        if i < column:
            element.Transformed[:,0]  = Data[:,i]
            i = i+1
    Data_reduced = T.traverse_tree(par_train);
    return Data_reduced, T



############################################################################
##  Correlation Calculation
#############################################################################
# Calculate Pearson Correlation
def pcc(X, Y):
   ''' Compute Pearson Correlation Coefficient. '''
   # Normalise X and Y
   X -= X.mean(0)
   Y -= Y.mean(0)
   # Standardise X and Y
   X /= X.std(0)
   Y /= Y.std(0)
   # Compute mean product
   return np.mean(X*Y)
########################################################
# calculate Partial Sum
def partialsum(Px,Py,Pc):
    # First shuffle y and c  according to x
    n = Px.shape[0]
    # Sort the data
    Ix = np.argsort(Px)
    sorted_lists = sorted(izip(Px, Py, Pc), reverse=False, key=lambda x: x[0])
    Px, Py, Pc = [[p[i] for p in sorted_lists] for i in range(3)]
    Py = np.array(Py)
    Pc = np.array(Pc)
    Iy = np.argsort(Py)
    # Declare preliminaries
    temp = np.arange(n, dtype=int)
    Index_store_x = np.zeros(n, dtype=int)
    Index_store_x[Ix] = temp
    Index_store_y = np.zeros(n, dtype=int)
    Index_store_y[Iy] = temp
    # cdot
    cdot = np.sum(Pc)
    # Syjc
    Iy = Iy.astype(int)
    sumtemp = np.cumsum(Pc[Iy])-Pc[Iy]
    sumtemp = sumtemp[Index_store_y]
    # sumc
    sumc = np.cumsum(Pc)-Pc;
    ## Final sum
    Gamma = np.zeros((n))
    # dyadupdate(Iy, Pc)
    for i in xrange(n):
        Ly   = Py[0:i]
        Lc   = Pc[0:i]
        index = np.where(Ly < Py[i])
        Gamma[i] = np.sum(Lc[index])
        #Gamma[i] = np.sum([Pc[j] for j in xrange(0,i) if Py[j] < Py[i]])
    G = cdot - Pc - 2*sumtemp - 2*sumc + 4*Gamma
    return G[Index_store_x]
###########################################################################
# Fast Distance Covariance
def Fast_Distance_Covariance(X, Y):
    n = X.shape[0]
    # Declare the arrays
    P1 = np.zeros(n)
    P2 = np.zeros(n)
    alphax1 = np.zeros(n)
    alphay1 = np.zeros(n)
    betax1 = np.zeros(n)
    betay1 = np.zeros(n)
    # calculate the sum and finally get the covariance
    # Sum --1
    ## Calculations for the faster algorithm
    temp = np.arange(n)
    # # We got xdot and ydot, this makes sense also
    xdot = np.sum(X)
    ydot = np.sum(Y)
    # Sort the array and get indices
    Ix0 = np.argsort(X)
    Iy0 = np.argsort(Y)
    ## Let us calculate the several parameters associated with calculating alphas and betas
    vx  = np.sort(X)
    vy  = np.sort(Y)
    sx = np.cumsum(vx)
    sy = np.cumsum(vy)
    Ix = np.zeros(n, dtype=int)
    Iy = np.zeros(n, dtype=int)
    Ix[Ix0] = temp
    Iy[Iy0] = temp
    # alpha values
    alphax  = Ix
    alphay  = Iy
    # Beta values
    betax = sx - vx
    betay = sy - vy
    betax = betax[Ix]
    betay = betay[Iy]
    # # adotdot
    adotdot = 2*np.dot(alphax,X) - 2* sum(betax)
    # # bdotdot
    bdotdot = 2*np.dot(alphay,Y) - 2* sum(betay)
    # Get all the gamma's
    ## print "gamma_1"
    gamma_1  = partialsum(X, Y,np.ones((n)) )
    ## print "gamma_x"
    gamma_x  = partialsum(X,Y, X)
    ## print "gamma_y"
    gamma_y  = partialsum(X,Y, Y)
    ## print "gamma_xy"
    gamma_xy = partialsum(X,Y, X*Y)
    sum1 = np.sum(X*Y*gamma_1)
    sum2 = np.sum(gamma_xy)
    sum3 = np.sum(X*gamma_y)
    sum4 = np.sum(Y*gamma_x)
    P1 = (sum1+sum2-sum3-sum4)
    P2 = (xdot + (((2*alphax)-n)*X - 2*betax)) * (ydot + (((2*alphay)-n)*Y - 2*betay))
    P3 =  adotdot*bdotdot
    P11 = (1/float(n*(n-3)))*np.sum(P1)
    P22 = (2/float(n*(n-2)*(n-3)))*np.sum(P2)
    P33 = (1/float(n*(n-1)*(n-2)*(n-3)))*(P3)
    return (P11-P22+P33)
###########################################################################
 ## Code a fast algorithm for calculating distance covariance..
def Dist_corr(X):
    n  = X.shape[0];
    m  = X.shape[1];
    C  = np.zeros((m,m));
    for i in range(m):
        for j in range(m):
            x = X[:,i]
            y = X[:,j]
            nu_xy = Fast_Distance_Covariance(x, y)
            nu_xx = Fast_Distance_Covariance(x, x)
            nu_yy = Fast_Distance_Covariance(y, y)
            if nu_xy*nu_xx*nu_yy < 1e-5:
                C[i][j] = 0.0
            else:
                C[i][j] = sqrt(nu_xy)/float(sqrt(sqrt(nu_xx)*sqrt(nu_yy)))
    return C
###########################################################################
def Dist_Corr_Pooled(Data, batch_size):
    rng = np.random.RandomState(0)
    m = Data.shape[0]
    n = Data.shape[1]
    C = np.zeros((n, n))
    for i in range(m/batch_size):
        #print "batch--", i
        idx = rng.randint(len(Data), size=batch_size)
        P = Data[idx, :]
        for i in xrange(n):
            for j in xrange(n):
                x = P[:,i]
                y = P[:,j]
                dcoryy = Fast_Distance_Covariance(y,y)
                # print dcoryy
                # print sqrt(dcoryy)
                dcorxx = Fast_Distance_Covariance(x,x)
                # print sqrt(dcoryy)
                dcorxy = Fast_Distance_Covariance(x,y)
                # print sqrt(dcorxy)
                if dcorxy*dcorxx*dcoryy <1e-05:
                    C[i][j] = C[i][j]+0.0
                else:
                    C[i][j] = C[i][j]+(float(batch_size-1)*(sqrt(dcorxy)/float(sqrt(sqrt(dcorxx)*sqrt(dcoryy)))))
    return  (C/float(m-(m/(batch_size))))
##############################################################################
def Pearson_Corr(Data, batch_size):
    rng = np.random.RandomState(0)
    m = Data.shape[0]
    n = Data.shape[1]
    C = np.zeros((n, n))
    for i in range(m/batch_size):
        idx = rng.randint(len(Data), size=batch_size)
        P = Data[idx, :]
        for i in xrange(n):
            for j in xrange(n):
                x = P[:,i]
                y = P[:,j]
                C[i][j] = C[i][j]+(float(batch_size-1)*pcc(x, y))
    return  (C/float(m-(m/(batch_size))))
