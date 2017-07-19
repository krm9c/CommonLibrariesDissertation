####################################################################
# Big Data Analysis
# Library  2.0 version
####################################################################

import random
import os,sys
from   math                            import *
import numpy                               as np
import warnings
from sklearn import preprocessing
from Library_Paper_one                 import traditional_MTS
from itertools import izip
import time

from scipy.spatial.distance  import pdist, squareform
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
                    self.Transformed = None;
                    self.par_trans = []
                    self.e_val = None
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
            # Declare Temporary variables
            temp  = []
            group_list = []
            node_count = -1
            if (len(level_list))==1:
                return temp;
            vargroup=0
            Gsize=gsize;
            g=0;
            while (node_count < len(level_list)-1):
                g=0;
                group_list=[]
                while (g < Gsize) and (node_count < len(level_list)-1):
                    node_count=node_count+1;
                    group_list.append(level_list[node_count].number)
                    g=g+1;

                if ((len(level_list)-node_count-1)==1):
                    group_list.append(level_list[len(level_list)-1].number)
                    node_count = node_count+1;
                temp.append(group_list);
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
        def tree_generate(self, row, column, gsize, output_dimension):
                curr_level_list=[]
                prev_level_list=[]
                tree=level()
                # For all the leaf nodes in the tree, let us initialize the connection as zero
                for i in xrange(0,column):
                    prev_level_list.append(0)
                flag = 0
                while len(prev_level_list)>=1:
                    curr_level_list= self.generate_level(len(prev_level_list),prev_level_list, row)
                    prev_level_list = []
                    prev_level_list= self.generate_group(curr_level_list,gsize)

                    if flag ==1:
                        tree.tree_levels.append(curr_level_list)
                        break

                    if len(prev_level_list) <= output_dimension:
                        s = [item for sublist in prev_level_list for item in sublist ]
                        prev_level_list =[]
                        prev_level_list.append(s)
                        flag =  1
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
                if level_flag is not 2:
                    if Paper =='paper_1':
                        dist_Corr = Pearson_Corr(D_scaled)
                    else:
                        # dist_Corr = Dist_Corr(D_scaled)
                        dist_Corr = dependence_calculation(D_scaled)
                    from scipy import linalg as LA


                    e_vals, e_vecs = LA.eig(dist_Corr)
                    temp_sum = 0;
                    temp_number = 0;

                    arg_sort  = e_vals.argsort()
                    s_eigvals = e_vals[arg_sort]
                    s_e_vecs  = e_vecs[arg_sort,:]
                    D_scaled = D_scaled[:,arg_sort]
                    s_eigvals = np.divide(s_eigvals, np.sum(s_eigvals))

                    for eigen in s_eigvals:
                        temp_sum = temp_sum + eigen
                        temp_number = temp_number+1;
                        if temp_sum > 0.95:
                            break;

                    e_vecs = s_e_vecs[0:temp_number,:]
                    eigenv = s_eigvals[0:temp_number]
                    Temp_proj = [];
                    for pc in e_vecs:
                        Temp_proj.append( np.array( [ np.dot(D_untransformed[p,:], pc) for p in xrange(D_scaled.shape[0]) ] ) )
                    Transform_array =  np.transpose(np.array(Temp_proj))
                    print(" The dimension of the arrray", Transform_array.shape)
                    print(" The dimension of the transformer", e_vecs.shape)


                    # from scipy import linalg as LA
                    # e_vals, e_vecs = LA.eig(dist_Corr)
                    # eig_pairs = [(np.abs(e_vals[i]), e_vecs[:,i]) for i in xrange(len(e_vals))]
                    # eig_pairs.sort(key=lambda x: x[0], reverse=True)
                    # pc = eig_pairs[0][1].reshape(column_test,1)
                    # Transform_array= np.array([np.dot(D_untransformed[i,:], pc) for i in xrange(row_test)])

                    return Transform_array, e_vecs, temp_scalar
                elif level_flag == 2:
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
                if (i == (len(Tree.tree_levels))-1):
                    level_flag   = 2;
                else:
                    level_flag = 213;
                # Go throught the groups now
                for group in Tree.tree_levels[i]:
                    group_connectivity=group.connected_nodes
                    temp = group_connectivity[0]
                    D_untransformed = np.zeros((group.Transformed.shape[0] , len(temp)))
                    var=0
                    for element in group_connectivity:
                        for e in element:
                            D_untransformed[:,var]=(Tree.tree_levels[i-1][e].Transformed[:,0]).reshape(-1)
                            var=var+1
                    print("The shape of the array that has been created", D_untransformed.shape)
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
def initialize_calculation(T, Data, gsize, par_train, output_dimension):
    # Part -- 1, Generation of the tree parameters
    row    = Data.shape[0];
    column = Data.shape[1];
    if par_train == 0 :
        # Generate the tree and groups
        T =  level()
        T =  T.tree_generate(row, column,  gsize, output_dimension)
    # Part -- 2, Calculate parameters at every level and store.
    i = 0
    T.resize(row)
    for element in T.tree_levels[0]:
        if i < column:
            element.Transformed[:,0]  = Data[:,i]
            i = i+1
    Data_reduced = T.traverse_tree(par_train);
    return Data_reduced, T

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
