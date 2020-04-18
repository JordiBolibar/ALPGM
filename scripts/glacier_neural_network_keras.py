# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:42:51 2019

@author: Jordi Bolibar

Glacier SMB Artificial Neural Network

"""
import os
#from itertools import cycle
#import seaborn as sns
#from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import copy
import numpy as np
#import seaborn as sns
import math
from numpy import genfromtxt
from pathlib import Path
import shutil
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.class_weight import compute_sample_weight
#from sklearn.neighbors import KernelDensity

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
#from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
from keras import backend as K
from keras.layers import GaussianNoise
from keras.models import load_model
from bisect import bisect
from random import random
#import tensorflow as tf
#from tensorflow.keras.backend import set_session
#from keras import regularizers

#from itertools import combinations 
from scipy.stats import gaussian_kde

### Force CPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

workspace = Path(os.getcwd()).parent 
root = str(workspace.parent) + '\\'
workspace = str(workspace) + '\\'
path_smb = workspace + 'glacier_data\\smb\\'
#SMB_raw_o = genfromtxt(path_smb + 'SMB_raw_extended.csv', delimiter=';', dtype=float)
SMB_raw = genfromtxt(path_smb + 'SMB_raw_temporal.csv', delimiter=';', dtype=float)
path_ann_LOGO = path_smb + 'ANN\\LOGO\\'
path_ann_LOYO = path_smb + 'ANN\\LOYO\\'
path_ann_LSYGO = path_smb + 'ANN\\LSYGO\\'
path_ann_LSYGO_past = path_smb + 'ANN\\LSYGO_past\\'


######################################
#  Training with or without weights  #
w_weights = False
#cross_validation = "LOGO"
#cross_validation = "LOYO"
cross_validation = "LSYGO"
#cross_validation = "LSYGO_past"
#######  Flag to switch between training with a         ###############
#######  single group of glaciers or cross-validation   ###############
training = False
# Train only the full model without training CV models
final_model_only = False
# Activate the ensemble modelling approach
final_model = False
# Only re-calculate fold performances based on previously trained models
recalculate_performance = False
########################################

if(cross_validation == 'LOGO'):
    path_ann = path_ann_LOGO
    path_cv_ann = path_ann + 'CV\\'
elif(cross_validation == 'LOYO'):
    path_ann = path_ann_LOYO
    path_cv_ann = path_ann + 'CV\\'
elif(cross_validation == 'LSYGO'):
    path_ann = path_ann_LSYGO
    path_cv_ann = path_ann + 'CV\\'
elif(cross_validation == 'LSYGO_past'):
    path_ann = path_ann_LSYGO_past
    path_cv_ann = path_ann + 'CV\\'


#############################################################################

def weighted_choice(values, weights):
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random() * total
    i = bisect(cum_weights, x)
    return values[i]

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def r2_keras_loss(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return SS_res/(SS_tot + K.epsilon())

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def create_logo_model(n_features, final):
    model = Sequential()
    
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    
    if(final):
        # Hidden layers
        model.add(Dense(60, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
    #        
        model.add(Dense(50, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(30, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
    else:
           # Hidden layers
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.2))
    #        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1)) 
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.05))
        
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.01))
    
    # Output layer
    model.add(Dense(1))
    
    ##### Optimizers  #######
#    optimizer = optimizers.rmsprop(lr=0.0002)
#    optimizer = optimizers.rmsprop(lr=0.0005)
    optimizer = optimizers.rmsprop(lr=0.001)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[r2_keras])
    
    return model

#############################
    
def create_loyo_model(n_features):
    model = Sequential()
    
    print("n_features:" + str(n_features))
    
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    
    # Hidden layers
    model.add(Dense(40, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.3))
    
    model.add(Dense(20, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.2))
        
    model.add(Dense(10, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.1))
    
    model.add(Dense(5, kernel_initializer='he_normal'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.1))
    
    # Output layer
    model.add(Dense(1))
    
    ##### Optimizers  #######
#    optimizer = optimizers.rmsprop(lr=0.05)
    optimizer = optimizers.rmsprop(lr=0.02)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[root_mean_squared_error])
#    model.compile(optimizer = optimizer, loss=r2_keras_loss, metrics=[r2_keras])

    
    return model


def create_lsygo_model(n_features, final):
    model = Sequential()
    
    print("n_features:" + str(n_features))
    
    # Input layer
    model.add(Dense(n_features, input_shape=(n_features,), kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    
    # Hidden layers
    if(final):
        # Hidden layers
        model.add(Dense(80, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
    #        
        model.add(Dense(60, kernel_initializer='he_normal'))
        model.add(BatchNormalization())  
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(50, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(30, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        
    else:
        model.add(GaussianNoise(0.1))
        model.add(Dense(40, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.35))
    #    model.add(Dropout(0.1))
        
        model.add(Dense(20, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.25))
    #    model.add(Dropout(0.1))
        
        model.add(Dense(10, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))
    #    model.add(Dropout(0.05))
    
        model.add(Dense(5, kernel_initializer='he_normal'))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.1))
    #    model.add(Dropout(0.05))
    
    # Output layer
    model.add(Dense(1))
    
    
    ##### Optimizers  #######
#    optimizer = optimizers.rmsprop(lr=0.004)
    optimizer = optimizers.rmsprop(lr=0.002)
#    optimizer = optimizers.rmsprop(lr=0.001)
#    optimizer = optimizers.rmsprop(lr=0.0005)
#    optimizer = optimizers.adam(lr=0.0005)
#    optimizer = optimizers.adam(lr=0.0003)
#    optimizer = optimizers.rmsprop(lr=0.0007)
#    optimizer = optimizers.rmsprop(lr=0.0008)
    
    # Compilation
    model.compile(optimizer = optimizer, loss=root_mean_squared_error, metrics=[root_mean_squared_error])
#    model.compile(optimizer = optimizer, loss=r2_keras_loss, metrics=[r2_keras])

    
    return model

###########################################################################

# Read features and ground truth
    
with open(root+'X_nn_extended.txt', 'rb') as x_f:
    X = np.load(x_f)
with open(root+'y_extended.txt', 'rb') as y_f:
    y_o = np.load(y_f)

y = y_o.flatten()

print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))


#######################################################

# Leave One Group Out indexes
groups = []
group_n = 1
group_i = 1

# Multi-glacier folds
#for glacier in SMB_raw:
#    groups = np.concatenate((groups, np.repeat(group_n, np.count_nonzero(~np.isnan(glacier)))), axis=None)
#    if(group_i > 3):
#        group_n = group_n+1
#        group_i = 1
#    group_i = group_i+1

# Single-glacier folds
#for glacier in SMB_raw:
#    groups = np.concatenate((groups, np.repeat(group_n, np.count_nonzero(~np.isnan(glacier)))), axis=None)
#    group_n = group_n+1

for glacier in SMB_raw:
    groups = np.concatenate((groups, np.repeat(group_n, glacier.size)), axis=None)
    group_n = group_n+1

    
# Single-year folds
year_groups = []
year_n = 1
current_group = 1
#for glacier_group in groups:
glacier_count = 0
for glacier in SMB_raw:
    for year in range(1, 58):
        year_groups.append(year)
    
    glacier_count = glacier_count+1
        
        
year_groups = np.asarray(year_groups)  
year_groups_n = year_groups.max()  

groups = np.asarray(groups)  

lsygo_test_matrixes, lsygo_train_matrixes = [],[]

# LSYGO folds
year_idx = 0
glacier_idx = 0
# TODO: see if block seed
#np.random.seed(10)
n_folds = 100
#n_folds = 15
random_years = np.random.randint(0, 57, n_folds*4) # Random year idxs
random_glaciers = np.random.randint(0, 32, n_folds*4) # Random glacier indexes

y_not_nan = np.isfinite(y_o)
y_is_nan = np.isnan(y_o)

#p_weights = compute_sample_weight(class_weight='balanced', y=y_o[y_not_nan])
p_weights = np.ones(y_o[y_not_nan].shape)

# We balance the positive SMB in the sample
#diff = (y_o[y_not_nan].max() - y_o[y_not_nan].min())/y_o[y_not_nan].mean()

idx, i, j = 0, 0, 0
for i in range(0, y_o.shape[0]):
    for j in range(0, y_o.shape[1]):
        if(np.isfinite(y_o[i,j])):
            if(j < 27):
                p_weights[idx] = p_weights[idx] + 1/2 # Add weight for the 1959-1967
#                if(i == y_o.shape[0]-1):
#                    p_weights[idx] = p_weights[idx] + 1/3 # Add weight for the 1959-1967
#                else:
#                    p_weights[idx] = p_weights[idx] + 1/2 # Add weight for the 1959-1967
                # Accounting for 33% of the time period
#            print("counter = " + str(idx))
            idx=idx+1

#p_weights = np.where(y_o[y_not_nan] > 0, p_weights + p_weights.max()/4, p_weights)
            
for fold in range(1, n_folds+1):
    test_matrix, train_matrix = np.zeros((32, 57), dtype=np.int8), np.ones((32, 57), dtype=np.int8)
    
    choice = weighted_choice(np.asarray(range(0,1048)), p_weights)
    
    idx, i, j = 0, 0, 0
    for i in range(0, y_o.shape[0]):
        for j in range(0, y_o.shape[1]):
            if(np.isfinite(y_o[i,j])):
                if(choice == idx):
                    glacier_idx = i
                    year_idx = j
    #            print("counter = " + str(idx))
                idx=idx+1
    
    print("\nChosen glacier: " + str(glacier_idx))
    print("Chosen year: " + str(year_idx))
    print("SMB: " + str(y_o[glacier_idx,year_idx]))
    
    # Make sure Sarennes and Saint Sorlin appear in the folds
#    if(fold >= 1 and fold < 5):
#        test_matrix[-1, :] = 1
#        # Fill train matrix
#        train_matrix[-1, :] = 0
#    if(fold >= 1 and fold < 11):
##        test_matrix[2, :] = 1
#        test_matrix[:, :27] = 1
#        # Fill train matrix
##        train_matrix[2, :] = 0
#        train_matrix[:, :27] = 0
##    else:
    # Fill test matrix
    test_matrix[glacier_idx, :] = 1
    # Fill train matrix
    train_matrix[glacier_idx, :] = 0
#        
    
    # Fill test matrix
#    test_matrix[random_glaciers[glacier_idx], :] = 1
#    test_matrix[random_glaciers[glacier_idx+1], :] = 1
#    if(not(fold >= 1 and fold < 11)):
    test_matrix[:, year_idx] = 1
#    test_matrix[:, random_years[year_idx+1]] = 1
    
#    # 1
#    test_matrix[random_glaciers[glacier_idx], random_years[year_idx]] = 1
#    test_matrix[random_glaciers[glacier_idx], random_years[year_idx+1]] = 1
#    test_matrix[random_glaciers[glacier_idx], random_years[year_idx+2]] = 1
#    test_matrix[random_glaciers[glacier_idx], random_years[year_idx+3]] = 1
#    
#    # 2
#    test_matrix[random_glaciers[glacier_idx+1], random_years[year_idx]] = 1
#    test_matrix[random_glaciers[glacier_idx+1], random_years[year_idx+1]] = 1
#    test_matrix[random_glaciers[glacier_idx+1], random_years[year_idx+2]] = 1
#    test_matrix[random_glaciers[glacier_idx+1], random_years[year_idx+3]] = 1
#    
#    # 3
#    test_matrix[random_glaciers[glacier_idx+2], random_years[year_idx]] = 1
#    test_matrix[random_glaciers[glacier_idx+2], random_years[year_idx+1]] = 1
#    test_matrix[random_glaciers[glacier_idx+2], random_years[year_idx+2]] = 1
#    test_matrix[random_glaciers[glacier_idx+2], random_years[year_idx+3]] = 1
#    
#    # 4
#    test_matrix[random_glaciers[glacier_idx+3], random_years[year_idx]] = 1
#    test_matrix[random_glaciers[glacier_idx+3], random_years[year_idx+1]] = 1
#    test_matrix[random_glaciers[glacier_idx+3], random_years[year_idx+2]] = 1
#    test_matrix[random_glaciers[glacier_idx+3], random_years[year_idx+3]] = 1
    
    lsygo_test_matrixes.append(test_matrix)
    
    # Fill train matrix
#    train_matrix[random_glaciers[glacier_idx], :] = 0
#    train_matrix[random_glaciers[glacier_idx+1], :] = 0
#    train_matrix[random_glaciers[glacier_idx+2], :] = 0
#    train_matrix[random_glaciers[glacier_idx+3], :] = 0
#    if(not(fold >= 1 and fold < 11)):
    train_matrix[:, year_idx] = 0
#    train_matrix[:, random_years[year_idx+1]] = 0
#    train_matrix[:, random_years[year_idx+2]] = 0
#    train_matrix[:, random_years[year_idx+3]] = 0
    
    lsygo_train_matrixes.append(train_matrix)
    
    year_idx = year_idx+4
    glacier_idx = glacier_idx+4

# We capture the mask from the SMB data to remove the nan gaps  
finite_mask = np.isfinite(y)

X = X[finite_mask,:]
y = y[finite_mask]


groups = groups[finite_mask]

year_groups = year_groups[finite_mask] - 25

# Remove negative fold indexes and set them to 0 (not used in CV)
year_groups = np.where(year_groups < 0, 0, year_groups)

# We flatten and filter the nan values
lsygo_test_int_folds, lsygo_train_int_folds = [],[]
# Filter LSYGO folds
for test_fold, train_fold in zip(lsygo_test_matrixes, lsygo_train_matrixes):
    lsygo_test_int_folds.append(test_fold.flatten()[finite_mask])
    lsygo_train_int_folds.append(train_fold.flatten()[finite_mask])

# From int to boolean
lsygo_test_folds = np.array(lsygo_test_int_folds, dtype=bool)
lsygo_train_folds = np.array(lsygo_train_int_folds, dtype=bool)

logo = LeaveOneGroupOut()
loyo = LeaveOneGroupOut()

# Leave-One-Glacier-Out
logo_splits = logo.split(X, groups=groups)

# Leave-One-Year-Out
loyo_splits = loyo.split(X, groups=year_groups)

# 1967-1985 validation
past_test_matrix, past_train_matrix = np.zeros((32, 57), dtype=np.int8), np.ones((32, 57), dtype=np.int8)


#past_test_matrix[:,:26] = 1
#past_train_matrix[:,:26] = 0

past_test_int_matrix = past_test_matrix.flatten()[finite_mask]
past_test_matrix = np.array(past_test_int_matrix, dtype=bool)
past_train_int_matrix = past_train_matrix.flatten()[finite_mask]
past_train_matrix = np.array(past_train_int_matrix, dtype=bool)

past_glaciers = np.array([1, 3, 31, 32])
past_folds_test, past_folds_train = [],[]
past_years = []
for past_glacier in past_glaciers:
    past_glacier_test_idx = np.intersect1d(np.where(groups == past_glacier)[0], np.where(year_groups == 0)[0])
    past_years = np.concatenate((past_years, past_glacier_test_idx), axis=None)
    current_past_fold_test = copy.deepcopy(past_test_matrix)
    current_past_fold_test[past_glacier_test_idx] = True
    past_folds_test.append(current_past_fold_test)
    current_past_fold_train = copy.deepcopy(past_train_matrix)
    past_glacier_train_idx = np.where(groups == past_glacier)[0]
    current_past_fold_train[past_glacier_train_idx] = False
    past_folds_train.append(current_past_fold_train)

past_folds_test = np.asarray(past_folds_test)
past_folds_train = np.asarray(past_folds_train)

# Indexes with all years before 1984
past_years = np.asarray(past_years)

#import pdb; pdb.set_trace()

#######################
glacier_subset_idx = 2
#######################

if(cross_validation == 'LOGO'):
# LOGO
    test_idx = np.where(groups == glacier_subset_idx)
    train_idx = np.where(groups != glacier_subset_idx)
elif(cross_validation == 'LOYO'):
# LOYO
    test_idx = np.where(year_groups == glacier_subset_idx)
    train_idx = np.where(year_groups != glacier_subset_idx)
elif(cross_validation == "LSYGO" or cross_validation == "LSYGO_past"):
    test_idx = lsygo_test_folds[glacier_subset_idx]
    train_idx = lsygo_train_folds[glacier_subset_idx]

#print("test_idx: " + str(test_idx))
#print("train_idx: " + str(train_idx))
    
#import pdb; pdb.set_trace()    
    
y_train = y[train_idx]

# Automatic weights
weights = compute_sample_weight(class_weight='balanced', y=y_train)

#######################################################################

weights_file = path_smb + '\\nn_weights\\model_weights'
extreme_weights_file = path_smb + '\\nn_weights\\model_extreme_weights'



print("\nTraining Neural Network...")   

if(training):
    
    #############################
    
    n_features = X[0,:].size
    
    if(cross_validation == 'LOYO'):
        model = create_loyo_model(n_features)
    elif(cross_validation == 'LOGO'):
        model = create_logo_model(n_features, final=False)
    elif(cross_validation == 'LSYGO' or cross_validation == 'LSYGO_past'):
        model = create_lsygo_model(n_features, final=False)
        
#    train_idx = np.asarray(train_idx)
#    test_idx = np.asarray(test_idx)
    
    
    X_test = X[test_idx]
    X_train = X[train_idx]
    
    y_test = y[test_idx]
    y_train = y[train_idx]
    
    print("test_idx: " + str(test_idx))
    print("train_idx: " + str(train_idx))
    

##############   TEST ON ALL DATA   ########################################################## 
#    weights_all = compute_sample_weight(class_weight='balanced', y=y)
#    history = model.fit(X, y, epochs=800, batch_size = 34, sample_weight=weights_all, verbose=1)
#    
#    SMB_nn = model.predict(X)
#    
#    plt.title("Glacier Neural Network")
#    plt.ylabel('SMB modeled by NN')
#    plt.xlabel('SMB from remote sensing (ground truth)')
#    plt.scatter(y, SMB_nn, alpha=0.7)
#    lineStart = y.min() 
#    lineEnd = y.max()  
#    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
#    plt.xlim(lineStart, lineEnd)
#    plt.ylim(lineStart, lineEnd)
#    plt.legend()
#    plt.show()
###############################################################################################
    
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=1000)
    mc = ModelCheckpoint(path_ann + 'best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=2000, batch_size = 32, verbose=1, callbacks=[es, mc])
    
    # load the saved model
    best_model = load_model(path_ann  + 'best_model.h5', custom_objects={"r2_keras": r2_keras, "r2_keras_loss": r2_keras_loss, "root_mean_squared_error": root_mean_squared_error})
    
    score = best_model.evaluate(X_test, y_test)
    print(best_model.metrics_names)
    print(score)
    
    SMB_nn = best_model.predict(X_test)
    SMB_nn = SMB_nn[:,0]
    finite_mask = np.isfinite(SMB_nn)
    SMB_nn = SMB_nn[finite_mask]
    
    y_test = y_test[finite_mask]
    
#    print("KernelDensity(kernel='gaussian', bandwidth=0.75).fit(y_test): " + str(KernelDensity(kernel='gaussian', bandwidth=0.75).fit(y_test)))
#    
#    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(y_test)
#    test_weights = 1/np.exp(kde.score_samples(y_test))
    
#    print("n\Manually computed r2: " + str(r2_score(y_test, SMB_nn,test_weights)))
    
    # list all data in history
    print(history.history.keys())
#    # summarize history for accuracy
#    plt.plot(history.history['r2_keras'])
#    plt.plot(history.history['val_r2_keras'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.ylim(0, 1)
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
#    plt.ylim(0.2, 1.5)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
###############################################################################    

# Training for all the folds    
else:
    
    # Set training sample weights and epochs
    weights_full = compute_sample_weight(class_weight='balanced', y=y)
    
    # TODO: check effect of positive SMB weights
#    positive_smb_weights = np.where(y > 0, 100, 1)
    
#    past_int_years = np.array(past_years, dtype='int')
#    positive_smb_weights = np.ones(y.shape)
#    positive_smb_weights[past_int_years] = 10
##    positive_smb_weights = np.where(y == y[past_int_years], 20, 1)
#    
##    import pdb; pdb.set_trace()
#    
#    weights_full = positive_smb_weights
    
    SMB_nn_all = []
    RMSE_nn_all, RMSE_nn_all_w = [],[]
    r2_nn_all, r2_nn_all_w = [],[]
    SMB_ref_all = []
    fold_idx = 1
    
    n_features = X[0,:].size
    
    if(cross_validation == "LOGO"):
        splits = logo_splits
        full_model = create_logo_model(n_features, final=False)
        fold_filter = -1
        n_epochs = 2500
    elif(cross_validation == "LOYO"):
        splits = loyo_splits
        full_model = create_loyo_model(n_features)
        fold_filter = 0
        n_epochs = 2000
    elif(cross_validation == 'LSYGO'):
        splits = zip(lsygo_train_folds, lsygo_test_folds)
        full_model = create_lsygo_model(n_features, final=False)
        fold_filter = -1
#        n_epochs = 2000
        n_epochs = 3000
#        n_epochs = 800
    elif(cross_validation == 'LSYGO_past'):
        splits = zip(past_folds_train, past_folds_test)
        full_model = create_lsygo_model(n_features, final=False)
        fold_filter = -1
        n_epochs = 2000
        
    if(not final_model_only):
        
        # TODO: remove after tests
#        fold_filter = 27
#        fold_filter = 50
        fold_count = 0
        average_overall_score = []
        
        for train_idx, test_idx in splits:
            # We skip the first dummy fold with the "extra years"
            if(fold_count > fold_filter):
            
                print("\nFold " + str(fold_idx))
                
                print("train_idx: " + str(train_idx) + "  -   " + str(np.count_nonzero(train_idx)) + " values")
                print("test_idx: " + str(test_idx) + "  -   " + str(np.count_nonzero(test_idx)) + " values")
                
    #            import pdb; pdb.set_trace()
        #        
                X_test = X[test_idx]
                y_test = y[test_idx]
                X_train = X[train_idx]
                y_train = y[train_idx]
                
                print("\nTesting SMB values: " + str(y_test))
                
                # Recalculate performance of previously trained models
                if(recalculate_performance):
                    SMB_model = load_model(path_cv_ann + 'glacier_' + str(fold_count+1) + '_model.h5', custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error}, compile=False)
                    
                    SMB_nn = SMB_model.predict(X_test, batch_size = 32)
                    SMB_nn_all = np.concatenate((SMB_nn_all, SMB_nn), axis=None)
                    
                # Train new models
                else:
                
        #            if(cross_validation == 'LSYGO'):
        #                print("test_idx: " + str(np.where(test_idx == True)))
                        
        #                print("\nglaciers test: " + str(groups[test_idx]))
        #                print("years test: " + str(year_groups[test_idx]))
        #                
        #                print("\nglaciers train: " + str(groups[train_idx]))
        #                print("years train: " + str(year_groups[train_idx]))
        #                print("train_idx: " + str(np.where(train_idx == True)))
                        
        #                import pdb; pdb.set_trace()
                    
                    weights_train = weights_full[train_idx]
                    weights_test = weights_full[test_idx]
                    
                    if(cross_validation == "LOGO"):
                        model = create_logo_model(n_features, final=False)
                        file_name = 'best_model_LOGO.h5'
                    elif(cross_validation == "LOYO"):
                        model = create_loyo_model(n_features)
                        file_name = 'best_model_LOYO.h5'
                    elif(cross_validation == "LSYGO"):
                        model = create_lsygo_model(n_features, final=False)
                        file_name = 'best_model_LSYGO.h5'
                    elif(cross_validation == "LSYGO_past"):
                        model = create_lsygo_model(n_features, final=False)
                        file_name = 'best_model_LSYGO_past.h5'
                    
                    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=1000)
                    mc = ModelCheckpoint(path_ann + str(file_name), monitor='val_loss', mode='min', save_best_only=True, verbose=1)
            
                    if(w_weights):
                        # Training with weights
                        print("\nMax weight " + str(weights_full.max()) + " for " + str(np.where(weights_full == 50)[0].size) + " values")
                        #TODO: decide if use MC or not
                        history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=n_epochs, batch_size = 32, callbacks=[es, mc], sample_weight = weights_train, verbose=1)
                        
                        SMB_fold_pred = model.predict(X_test)
                        
    #                    print("\nReference SMB: " + str(y_test))
    #                    print("\nPredicted SMB: " + str(SMB_fold_pred))
                        
    #                    # summarize history for loss
    #                    plt.plot(history.history['loss'])
    #                    plt.plot(history.history['val_loss'])
    #                    plt.title('model loss')
    #                    plt.ylabel('loss')
    #                    plt.ylim(0, 2)
    #                    plt.xlabel('epoch')
    #                    plt.legend(['train', 'test'], loc='upper left')
    #                    plt.show()
                    else:
                        # Training without weights
                        history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=n_epochs, batch_size = 32, callbacks=[es, mc], verbose=1)
                    
                    # load the saved model
                    best_model = load_model(path_ann  + str(file_name), custom_objects={"r2_keras": r2_keras, "r2_keras_loss": r2_keras_loss, "root_mean_squared_error": root_mean_squared_error})
                    
                    score = best_model.evaluate(X_test, y_test)
                    print(best_model.metrics_names)
                    print(score)
                    average_overall_score.append(score[0])
                    print("\nAverage overall score so far: " + str(np.average(average_overall_score)))
                    
                    SMB_nn = best_model.predict(X_test, batch_size = 32)
                    SMB_nn_all = np.concatenate((SMB_nn_all, SMB_nn), axis=None)
                    
                    #### We store the CV model
                    if not os.path.exists(path_cv_ann):
                        os.makedirs(path_cv_ann)
                    ##### We save the model in HDF5 format
                    best_model.save(path_cv_ann + 'glacier_' + str(fold_idx) + '_model.h5')
                    print("CV model saved to disk")
                
                # Compute the performance of the current model
                
                print("Manual r2: " + str(r2_score(y_test, SMB_nn)))
                print("Manual RMSE: " + str(math.sqrt(mean_squared_error(y_test, SMB_nn))))
                
                if(w_weights):
                    print("Manual r2 w/ weights: " + str(r2_score(y_test, SMB_nn, weights_test)))
                    print("Manual RMSE w/ weights: " + str(math.sqrt(mean_squared_error(y_test, SMB_nn, weights_test))))
                
                
                  # We plot the current fold
        #        plt.scatter(y_test, SMB_nn, alpha=0.7, marker = next(marker), label='Glacier ' + str(glacier_idx))
                
                r2_nn_all = np.concatenate((r2_nn_all, r2_score(y_test, SMB_nn)), axis=None)
                RMSE_nn_all = np.concatenate((RMSE_nn_all, math.sqrt(mean_squared_error(y_test, SMB_nn))), axis=None)
                SMB_ref_all = np.concatenate((SMB_ref_all, y_test), axis=None)
                
                if(w_weights):
                    r2_nn_all_w = np.concatenate((r2_nn_all_w, r2_score(y_test, SMB_nn, weights_test)), axis=None)
                    RMSE_nn_all_w = np.concatenate((RMSE_nn_all_w, math.sqrt(mean_squared_error(y_test, SMB_nn, weights_test))), axis=None)
                
                # Clear tensorflow graph to avoid slowing CPU down
                if(fold_idx != 64):
                    K.clear_session()
                
                fold_idx = fold_idx+1
                
            fold_count = fold_count+1
            
        r2_nn_all = np.asarray(r2_nn_all)
        RMSE_nn_all = np.asarray(RMSE_nn_all)
        if(w_weights):
            r2_nn_all_w = np.asarray(r2_nn_all_w)
            RMSE_nn_all_w = np.asarray(RMSE_nn_all_w)
        
        weights_validation = compute_sample_weight(class_weight='balanced', y=SMB_ref_all)
        
        print("\nScores from averaging folds: ")
        print("\nMean overall r2: " + str(r2_nn_all.mean()))
        print("\nMean overall RMSE: " + str(RMSE_nn_all.mean()))
        if(w_weights):
            print("\nMean overall r2 w/ weights: " + str(r2_nn_all_w.mean()))
            print("\nMean overall RMSE w/ weights: " + str(RMSE_nn_all_w.mean()))
        print("--------------------------")
        
        print("\nScores computed on all values together:")
        print("\nMean overall r2: " + str(r2_score(SMB_ref_all, SMB_nn_all)))
        print("\nMean overall RMSE: " + str(math.sqrt(mean_squared_error(SMB_ref_all, SMB_nn_all))))
        print("\nMean overall MAE: " + str(mean_absolute_error(SMB_ref_all, SMB_nn_all)))
        
        if(w_weights):
            print("\nMean overall r2 w/ weights: " + str(r2_score(SMB_ref_all, SMB_nn_all, weights_validation)))
            print("\nMean overall RMSE w/ weights: " + str(math.sqrt(mean_squared_error(SMB_ref_all, SMB_nn_all, weights_validation))))
            
        print("\nRMSE per fold: " + str(RMSE_nn_all))
        
        print("\nAverage bias: " + str(np.average(SMB_nn_all - SMB_ref_all)))
        
        with open(path_ann + 'RMSE_per_fold.txt', 'wb') as rmse_f: 
            np.save(rmse_f, RMSE_nn_all)
        
#        import pdb; pdb.set_trace()
        
        # Calculate the point density
        xy = np.vstack([SMB_ref_all,SMB_nn_all])
        z = gaussian_kde(xy)(xy)
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        y_plt, ann_plt, z = SMB_ref_all[idx], SMB_nn_all[idx], z[idx]
        
        plt.figure(figsize=(6,6))
        plt.title("Deep learning glacier-wide SMB simulation (1959-1983)", fontsize=16)
        plt.ylabel('SMB modeled by ANN', fontsize=14)
        plt.xlabel('SMB from remote sensing (ground truth)', fontsize=14)
        lineStart = SMB_ref_all.min() 
        lineEnd = SMB_ref_all.max()  
        sc = plt.scatter(y_plt, ann_plt, c=z, s=50)
        cbar = plt.colorbar(sc, label="Kernel density estimation")
        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
        plt.xlim(lineStart, lineEnd)
        plt.ylim(lineStart, lineEnd)
        plt.legend()
        plt.draw()
    
    
    ###############    We train the model with the full dataset and we store it   #######################################
    
    # We create N models in an ensemble approach to be averaged
    if(final_model):
        ensemble_size = 30
        path_ann_ensemble = path_ann + 'ensemble\\'
        average_overall_score, average_pos_rmse, average_neg_rmse, average_bias = [],[],[],[]
        
        # We clear all the previous models to avoid problems loading files
        if(os.path.exists(path_ann_ensemble)):
            shutil.rmtree(path_ann_ensemble)
            os.makedirs(path_ann_ensemble)
        
        for e_idx in range(1, ensemble_size+1):
            file_name = 'best_model_full_' + str(e_idx) + '.h5'
            
            # Create new model to avoid re-training the previous one
            if(cross_validation == "LOGO"):
                full_model = create_logo_model(n_features, final=True)
                n_epochs = 2000
            elif(cross_validation == "LOYO"):
                full_model = create_loyo_model(n_features)
                n_epochs = 2000
            elif(cross_validation == 'LSYGO'):
                full_model = create_lsygo_model(n_features, final=True)
                n_epochs = 3500
            elif(cross_validation == 'LSYGO_past'):
                full_model = create_lsygo_model(n_features, final=True)
                n_epochs = 2000
             
            es = EarlyStopping(monitor='loss', mode='min', min_delta=0.01, patience=1000)
            
            if(w_weights):
                # Training with weights
                print("\nMax weight " + str(weights_full.max()) + " for " + str(np.where(weights_full == 50)[0].size) + " values")
                history = full_model.fit(X, y, epochs=n_epochs, batch_size = 32, sample_weight = weights_full, verbose=1)
                 
            else:
                # Training without weights
                history = full_model.fit(X, y, epochs=n_epochs, batch_size = 32, verbose=1)
                
            full_original_score = full_model.evaluate(X, y, batch_size = 32)
            
            SMB_nn_original = full_model.predict(X, batch_size = 32)
            
            print("\nMean original overall r2: " + str(r2_score(y, SMB_nn_original)))
            print("\nMean original overall RMSE: " + str(math.sqrt(mean_squared_error(y, SMB_nn_original))))
            
            # Extreme values bias check
            y_pos_idx = np.where(y > 0)
            pos_bias = (SMB_nn_original[y_pos_idx] - y[y_pos_idx]).mean()
            pos_rmse = math.sqrt(mean_squared_error(y[y_pos_idx], SMB_nn_original[y_pos_idx]))
            print("\nRMSE for positive SMB values: " + str(pos_rmse))
            
            y_neg_idx = np.where(y < 0)
            neg_bias = (SMB_nn_original[y_neg_idx] - y[y_neg_idx]).mean()
            neg_rmse = math.sqrt(mean_squared_error(y[y_neg_idx], SMB_nn_original[y_neg_idx]))
            print("\nRMSE for negative SMB values: " + str(neg_rmse))
            
            overall_bias = (SMB_nn_original - y).mean()
            
            print("\nFull model score: " + str(full_original_score))
            print("\nFull model bias: " + str(overall_bias))
            
            if(pos_rmse < 0.75 and neg_rmse < 0.75 and np.abs(overall_bias) < 0.1):
                average_overall_score.append(full_original_score[0])
                average_pos_rmse.append(pos_rmse)
                average_neg_rmse.append(neg_rmse)
                average_bias.append(overall_bias)
                print("\nModel added to ensemble")
                
                # Create folder for each ensemble member
                path_e_member = path_ann_ensemble + str(e_idx) + '\\'
                if not os.path.exists(path_e_member):
                    os.makedirs(path_e_member)
                
                #### We store the full model
                ##### We save the model to HDF5
                full_model.save(path_e_member + 'ann_glacier_model.h5')
                print("Full model saved to disk")
                            
                with open(path_e_member + 'SMB_nn.txt', 'wb') as SMB_nn_all_f: 
                    np.save(SMB_nn_all_f, SMB_nn_original)
            
            if(len(average_overall_score) > 0):
                print("\n Overall ensemble performance so far: " + str(np.average(average_overall_score)))
                print("\n Overall ensemble positive SMB RMSE so far: " + str(np.average(average_pos_rmse)))
                print("\n Overall ensemble negative SMB RMSE so far: " + str(np.average(average_neg_rmse)))
                print("\n Overall ensemble average bias so far: " + str(np.average(average_bias)))
            
            # Clear tensorflow graph to avoid slowing CPU down
            K.clear_session()
            
        print("\nEnsemble model members completed")
        
        print("\nSaving model performance..." )
        with open(path_ann_LSYGO + 'overall_average_score.txt', 'wb') as o_a_s_f: 
            np.save(o_a_s_f, np.asarray(average_overall_score))
        with open(path_ann_LSYGO + 'overall_positive_RMSE.txt', 'wb') as o_p_rmse_f: 
            np.save(o_p_rmse_f, np.asarray(average_pos_rmse))
        with open(path_ann_LSYGO + 'overall_negative_RMSE.txt', 'wb') as o_n_rmse_f: 
            np.save(o_n_rmse_f, np.asarray(average_neg_rmse))
        with open(path_ann_LSYGO + 'overall_bias.txt', 'wb') as obias_f: 
            np.save(obias_f, np.asarray(average_bias))
        
        
#    import pdb; pdb.set_trace()
    
    ################################################################################################################
    
#    print("\nSimulated ANN SMB values: " + str(SMB_nn_all))
    
#    hypsometry, bins = np.histogram(RMSE_nn_all_w, bins="auto")
#    plt.xlabel('RMSE')
#    plt.title("Weighted RMSE score distribution")
#    plt.bar(bins[:-1] + np.diff(bins) / 2, hypsometry, np.diff(bins), label='Hypsometry')
#    plt.show()
#    
#    print("\nr2_nn_all_w: " + str(r2_nn_all_w))
#    
#    hypsometry, bins = np.histogram(r2_nn_all_w, bins="auto")
#    plt.xlabel('R2')
#    plt.title("Weighted R2 score distribution")
#    plt.bar(bins[:-1] + np.diff(bins) / 2, hypsometry, np.diff(bins), label='Hypsometry')
#    plt.show()
#    
#    density_nn = gaussian_kde(SMB_nn_all)
#    density = gaussian_kde(y)
#    xs = np.linspace(-4,3,200)
#    density.covariance_factor = lambda : .25
#    density_nn.covariance_factor = lambda : .25
#    density._compute_covariance()
#    density_nn._compute_covariance()
#    plt.title("Neural network SMB density distribution")
#    plt.plot(xs,density(xs), label='Ground truth')
#    plt.plot(xs,density_nn(xs), label='NN simulation')
#    plt.legend()
#    plt.show()
