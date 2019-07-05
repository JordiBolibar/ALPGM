# -*- coding: utf-8 -*-
"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

LINEAR LASSO ERROR MODEL TRAINING

"""

import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from numpy import genfromtxt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

workspace = Path(os.getcwd()).parent 
root = str(workspace.parent) + '\\'
workspace = str(workspace) + '\\'
path_smb = workspace + 'glacier_data\\smb\\'
path_smb_function = path_smb + 'smb_function\\'
path_ann = path_smb + 'ANN\\'
path_smb_errors = 'C:\\Jordi\\PhD\\Estadistica\\SMB_errors_ANN\\'

SMB_raw = genfromtxt(path_smb + 'SMB_raw_extended.csv', delimiter=';', dtype=float)

SMB_lasso = genfromtxt(path_smb_errors + 'SMB_lasso_all.csv', delimiter=';', dtype=float)


# Read features and ground truth
#with open(workspace+'X_nn.txt', 'rb') as x_f:
#    X = np.load(x_f)
with open(root+'X_nn.txt', 'rb') as x_f:
    X = np.load(x_f)
with open(root+'y.txt', 'rb') as y_f:
    y = np.load(y_f)
    
with open(path_ann+'SMB_nn_all.txt', 'rb') as y_f:
    y_ann = np.load(y_f)

###################################################################

subset = np.array([0, 1, 2, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])  
y_error = np.sqrt((y - y_ann)**2)
error_scaler = StandardScaler()

X = X[:,subset]
X_scaled = error_scaler.fit_transform(X)

model_CV_error = LassoCV(cv=90).fit(X_scaled, y_error)

coefs_error = np.absolute(model_CV_error.coef_)

sorted_coef_error_idxs = np.flip(np.argsort(coefs_error))

sorted_coef_error_perc = np.flip(np.sort((coefs_error/np.sum(coefs_error)*100)))

#import pdb; pdb.set_trace()

# We plot the error distribution for the first two variables
# Leave One Group Out indexes
groups = []
group_n = 1

# Single-glacier folds
for glacier in SMB_raw:
    groups = np.concatenate((groups, np.repeat(group_n, np.count_nonzero(~np.isnan(glacier)))), axis=None)
    group_n = group_n+1
    
rmse_glaciers, mae_glaciers, bias_glaciers, lat_glaciers, lon_glaciers = [],[],[],[],[]
rmse_glaciers_lasso, mae_glaciers_lasso, bias_glaciers_lasso = [],[],[]

for idx in range(1, group_n):
    glacier_idxs = np.where(groups == idx)[0]
    
    # Deep learning
    glacier_error = math.sqrt(mean_squared_error(y[glacier_idxs], y_ann[glacier_idxs]))
    glacier_mae = np.mean(abs(y[glacier_idxs] - y_ann[glacier_idxs]))
    glacier_bias = np.mean(y[glacier_idxs] - y_ann[glacier_idxs])
    glacier_lat = X[glacier_idxs[0],5]
    glacier_lon = X[glacier_idxs[0],4]
    rmse_glaciers.append(glacier_error)
    mae_glaciers.append(glacier_mae)
    bias_glaciers.append(glacier_bias)
    lat_glaciers.append(glacier_lat)
    lon_glaciers.append(glacier_lon)
    
    # Lasso
    glacier_error_lasso = math.sqrt(mean_squared_error(y[glacier_idxs], SMB_lasso[glacier_idxs]))
    glacier_mae_lasso = np.mean(abs(y[glacier_idxs] - SMB_lasso[glacier_idxs]))
    glacier_bias_lasso = np.mean(y[glacier_idxs] - SMB_lasso[glacier_idxs])
    glacier_lat = X[glacier_idxs[0],5]
    glacier_lon = X[glacier_idxs[0],4]
    rmse_glaciers_lasso.append(glacier_error_lasso)
    mae_glaciers_lasso.append(glacier_mae_lasso)
    bias_glaciers_lasso.append(glacier_bias_lasso)
    
rmse_glaciers = np.asarray(rmse_glaciers)
mae_glaciers = np.asarray(mae_glaciers)
bias_glaciers = np.asarray(bias_glaciers)
lat_glaciers = np.asarray(lat_glaciers)
lon_glaciers = np.asarray(lon_glaciers)

rmse_glaciers_lasso = np.asarray(rmse_glaciers_lasso)
mae_glaciers_lasso = np.asarray(mae_glaciers_lasso)
bias_glaciers_lasso = np.asarray(bias_glaciers_lasso)

np.savetxt(path_smb_errors + 'smb_ann_MAE.csv', mae_glaciers, delimiter=";", fmt="%.7f")
np.savetxt(path_smb_errors + 'smb_ann_RMSE.csv', rmse_glaciers, delimiter=";", fmt="%.7f")
np.savetxt(path_smb_errors + 'smb_ann_bias.csv', bias_glaciers, delimiter=";", fmt="%.7f")

np.savetxt(path_smb_errors + 'smb_lasso_MAE.csv', mae_glaciers_lasso, delimiter=";", fmt="%.7f")
np.savetxt(path_smb_errors + 'smb_lasso_RMSE.csv', rmse_glaciers_lasso, delimiter=";", fmt="%.7f")
np.savetxt(path_smb_errors + 'smb_lasso_bias.csv', bias_glaciers_lasso, delimiter=";", fmt="%.7f")

##############   PLOTS    ############################



idx = y_error.argsort()
y_error_plt, aug_temp_plt, oct_temp_plt, lat_plt, lon_plt = y_error[idx], X[:,sorted_coef_error_idxs[3]][idx], X[:,sorted_coef_error_idxs[2]][idx], X[:,sorted_coef_error_idxs[0]][idx], X[:,sorted_coef_error_idxs[1]][idx]
march_temp_plt, apr_snow_plt = X[:,sorted_coef_error_idxs[4]][idx], X[:,sorted_coef_error_idxs[5]][idx]

idx_rmse = rmse_glaciers.argsort()

rmse_glaciers_plt, lat_glaciers_plt, lon_glaciers_plt = rmse_glaciers[idx_rmse], lat_glaciers[idx_rmse], lon_glaciers[idx_rmse]



# Data points
#plt.figure(figsize=(6,6))
#cm = plt.cm.get_cmap('magma')
#plt.title("Error distribution", fontsize=16)
#plt.ylabel('January temperature', fontsize=14)
#plt.xlabel('November temperature', fontsize=14)
#sc = plt.scatter(nov_temp_plt, jan_temp_plt, c=y_error_plt, s=100, alpha=0.5, edgecolors='none', cmap=cm)
#plt.colorbar(sc)
#plt.show()

#import pdb; pdb.set_trace()

# Hexbin
plt.figure(1, figsize=(6,6))
hb = plt.hexbin(aug_temp_plt, oct_temp_plt, C = y_error_plt, gridsize=5, reduce_C_function = np.median, cmap='OrRd')
plt.title("Error distribution", fontsize=16)
plt.ylabel('October temperature anomaly', fontsize=14)
plt.xlabel('August temperature anomaly', fontsize=14)
cb = plt.colorbar(hb)
cb.set_label('Median error')
plt.show()

plt.figure(2, figsize=(6,6))
hb = plt.hexbin(apr_snow_plt, march_temp_plt, C = y_error_plt, gridsize=5, reduce_C_function = np.median, cmap='OrRd')
plt.title("Error distribution", fontsize=16)
plt.xlabel('April snow anomaly', fontsize=14)
plt.ylabel('March temperature anomaly', fontsize=14)
cb = plt.colorbar(hb)
cb.set_label('Median error')
plt.show()

plt.figure(3, figsize=(6,6))
cm = plt.cm.get_cmap('magma')
plt.title("Glacier RMSE distribution", fontsize=16)
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
sc = plt.scatter(lon_glaciers_plt, lat_glaciers_plt, c=rmse_glaciers_plt, s=200, alpha=0.7, edgecolors='none', cmap=cm)
plt.colorbar(sc)
plt.show()