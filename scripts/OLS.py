# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:43:57 2018

@author: bolibarj
"""


## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import numpy.polynomial.polynomial as poly
from scipy import stats
from scipy.optimize import curve_fit

######   FILE PATHS    #######
    
# Folders     
workspace = 'C:\\Jordi\\PhD\\Python\\'
#path_obs = 'C:\\Jordi\\PhD\\Data\\Obs\\'
path_smb = workspace + 'ALPGM\\glacier_data\\smb\\'
path_glacier_coordinates = workspace + 'ALPGM\\glacier_data\\glacier_coordinates\\'
path_smb_simulations = path_smb + 'smb_simulations\\'

#smb_models_3_5 = genfromtxt(workspace + 'model_combinations_3_5_params.csv', delimiter=';', skip_header=1, dtype=[('ID', '<i8'), ('combination_idx', np.dtype(np.int16)), ('r2_adj', '<f8'), ('VIF', np.dtype(np.float64)), ('p-value', '<f8')]) 
smb_models_3_5 = genfromtxt(workspace + 'model_combinations_3_5_params.csv', delimiter=';', skip_header=1, dtype=None) 
#smb_models_6_7 = genfromtxt(workspace + 'model_combinations_6_7_params.csv', delimiter=';', skip_header=1, dtype=None) 


# 3 to 5 parameters
sorted_models_3_5 = np.flip(np.sort(smb_models_3_5, order='f2'))
count = 0
chosen_models_3_5 = []
for model in sorted_models_3_5:
#    print("model['f3']: " + str(model['f3']))
    idxs = np.asarray(eval(model['f2']))
    vif = np.asarray(eval(model['f3']))
    pvalue = model['f4']
#    print("vif: " + str(vif.max()))
#    print("np.any(vif >= 10): " + str(np.any(vif >= 10)))
    # Ensuring Bonferroni condition with p-value divided by number of cases n^r
    if(vif.max() >= 2 or (pvalue > 0.005/24300000) or idxs.size < 5):
        continue
    else:
       chosen_models_3_5.append(model)
       count = count+1
    if(count > 50):
        break

chosen_models_3_5 = np.asarray(chosen_models_3_5)
sorted_chosen_3_5 = np.flip(np.sort(chosen_models_3_5, order = 'f2'))
#np.savetxt(workspace + 'chosen_models_3_5.csv', sorted_chosen_3_5, delimiter=',')
with open(workspace + 'chosen_models_3_5.csv', 'wb') as smb_f:
                        np.savetxt(smb_f, sorted_chosen_3_5, delimiter = ';', fmt='%s')
                        
# 6 to 7 parameters
#sorted_models_6_7 = np.flip(np.sort(smb_models_6_7, order='f2'))
#count = 0
#chosen_models_6_7 = []
#for model in smb_models_6_7:
##    print("model['f3']: " + str(model['f3']))
#    vif = np.asarray(eval(model['f3']))
##    print("vif: " + str(vif.max()))
##    print("np.any(vif >= 10): " + str(np.any(vif >= 10)))
#    if(vif.max() >= 4):
#        continue
#    else:
#       chosen_models_6_7.append(model)
#       count = count+1
#    if(count > 50):
#        break
#
#chosen_models_6_7 = np.asarray(chosen_models_6_7)
#sorted_chosen_6_7 = np.flip(np.sort(chosen_models_6_7, order = 'f2'))
#with open(workspace + 'chosen_models_6_7.csv', 'wb') as smb2_f:
#                        np.savetxt(smb2_f, sorted_chosen_6_7, delimiter = ';', fmt='%s')
