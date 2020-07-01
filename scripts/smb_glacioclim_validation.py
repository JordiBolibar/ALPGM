# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:42:26 2020

@author: bolibarj


SCRIPT COMPARING 1984-2014 ANNUAL GLACIER-WIDE SMB DATA
BETWEEN GLACIOCLIM, ALPGM CV MODELS AND ALPGM FITTED MODELS


"""

## Dependencies: ##
import matplotlib.pyplot as plt
import matplotlib as mpl
import proplot as plot
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy import stats
import pandas as pd

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def retrieve_glacier_reconstructions(glacier_info, glims_2003):
    
    ### Start with final dataset  (no cross-validation)  ###
    glacier_list = np.asarray(os.listdir(path_reconstructions))
    for glacier_file in glacier_list:
        
        # Retrieve SMB reconstructions
        if(isfloat(glacier_file[15:19])):
        
            if(glacier_file[:14] == glacier_info['GLIMS_ID'] and float(glacier_file[15:19]) == glacier_info['RGI_ID']):
                glacier_SMB_reconstruction = genfromtxt(os.path.join(path_reconstructions, glacier_file), delimiter=';')[:,1][-32:]
                
                # Retrieve glacier info from database
                
                glacier_rgi_id_idx = np.where((glims_2003['ID'] == float(glacier_info['RGI_ID'])))[0]
                
    #            print(glacier_rgi_id_idx)
    #            import pdb; pdb.set_trace()
                
                if(len(glacier_rgi_id_idx) > 1):
    #                print("More than one match")
    #                import pdb; pdb.set_trace()
                    glacier_glims_idx = np.where((glims_2003['GLIMS_ID'] == glacier_info['GLIMS_ID'])[glacier_rgi_id_idx])[0][0]
                    glacier_idx = glacier_rgi_id_idx[glacier_glims_idx]
                    glacier_sim_info = {'name': glims_2003['Glacier'][glacier_idx], 'GLIMS_ID': glims_2003['GLIMS_ID'][glacier_idx], 'ID': glims_2003['ID'][glacier_idx]}
                else:
    #                import pdb; pdb.set_trace()
                    glacier_idx = np.where((glims_2003['ID'] == float(glacier_info['RGI_ID'])))[0][0]
                    glacier_sim_info = {'name': glims_2003['Glacier'][glacier_idx], 'GLIMS_ID': glims_2003['GLIMS_ID'][glacier_idx], 'ID': glims_2003['ID'][glacier_idx]}
                
    
    ####  Now do the same with the CV dataset  ####   
    path_smb_training = os.path.join(workspace, 'glacier_data', 'glacier_evolution', 'training', 'SMB')
    training_glacier_list = np.asarray(os.listdir(path_smb_training))
    
    for glacier_file in training_glacier_list:
        
        if(glacier_file[:14] == glacier_info['GLIMS_ID']):
            
            glacier_SMB_training = genfromtxt(os.path.join(path_smb_training, glacier_file), delimiter=';')[:,1]
            
#            nan_head = np.zeros(1984-1967)
#            nan_head[:] = np.nan
#            glacier_SMB_training = np.concatenate((nan_head, np.cumsum(glacier_SMB_training[:,1])))   #### Accumulated!
            glacier_SMB_training = np.concatenate((glacier_SMB_training, np.array([np.nan])))
        
    return glacier_SMB_reconstruction, glacier_SMB_training

######   FILE PATHS    #######

# Folders     
workspace = Path(os.getcwd()).parent
path_glims = os.path.join(workspace, 'glacier_data', 'GLIMS') 
path_obs = 'C:\\Jordi\\PhD\\Data\\SMB\\'
path_smb_root = os.path.join(workspace, 'glacier_data', 'smb')
path_smb = os.path.join(workspace, 'glacier_data', 'smb', 'smb_simulations', 'SAFRAN', '1', 'all_glaciers_1967_2015', 'smb')
path_smb_validation = os.path.join(workspace, 'glacier_data', 'smb', 'smb_validation')
path_smb_plots = os.path.join(workspace, 'glacier_data', 'smb', 'smb_simulations', 'reconstruction_plots')

path_reconstructions = 'C:\\Jordi\\PhD\\Publications\\Second article\\Dataset\\Updated dataset\\'

# Open the glacier inventory for the French Alps
glims_2003 = pd.read_csv(os.path.join(path_glims, 'GLIMS_2003.csv'), sep=';')
smb_raw_df = pd.read_csv(os.path.join(path_smb_root, 'SMB_raw_temporal.csv'), sep=';', header=None)

glacioclim_glaciers = {'Name': np.array(['Mer de Glace', 'Argentiere', 'Saint Sorlin', 'Gebroulaz']), 'SMB_GLACIOCLIM': [], 'SMB_ALPGM_full': [], 'SMB_ALPGM_CV': [],
                       'RGI_IDs': np.array([3643, 3638, 3674, 3671]), 'GLIMS_IDs': np.array(['G006934E45883N','G006985E45951N','G006159E45160N','G006629E45295N']),
                       'idx_training': np.array([0,30,2,13])}

for glacier, rgi_id, glims_id, idx_training in zip(glacioclim_glaciers['Name'], glacioclim_glaciers['RGI_IDs'], glacioclim_glaciers['GLIMS_IDs'], glacioclim_glaciers['idx_training']):
    
    glacier_info = {'GLIMS_ID': glims_id, 'RGI_ID': rgi_id}
    glacier_SMB_reconstruction, glacier_SMB_training = retrieve_glacier_reconstructions(glacier_info, glims_2003)
    
    glacioclim_glaciers['SMB_GLACIOCLIM'].append(smb_raw_df.iloc[idx_training][-32:])
    glacioclim_glaciers['SMB_ALPGM_full'].append(glacier_SMB_reconstruction)
    glacioclim_glaciers['SMB_ALPGM_CV'].append(glacier_SMB_training)  


##### Plot  #####
    
fig1, ax1 = plot.subplots(ncols=2, nrows=2, axwidth=2.5, sharey=1, sharex=3)
#fig1.suptitle("MB reconstructions vs glaciological observations")

years = range(1984, 2016)

for i in range(0,4):
    ax1[i].format(title=glacioclim_glaciers['Name'][i])
    ax1[i].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    h1 = ax1[i].plot(years, np.cumsum(glacioclim_glaciers['SMB_GLACIOCLIM'][i]), label='GLACIOCLIM', c='darkgreen')
    h2 = ax1[i].plot(years, np.cumsum(glacioclim_glaciers['SMB_ALPGM_full'][i]), label='ANN', c='steelblue')
    h3 = ax1[i].plot(years, np.cumsum(glacioclim_glaciers['SMB_ALPGM_CV'][i]), label='ANN CV', c='darkgoldenrod')
    
fig1.legend(((h1, h2, h3)), loc='r', ncols=1, frame=False)
    
#    import pdb; pdb.set_trace()
ax1.format(
        abc=True, abcloc='ll',
        ygridminor=True,
        ylabel="MB (m.w.e.)", xlabel="Year",
        ytickloc='both', yticklabelloc='left'
)
  
plt.show()

#import pdb; pdb.set_trace()


