# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:49:34 2020

@author: Jordi Bolibar

SMB ANNUAL BIAS CORRECTION COMPUTATION WITH ASTER DATA
FROM DAVAZE ET AL. (2020) FOR ALL AVAILABLE GLACIERS IN THE FRENCH ALPS

"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
#import copy
from pathlib import Path
from sklearn.metrics import mean_squared_error
import proplot as plot
import subprocess
import pandas as pd
from difflib import SequenceMatcher

######   FLAGS    #######
only_final = False

calibrate_training_data = False

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

#######################    FUNCTIONS    ##########################################################

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def retrieve_glacier_reconstructions(glacier_info, glims_2003):
    
    glacier_list = np.asarray(os.listdir(path_reconstructions))
    for glacier_file in glacier_list:
        
        # Retrieve SMB reconstructions
        if(isfloat(glacier_file[15:19])):
        
            if(glacier_file[:14] == glacier_info['GLIMS_ID'] and float(glacier_file[15:19]) == glacier_info['RGI_ID']):
                glacier_SMB_reconstruction = genfromtxt(os.path.join(path_reconstructions, glacier_file), delimiter=';') 
                
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
                
                return True, glacier_SMB_reconstruction, glacier_sim_info
        
    return False, [], []


def similar(a, b):
    ratios = []
    for glacier_name in a:
        ratios.append(SequenceMatcher(None, glacier_name, b).ratio())
    ratios = np.asarray(ratios)
    return ratios.max(), ratios.argmax()


##################################################################################################



###############################################################################
###                           MAIN                                          ###
###############################################################################
    
    
if(not only_final):

    # Open the remote sensing glacier observations for the European Alps (Davaze et al., in review)
    
    aster_2000_2016 = pd.read_csv(os.path.join(path_obs, 'MB_gla_by_gla_2000.5-2016.8.csv'), sep=';')
    
    # Open the glacier inventory for the French Alps
    glims_2003 = pd.read_csv(os.path.join(path_glims, 'GLIMS_2003.csv'), sep=';')
    glims_2015 = pd.read_csv(os.path.join(path_glims, 'GLIMS_2015.csv'), sep=';')
    glims_rabatel = pd.read_csv(os.path.join(path_glims, 'GLIMS_Rabatel_30_2003.csv'), sep=';')
    
    bias_correction = {'ID':[], 'GLIMS_ID':[], 'Name':[], 'bias_correction':[], 'bias_correction_perc':[]}
    
    # Iterate and compare glacier SMB observations in order to compute metrics
    plot_count = 1
    for index, glacier_obs in aster_2000_2016.iterrows():
        rgi_id = float(glacier_obs['RGIId'][-5:])
        
        id_idx_2003 = np.where(rgi_id == glims_2003['ID'])[0]
        
        glims_id = glims_2003['GLIMS_ID'][id_idx_2003].values
        
        glims_id_rabatel = glims_rabatel['GLIMS_ID'].values
        is_training = np.any(glims_id == glims_id_rabatel)
        
        # We filter glaciers with errors greater than 0.15 m.w.e. a-1 and glaciers from the training dataset
        if(id_idx_2003.size > 0 and glims_id.size == 1 and glacier_obs['MB_error [m w.e a-1]'] < 0.15 and not is_training):
            
            id_idx_2015 = np.where(glims_id[0] == glims_2015['GLIMS_ID'].values)[0]
            
            glims_glacier_2003 = glims_2003.iloc[id_idx_2003]
            glims_glacier_2015 = glims_2015.iloc[id_idx_2015]
            
            if(glims_glacier_2003['Area'].values[0] > 1 and glims_glacier_2015['Area'].values.size > 0 and glims_glacier_2003['Area'].values.size > 0):
                
#                print("\n--------------------------- " )
#                print("id_idx_2015:  " + str(id_idx_2015))
#                print("glims_glacier_2015['Area'].values: " + str(glims_glacier_2015['Area'].values))
#                print("glims_glacier_2003['Area'].values: " + str(glims_glacier_2003['Area'].values))
                
#                import pdb; pdb.set_trace()
                
                if(glims_glacier_2015['Area'].values.size > 1):
                    glims_area_2015 = np.sum(glims_glacier_2015['Area'].values)
                else: 
                    glims_area_2015 = glims_glacier_2015['Area'].values[0]
                
                # We correct the surface area error in the ASTER data
#                print("\nUncorrected ASTER MB: " + str(glacier_obs['MB [m w.e a-1]']))
#                print("glims_id: " + str(glims_id))
#                print("Area 2003: " + str(glims_glacier_2003['Area'].values[0]))
#                print("Area 2015: " + str(glims_area_2015))
                
                mb_aster = (glacier_obs['MB [m w.e a-1]']*glims_glacier_2003['Area'].values[0])/((glims_glacier_2003['Area'].values[0] + glims_area_2015)/2)
                
                print("Corrected ASTER MB: " + str(mb_aster))
                
                # Gather and format all glacier information
                glacier_info = {'RGI_ID': glims_glacier_2003['ID'].values[0], 'GLIMS_ID': glims_glacier_2003['GLIMS_ID'].values[0], 
                                'name': glims_glacier_2003['Glacier'].values[0], 'massif': glims_glacier_2003['Massif'].values[0], 
                                'alpgm_SMB': np.nan, 'aster_SMB': mb_aster}
                
                # We retrieve the SMB reconstructions for this glacier
                found, glacier_SMB_reconstruction, glacier_sim_info = retrieve_glacier_reconstructions(glacier_info, glims_2003)
                aster_obs = glacier_info['aster_SMB']
                
                if(found and aster_obs < 0):
    #                print("\nRemote sensing obs name: " + str(glacier_info['name']))
    #                print("Obs GLIMS ID: " + str(glacier_info['GLIMS_ID']))
    #                print("Obs RGI ID: " + str(glacier_info['RGI_ID']))
    #                
    #                print("\nReconstructions name: " + str(glacier_sim_info['name']))
    #                print("Reconstructions GLIMS ID: " + str(glacier_sim_info['GLIMS_ID']))
    #                print("Reconstructions RGI ID: " + str(glacier_sim_info['ID']))
                    
                    alpgm_reconstructions = glacier_SMB_reconstruction[-15:,1].mean()
                    
                    correction = aster_obs - alpgm_reconstructions
                    correction_perc = aster_obs/alpgm_reconstructions
                    
                    #                if((aster_obs/alpgm_reconstructions) > 2 or (aster_obs/alpgm_reconstructions) < 0):
    #                print("\naster_obs: " + str(aster_obs))
    #                print("alpgm_reconstructions: " + str(alpgm_reconstructions))
    #                print("SMB series: " + str(glacier_SMB_reconstruction[-15:,1]))
                    
    #                if(correction > 0.1 and correction < 2):
                    
                    bias_correction['ID'].append(glacier_info['RGI_ID'])
                    bias_correction['GLIMS_ID'].append(glacier_info['GLIMS_ID'])
                    bias_correction['Name'].append(glacier_info['name'])
                    bias_correction['bias_correction'].append(correction)
                    bias_correction['bias_correction_perc'].append(correction_perc)
                
bias_correction_df = pd.DataFrame(bias_correction)
bias_correction_df.to_csv(os.path.join(path_smb_validation, 'SMB_bias_correction.csv'), sep=";")

######### Recalibrate training data  #############

if(calibrate_training_data):

    # Open the glacier inventory for the French Alps
    glims_rabatel_df = pd.read_csv(os.path.join(path_glims, 'GLIMS_Rabatel_30_2003.csv'), sep=';')
    
    smb_raw_df = pd.read_csv(os.path.join(path_smb_root, 'SMB_raw_temporal.csv'), sep=';', header=None)
    
    for (glacier_idx, glacier_info), (smb_idx, glacier_smb) in zip(glims_rabatel_df.iterrows(), smb_raw_df.iterrows()):
        
        print("\nGlacier: " + str(glacier_info['Glacier']))
        
        # Bias correction based on ASTER SMB (2000-2016)
        smb_bias_correction = bias_correction_df['bias_correction'][bias_correction_df['GLIMS_ID'] == glacier_info['GLIMS_ID']]
        rgi_ID = bias_correction_df['ID'][bias_correction_df['GLIMS_ID'] == glacier_info['GLIMS_ID']]
        
        print("RGI ID: " + str(rgi_ID))
        
        if(rgi_ID.size > 1):
            areas = []
            for glacier_id in rgi_ID:
    #            import pdb; pdb.set_trace()
                areas.append(aster_2000_2016['Tot_GLA_area [km2]'][aster_2000_2016['RGIId'].values == "RGI60-11.0" + str(int(glacier_id))].values[0])
            areas = np.asarray(areas)
            idx_max = np.argmax(areas)
            
            rgi_ID = rgi_ID.values[idx_max]
            smb_bias_correction = smb_bias_correction.values[idx_max]
        else:
            smb_bias_correction = smb_bias_correction.values[0]
        
        print("Bias correction: " + str(smb_bias_correction) + " m.w.e. a-1")
        
        smb_raw_df.iloc[smb_idx] = glacier_smb + smb_bias_correction
        
    
    # Store corrected SMB observations
    smb_raw_df.to_csv(os.path.join(path_smb_root,  'SMB_raw_temporal_ASTER.csv'), sep=';', header=None, index=False)

#bias_correction_df.plot('ID', 'bias_correction')
#plt.show()




