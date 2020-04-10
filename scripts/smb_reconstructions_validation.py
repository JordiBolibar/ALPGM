# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:49:34 2020

@author: Jordi Bolibar
"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

######   FILE PATHS    #######

# Folders     
workspace = str(Path(os.getcwd()).parent) + '\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 
path_obs = 'C:\\Jordi\\PhD\\Data\\SMB\\'
path_smb = workspace + 'glacier_data\\smb\\smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\smb\\'

#######################    FUNCTIONS    ##########################################################

def retrieve_glacier_reconstructions(glacier_info, glims_2003):
    
    glacier_list = np.asarray(os.listdir(path_smb))
    for glacier_file in glacier_list:
        
        # Retrieve SMB reconstructions
        if(glacier_file[:14] == glacier_info['GLIMS_ID'] and glacier_file[15:19] == glacier_info['RGI_ID']):
            glacier_SMB_reconstruction = genfromtxt(path_smb + glacier_file, delimiter=';') 
            
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

def plot_and_save(glacier_info, glacier_annual_obs, glacier_annual_reconstructions, plot_count):
    fig = plt.figure(plot_count)
    plt.title("Glacier " + str(glacier_info['name']))
    plt.plot(range(2001, 2016), glacier_annual_obs, label="Obs")
    plt.plot(range(2001, 2016), glacier_annual_reconstructions, label="Reconstructions")
    plt.legend()
    
    fig.savefig(path_obs + 'obs_reconstructions_comparison\\' + str(glacier_info['name']) + '_2000_2015_Davaze_et_al_ALPGM_SMB_comparison.png')
    
    plt.close()

##################################################################################################



###############################################################################
###                           MAIN                                          ###
###############################################################################

# Open the remote sensing glacier observations for the European Alps (Davaze et al., in review)
SMB_rs_european_alps = genfromtxt(path_obs + 'MB_remote_sensing_Davaze_et_al_French_Alps.csv', delimiter=';', skip_header=1, dtype=str) 

# Open the glacier inventory for the French Alps
glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1, dtype=[('Area', '<f8'), 
                     ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), 
                     ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), 
                     ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8'), ('ID', '<f8')])
    
    
glacier_metrics = {'RGI_ID': [], 'name': [], 'bias': [], 'RMSE': []}

# Iterate and compare glacier SMB observations in order to compute metrics
    
plot_count = 1
for glacier_obs in SMB_rs_european_alps:
    
    if(float(glacier_obs[-3]) != 9999000):
        cum_smb = float(glacier_obs[-3])/1000
    else:
        cum_smb = float(glacier_obs[-2])/1000
    
    # Gather and format all glacier information
    glacier_info = {'RGI_ID': glacier_obs[10], 'GLIMS_ID': glacier_obs[11], 
                    'name': glacier_obs[9], 'massif': glacier_obs[12], 
                    'annual_SMB': np.asarray(glacier_obs[14:30], dtype=float)/1000, 'cumulative_SMB':cum_smb}
    
    # We retrieve the SMB reconstructions for this glacier
    found, glacier_SMB_reconstruction, glacier_sim_info = retrieve_glacier_reconstructions(glacier_info, glims_2003)
    
    if(found):
        print("\nRemote sensing obs name: " + str(glacier_info['name']))
        print("Obs GLIMS ID: " + str(glacier_info['GLIMS_ID']))
        print("Obs RGI ID: " + str(glacier_info['RGI_ID']))
        
        print("\nReconstructions name: " + str(glacier_sim_info['name']))
        print("Reconstructions GLIMS ID: " + str(glacier_sim_info['GLIMS_ID']))
        print("Reconstructions RGI ID: " + str(glacier_sim_info['ID']))
        
        glacier_metrics['name'].append(glacier_info['name'])
        glacier_metrics['RGI_ID'].append(glacier_info['RGI_ID'])
        
        glacier_annual_obs = glacier_info['annual_SMB'][:-1]
        glacier_annual_reconstructions = glacier_SMB_reconstruction[-15:,1].flatten()
        
#            print("Remote sensing obs annual series: " + str(glacier_annual_obs))
#            
#            print("\nRemote sensing reconstructions: " + str(glacier_annual_reconstructions))
        
        bias = (np.sum(glacier_annual_reconstructions) - glacier_info['cumulative_SMB'])/len(range(2001, 2016))
        glacier_metrics['bias'].append(bias)
        
        print("\nBias: " + str(bias))
        print("\nRemote sensing SMB: " + str(glacier_info['cumulative_SMB']) + " m w.e.")
        print("Reconstructions SMB: " + str(np.sum(glacier_annual_reconstructions)) + " m w.e.")
        
        print("\nRemote sensing annual SMB: " + str(glacier_info['cumulative_SMB']/len(range(2001, 2016))) + " m w.e. a-1")
        print("Reconstructions annual SMB: " + str(np.sum(glacier_annual_reconstructions)/len(range(2001, 2016))) + " m w.e. a-1")
        
        print("\n----------------------------------------------")
            
        if(glacier_info['annual_SMB'][0] != 9999):
            
            rmse = np.sqrt(mean_squared_error(glacier_annual_obs, glacier_annual_reconstructions))
            glacier_metrics['RMSE'].append(rmse)
            
            # We plot the comparison and store the individual plots
            plot_and_save(glacier_info, glacier_annual_obs, glacier_annual_reconstructions, plot_count)

            plot_count=plot_count+1
            
        else:
            glacier_metrics['RMSE'].append([])
            
print("Average annual bias: " + str(np.mean(glacier_metrics['bias'])))


# We compute the histogram of bias            
bias_histogram, bins = np.histogram(glacier_metrics['bias'], bins='auto')

fig = plt.figure(plot_count)
#plt.ylabel('Bias (m.w.e.)')
plt.title("SMB bias for French alpine glaciers (2000 - 2015)")
plt.xlabel('Bias (m.w.e. a$^{-1}$)')
plt.bar(bins[:-1] + np.diff(bins) / 2, bias_histogram, np.diff(bins), label='Bias histogram', color='steelblue')
plt.axvline(x=0, color='black')
plt.legend()
plt.show()

            
#import pdb; pdb.set_trace()