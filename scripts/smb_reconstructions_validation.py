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
from pathlib import Path
from sklearn.metrics import mean_squared_error

######   FILE PATHS    #######

# Folders     
workspace = str(Path(os.getcwd()).parent) + '\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 
path_obs = 'C:\\Jordi\\PhD\\Data\\SMB\\'
path_smb = workspace + 'glacier_data\\smb\\smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\smb\\'

#######################    FUNCTIONS    ##########################################################

def retrieve_glacier_reconstructions(glacier_info):
    
    glacier_list = np.asarray(os.listdir(path_smb))
    for glacier_file in glacier_list:
            
        if(glacier_file[15:19] == glacier_info['RGI_ID']):
            glacier_SMB_reconstruction = genfromtxt(path_smb + glacier_file, delimiter=';') 
            
            return True, glacier_SMB_reconstruction
        
    return False, []

def plot_and_save(glacier_info, glacier_annual_obs, glacier_annual_reconstructions, plot_count):
    fig = plt.figure(plot_count)
    plt.title("Glacier " + str(glacier_info['name']))
    plt.plot(range(2000, 2016), glacier_annual_obs, label="Obs")
    plt.plot(range(2000, 2016), glacier_annual_reconstructions, label="Reconstructions")
    plt.legend()
    
    fig.savefig(path_obs + 'obs_reconstructions_comparison\\' + str(glacier_info['name']) + '_2000_2015_Davaze_et_al_ALPGM_SMB_comparison.png')
    
    plt.close()

##################################################################################################



###############################################################################
###                           MAIN                                          ###
###############################################################################

# Open the remote sensing glacier observations for the European Alps (Davaze et al., in review)
SMB_rs_european_alps = genfromtxt(path_obs + 'Data_MB_annuel_European_Alps.csv', delimiter=';', skip_header=1, dtype=str) 

# Open the glacier inventory for the French Alps
glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), 
                       ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), 
                       ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8')])
    
    
glacier_metrics = {'RGI_ID': [], 'name': [], 'bias': [], 'RMSE': []}

# Iterate and compare glacier SMB observations in order to compute metrics
    
plot_count = 1
for glacier_obs in SMB_rs_european_alps:
    
    # Gather and format all glacier information
    glacier_info = {'RGI_ID': glacier_obs[0][-4:], 'GLIMS_ID': glacier_obs[1], 
                    'name': glacier_obs[9], 'massif': glacier_obs[12], 
                    'annual_SMB': np.asarray(glacier_obs[13:30], dtype=float)/1000, 'cumulative_SMB':float(glacier_obs[-2])/1000}
    
    # We retrieve the SMB reconstructions for this glacier
    found, glacier_SMB_reconstruction = retrieve_glacier_reconstructions(glacier_info)
    
    if(found):
        print("\nRemote sensing obs name: " + str(glacier_info['name']))
        
        glacier_metrics['name'].append(glacier_info['name'])
        glacier_metrics['RGI_ID'].append(glacier_info['RGI_ID'])
        
        glacier_annual_obs = glacier_info['annual_SMB'][:-1]
        glacier_annual_reconstructions = glacier_SMB_reconstruction[-16:,1].flatten()
        
#            print("Remote sensing obs annual series: " + str(glacier_annual_obs))
#            
#            print("\nRemote sensing reconstructions: " + str(glacier_annual_reconstructions))
        
        bias = (np.sum(glacier_annual_reconstructions) - glacier_info['cumulative_SMB'])/len(range(2000, 2016))
        glacier_metrics['bias'].append(bias)
        
        print("\nBias: " + str(bias))
        print("Remote sensing reconstructions: " + str(np.sum(glacier_annual_reconstructions)))
        print("Remote sensing obs: " + str(glacier_info['cumulative_SMB']))
            
        if(glacier_info['annual_SMB'][0] != 9999):
            
            rmse = np.sqrt(mean_squared_error(glacier_annual_obs, glacier_annual_reconstructions))
            glacier_metrics['RMSE'].append(rmse)
            
            # We plot the comparison and store the individual plots
            plot_and_save(glacier_info, glacier_annual_obs, glacier_annual_reconstructions, plot_count)

            plot_count=plot_count+1
            
        else:
            glacier_metrics['RMSE'].append([])
            

# We compute the histogram of bias            
bias_histogram, bins = np.histogram(glacier_metrics['bias'], bins="auto")

fig = plt.figure(plot_count)
#plt.ylabel('Bias (m.w.e.)')
plt.xlabel('Bias (m.w.e. a$^{-1}$)')
plt.bar(bins[:-1] + np.diff(bins) / 2, bias_histogram, np.diff(bins), label='Bias histogram', color='steelblue')
plt.axvline(x=0, color='black')
plt.legend()
plt.show()

            
#import pdb; pdb.set_trace()