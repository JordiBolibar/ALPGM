# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:49:34 2020

@author: Jordi Bolibar

PROCESS AND UPDATE BOLIBAR ET AL. (2020B) SMB DATASET

"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path
from sklearn.metrics import mean_squared_error
import proplot as plot
import subprocess
import pandas as pd
import xarray as xr
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

path_backup = 'C:\\Users\\bolibarj\\Desktop\\ALPGM_backup\\smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\'

path_reconstructions = 'C:\\Jordi\\PhD\\Publications\\Second article\\Dataset\\Updated dataset\\'
path_aster_reconstructions = 'C:\\Jordi\\PhD\\Publications\\Second article\\Dataset\\ASTER dataset\\'
path_dataset = 'C:\\Jordi\\PhD\\Publications\\Second article\\Dataset\\'


#######################    FUNCTIONS    ##########################################################

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def automatic_file_name_save(file_name_h, file_name_t, data, f_format):
    file_name = file_name_h + file_name_t
    appendix = 2
    stored = False
    while not stored:
        if not os.path.exists(file_name):
            try:
                if(f_format == 'csv'):
                    np.savetxt(file_name, data, delimiter=";", fmt="%.7f")
                else:
                    with open(file_name, 'wb') as file_f:
                        np.save(file_f, data)
                stored = True
            except IOError:
                print("File currently opened. Please close it to proceed.")
                os.system('pause')
                # We try again
                try:
                    print("\nRetrying storing " + str(file_name))
                    if(f_format == 'csv'):
                        np.savetxt(file_name, data, delimiter=";", fmt="%.7f")
                    else:
                        with open(file_name, 'wb') as file_f:
                            np.save(file_f, data)
                    stored = True
                except IOError:
                    print("File still not available. Aborting simulations.")
        else:
            file_name = file_name_h + str(appendix) + "_" + file_name_t
            appendix = appendix+1

def store_file(data, path, midfolder, file_description, glimsID, year_start, year):
    year_range = np.asarray(range(year_start, year))
    data =  np.asarray(data).reshape(-1,1)
    data_w_years = np.column_stack((year_range, data))
    path_midfolder = os.path.join(path, midfolder)
    if not os.path.exists(path_midfolder):
        os.makedirs(path_midfolder)
        
#    file_name = path + midfolder + glimsID + "_" + str(file_description) + '.csv'
    file_name_h = os.path.join(path, midfolder, str(glimsID) + "_")
    file_name_t = str(file_description) + '.csv'
    # We save the file with an unexisting name
    automatic_file_name_save(file_name_h, file_name_t, data_w_years, 'csv')

###################################################################################################
    

###############################################################################
###                           MAIN                                          ###
###############################################################################

aster_2000_2016 = pd.read_csv(os.path.join(path_obs, 'MB_gla_by_gla_2000.5-2016.8.csv'), sep=';')

# Open the glacier inventory for the French Alps
glims_2003 = pd.read_csv(os.path.join(path_glims, 'GLIMS_2003.csv'), sep=';')
glims_2015 = pd.read_csv(os.path.join(path_glims, 'GLIMS_2015.csv'), sep=';')

bias_correction_df = pd.read_csv(os.path.join(path_smb_validation, 'SMB_bias_correction.csv'), sep=";")

final_smb_reconstructions_dict = {'SMB':[], 'SMB_ASTER':[], 'RGI_ID':[], 'GLIMS_ID':[], 'Name':[]}

aster_ID_head = aster_2000_2016['RGIId'].values[0][:-5]

path_smb_reconstructions = os.path.join(path_backup, 'smb')
path_smb_ensemble = os.path.join(path_backup, 'ensemble_smb')

glacier_list = np.asarray(os.listdir(path_smb_reconstructions))
glacier_ensemble_list = np.asarray(os.listdir(path_smb_ensemble))

first = True
for glacier_file, ensemble_file in zip(glacier_list, glacier_ensemble_list):
    
    # Retrieve SMB reconstructions
    if(isfloat(glacier_file[15:19])):
        rgiID = float(glacier_file[15:19])
    else:
        rgiID = 0
        
    glimsID = glacier_file[:14]
    
    # Load MB series
    glacier_SMB = genfromtxt(os.path.join(path_reconstructions, glacier_file), delimiter=';') 
    
    # Load MB ensembles
    with open(os.path.join(path_smb_ensemble, ensemble_file), 'rb') as ens_f:
        glacier_ensemble = np.load(ens_f, encoding='latin1', allow_pickle=True)
        
#    import pdb; pdb.set_trace()
    
    # Retrieve glacier info from database
    
    glacier_rgi_id_idx = np.where((aster_2000_2016['RGIId'].values == (str(aster_ID_head) + '0' +  str(int(rgiID)))))[0]
    
    glacier_idx = np.where((glims_2003['ID'] == rgiID))[0][0]
    
    if(glacier_idx.size > 0):
        glacier_name = glims_2003['Glacier'][glacier_idx]
    elif():
        glacier_idx = np.where((glims_2003['GLIMS_ID'] == glimsID))[0][0]
        glacier_name = glims_2003['Glacier'][glacier_idx]
        
#    print("glacier_name: " + str(glacier_name))
    
    bias_correction = bias_correction_df['bias_correction'][bias_correction_df['ID'] == rgiID].values
    
    if(glacier_SMB[:,1].size < 49):
        nan_tail = np.ones(12)
        nan_tail[:] = np.nan
        glacier_SMB_reconstruction = np.concatenate((glacier_SMB[:,1], nan_tail))
    else:
        glacier_SMB_reconstruction = glacier_SMB[:,1]
    
    glacier_calibrated_SMB_reconstruction = copy.deepcopy(glacier_SMB_reconstruction)
    
    if(bias_correction.size > 0):
        
        print("\nGlacier: " + str(glacier_name))
        print("Bias correction: " + str(bias_correction[0]))
        
        # We apply the correction to the average MB series
        glacier_calibrated_SMB_reconstruction[-15:]  = glacier_calibrated_SMB_reconstruction[-15:] + bias_correction[0]
        
        # We apply the correction to the ensemble MB series
        glacier_ensemble[:,-15:] = glacier_ensemble[:,-15:] + bias_correction[0]
    
    if(first):
        final_smb_reconstructions_dict['SMB'] = glacier_SMB_reconstruction
        final_smb_reconstructions_dict['SMB_ASTER'] = glacier_calibrated_SMB_reconstruction
        first = False
    else:
        final_smb_reconstructions_dict['SMB'] = np.vstack((final_smb_reconstructions_dict['SMB'], glacier_SMB_reconstruction))
        final_smb_reconstructions_dict['SMB_ASTER'] = np.vstack((final_smb_reconstructions_dict['SMB_ASTER'], glacier_calibrated_SMB_reconstruction))
    
    final_smb_reconstructions_dict['RGI_ID'].append(int(rgiID))
    final_smb_reconstructions_dict['GLIMS_ID'].append(glimsID)
    final_smb_reconstructions_dict['Name'].append(glacier_name)
    
    # We store the SMB calibrated reconstructions in individual files
    if(rgiID != 0):
        combined_ID = str(glimsID) + "_" + str(int(rgiID)) 
    else:
        combined_ID = str(glimsID) + "_0"
        
    store_file(glacier_calibrated_SMB_reconstruction, os.path.join(path_aster_reconstructions, 'smb'), "", "SMB", combined_ID, 1967, 2016)
    
    if not os.path.exists(os.path.join(path_aster_reconstructions, 'ensemble_smb')):
        os.makedirs(os.path.join(path_aster_reconstructions, 'ensemble_smb'))
    with open(os.path.join(os.path.join(path_aster_reconstructions, 'ensemble_smb'), ensemble_file), 'wb') as smb_f: 
        np.save(smb_f, glacier_ensemble)
        
#            print(glacier_rgi_id_idx)
    
final_smb_reconstructions_dict['SMB'] = np.asarray(final_smb_reconstructions_dict['SMB'])
final_smb_reconstructions_dict['SMB_ASTER'] = np.asarray(final_smb_reconstructions_dict['SMB_ASTER'])
final_smb_reconstructions_dict['RGI_ID'] = np.asarray(final_smb_reconstructions_dict['RGI_ID'])
final_smb_reconstructions_dict['GLIMS_ID'] = np.asarray(final_smb_reconstructions_dict['GLIMS_ID'])
final_smb_reconstructions_dict['Name'] = np.asarray(final_smb_reconstructions_dict['Name'])

    
# Transfer dictionary to xarray dataset       
ds_smb_reconstructions = xr.Dataset(data_vars={'SMB': (('RGI_ID', 'year'), final_smb_reconstructions_dict['SMB_ASTER']),
                                               'GLIMS_ID': final_smb_reconstructions_dict['GLIMS_ID'],
                                               'name':final_smb_reconstructions_dict['Name']},
                                            coords={'RGI_ID': final_smb_reconstructions_dict['RGI_ID'], 
                                                    'year': range(1967, 2016)},
                                            attrs={'Content': "French Alpine annual glacier-wide MB reconstructions (1967-2015) (m.w.e./yr) from the ALpine Parametrized Glacier Model. SMB contain the default MB reconstructions and SMB_ASTER contain the recalibrated MB reconstructions using ASTER-derived geodetic MB from Davaze et al. (2020) for the 2000-2015 sub-period.",
                                                   'Producer': "Jordi Bolibar", 'Co-producers': "Antoine Rabatel, Isabelle Gouttevin, Clovis Galiez",
                                                   'Affiliation': "Institute of Environmental Geosciences (University Grenoble Alpes / INRAE)"})

ds_smb_reconstructions.to_netcdf(os.path.join(path_dataset, 'french_alpine_glaciers_MB_reconstruction_1967_2015.nc'))

#import pdb; pdb.set_trace()  