# -*- coding: utf-8 -*-

"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

ALPGM INTERFACE SETTINGS AND GLOBAL VARIABLES

"""

import os
import imp
import numpy as np
import delta_h_alps
import safran_forcings
import adamont_forcings
import smb_model_training
import smb_validation
import glacier_evolution

from pathlib import Path

workspace = str(Path(os.getcwd()).parent) + '\\'
path_adamont = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\treated\\'
#path_adamont = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\FORCING_ADAMONT_IGE_BERGER\\normal\\'
#path_adamont = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\FORCING_ADAMONT_IGE_BERGER\\INERIS\\'
#path_adamont = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\FORCING_ADAMONT_IGE_BERGER\\HIRHAM5\\'
#path_adamont = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\FORCING_ADAMONT_IGE_BERGER\\projections\\'
#path_adamont = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\FORCING_ADAMONT_IGE_BERGER\\subset_AGU\\projections\\'
path_smb = workspace + 'glacier_data\\smb\\'

def init(hist_forcing, proj_forcing, simu_type, smb_model):
    print("Applying settings...")
    
    global historical_forcing
    historical_forcing = hist_forcing
    global projection_forcing
    projection_forcing = proj_forcing
    global simulation_type
    simulation_type = simu_type
    global ADAMONT_proj_filepaths
    ADAMONT_proj_filepaths = np.asarray(os.listdir(path_adamont))
    global current_ADAMONT_combination
    
    global path_ann
    global path_cv_ann
    global smb_model_type
    if(smb_model == 'ann_no_weights'):
        path_ann = path_smb + 'ANN\\LSYGO\\no_weights\\'
#        path_ann = path_smb + 'ANN\\LOGO\\no_weights\\'
#        path_ann = path_smb + 'ANN\\LOYO\\no_weights\\'
        path_cv_ann = path_ann + 'CV\\'
        smb_model_type = smb_model
    elif(smb_model == 'ann_weights'):
        path_ann = path_smb + 'ANN\\LSYGO\\weights\\'
        path_cv_ann = path_ann + 'CV\\'
        smb_model_type = smb_model
    elif(smb_model == 'lasso'):
        smb_model_type = smb_model 
    
# Auxiliary main functions

def train_smb_model(historical_forcing, compute_forcing, train_model):
    imp.reload(safran_forcings)
    imp.reload(smb_model_training)

    if(historical_forcing == "SAFRAN"):
        safran_forcings.main(compute_forcing)
    else:
        print("\n[ERROR] Wrong historical forcing!")
        
    smb_model_training.main(train_model)
    
def glacier_parameterized_functions(compute, overwrite):
    imp.reload(delta_h_alps)
    delta_h_alps.main(compute, overwrite)
        
def adamont_simulation(simulation_type, compute_projection_forcings, compute_evolution, counter_threshold, overwrite):
    ###   ADAMONT PROJECTIONS   ###
    global current_ADAMONT_forcing_mean
    global current_ADAMONT_forcing_sum
    global current_ADAMONT_model_daymean 
    global current_ADAMONT_model_daysum 
    counter = 0
    forcing_threshold = 0
    if(simulation_type == 'future'):
#        for thickness_idx in range(0,3):
        thickness_idx = 0
#        for thickness_idx in range(0,2):
        for i in range(0, ADAMONT_proj_filepaths.size, 2):
            if(forcing_threshold <= counter):
                current_ADAMONT_model_daymean = str(ADAMONT_proj_filepaths[i])
                current_ADAMONT_model_daysum = str(ADAMONT_proj_filepaths[i+1])
                current_ADAMONT_forcing_mean = 'projections\\' + ADAMONT_proj_filepaths[i]
                current_ADAMONT_forcing_sum =  'projections\\' + ADAMONT_proj_filepaths[i+1]
                adamont_forcings.main(compute_projection_forcings)
                glacier_evolution.main(compute_evolution, overwrite, counter_threshold, thickness_idx)
            counter = counter+1

def glacier_simulation(simulation_type, counter_threshold, validate_SMB, compute_projection_forcings, compute_evolution, reconstruct, overwrite):
    imp.reload(adamont_forcings)
    imp.reload(smb_validation)
    imp.reload(glacier_evolution)
    
    if(simulation_type == "future"):
        adamont_simulation(simulation_type, compute_projection_forcings, compute_evolution, counter_threshold, overwrite)
    elif(simulation_type == "historical"):
        smb_validation.main(validate_SMB, reconstruct)
#        for thickness_idx in range(0,3):
        # 0 = original ice thickness / 1 = 1.3*ice thickness /  2 =  0.7*ice thickness 
        glacier_evolution.main(compute_evolution, overwrite, counter_threshold, 0) # thickness idx = 0 by default
    else:
        print("\n[ERROR] Wrong type of projection!")
        
def simulation_settings(projection):
    if(projection):
        historical_forcing = "SAFRAN"
        projection_forcing = "ADAMONT"
        simulation_type = "future"
    else:
        historical_forcing = "SAFRAN"
        projection_forcing = "SAFRAN"
        simulation_type = "historical"
    
    return historical_forcing, projection_forcing, simulation_type
        
    
