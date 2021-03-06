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

workspace = str(Path(os.getcwd()).parent)

path_smb = os.path.join(workspace, 'glacier_data', 'smb')

def init(hist_forcing, proj_forcing, simu_type, smb_model, cluster, static_geometry_mode, ASTER_calibration):
    print("Applying settings...")
    
    global ADAMONT_proj_filepaths
    global path_safran
    global path_adamont
    global use_cluster
    use_cluster = cluster
    if(use_cluster):
        path_adamont = '/bettik/jordibn/python/Data/ADAMONT/'
        path_safran = '/bettik/jordibn/python/Data/SAFRAN/'
    else:
        path_adamont = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\treated\\'
        path_safran = 'C:\\Jordi\\PhD\\Data\\SAFRAN-Nivo-2017\\'
    
    global static_geometry
    static_geometry = static_geometry_mode     
    
    global historical_forcing
    historical_forcing = hist_forcing
    global projection_forcing
    projection_forcing = proj_forcing
    global simulation_type
    simulation_type = simu_type
    ADAMONT_proj_filepaths = np.asarray(os.listdir(path_adamont))
    global current_ADAMONT_combination
    
    global path_ann
    global path_cv_ann
    global path_cv_lasso
    global path_ensemble_ann
    global smb_model_type
    
    global aster
    aster = ASTER_calibration
    
    global stacking_coefs
    
    # Save stacking model
    with open(os.path.join(path_smb,'smb_function', 'stacking_coeffs.txt'), 'rb') as model_lasso_gbl_f:
        stacking_coefs = np.load(model_lasso_gbl_f,  allow_pickle=True)
    
    if(smb_model == 'ann_no_weights'):
        if(simulation_type == 'historical'):
            path_ann = os.path.join(path_smb, 'ANN', 'LSYGO_soft')
        elif(simulation_type == 'future'):
            path_ann = os.path.join(path_smb, 'ANN', 'LSYGO_soft')
#            path_ann = os.path.join(path_smb, 'ANN', 'LSYGO')
#        path_ann = path_smb + 'ANN\\LOGO\\'
#        path_ann = path_smb + 'ANN\\LOYO\\'
        path_cv_ann = os.path.join(path_ann, 'CV')
        path_ensemble_ann = os.path.join(path_ann, 'ensemble')
        path_cv_lasso = os.path.join(path_smb, 'smb_function', 'Lasso_LSYGO_ensemble')
        smb_model_type = smb_model
    elif(smb_model == 'ann_weights'):
        path_ann = os.path.join(path_smb, 'ANN', 'LSYGO', 'weights')
        path_cv_ann = os.path.join(path_ann, 'CV')
        path_ensemble_ann = os.path.join(path_ann, 'ensemble')
        smb_model_type = smb_model
    elif(smb_model == 'lasso'):
        smb_model_type = smb_model 
        path_cv_lasso = os.path.join(path_smb, 'smb_function', 'Lasso_LSYGO_ensemble')
        
        # Set ANN paths for compatibility
        if(simulation_type == 'historical'):
            path_ann = os.path.join(path_smb, 'ANN', 'LSYGO_soft')
        elif(simulation_type == 'future'):
            path_ann = os.path.join(path_smb, 'ANN', 'LSYGO_hard')
        path_cv_ann = os.path.join(path_ann, 'CV')
        path_ensemble_ann = os.path.join(path_ann, 'ensemble')
    
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
        
def adamont_simulation(simulation_type, compute_projection_forcings, compute_evolution, counter_threshold, overwrite, filter_glacier):
    ###   ADAMONT PROJECTIONS   ###
    global current_ADAMONT_forcing_mean
    global current_ADAMONT_forcing_sum
    global current_ADAMONT_model_daymean 
    global current_ADAMONT_model_daysum 

    counter = 0
    forcing_threshold = 0
    if(simulation_type == 'future'):
        thickness_idx = 0
#        for thickness_idx in range(0,2):
        
        n_members = ADAMONT_proj_filepaths.size
        # Preload the ensemble SMB models to speed up simulations
        start = 0
        ensemble_SMB_models = glacier_evolution.preload_ensemble_SMB_models()
        
        for i in range(start, n_members, 2):
            if(forcing_threshold <= counter):
                current_ADAMONT_model_daymean = str(os.path.join(path_adamont, ADAMONT_proj_filepaths[i]))
                current_ADAMONT_model_daysum = str(os.path.join(path_adamont, ADAMONT_proj_filepaths[i+1]))
                current_ADAMONT_forcing_mean = os.path.join('projections', ADAMONT_proj_filepaths[i])
                current_ADAMONT_forcing_sum =  os.path.join('projections', ADAMONT_proj_filepaths[i+1])
                adamont_forcings.main(compute_projection_forcings)
                glacier_evolution.main(compute_evolution, ensemble_SMB_models, overwrite, counter_threshold, thickness_idx, filter_glacier)
            counter = counter+1

def glacier_simulation(simulation_type, counter_threshold, validate_SMB, compute_projection_forcings, compute_evolution, reconstruct, overwrite, filter_glacier):
    imp.reload(adamont_forcings)
    imp.reload(smb_validation)
    imp.reload(glacier_evolution)
    
    if(simulation_type == "future"):
        adamont_simulation(simulation_type, compute_projection_forcings, compute_evolution, counter_threshold, overwrite, filter_glacier)
    elif(simulation_type == "historical"):
        smb_validation.main(validate_SMB, reconstruct, smb_model_type)
        ensemble_SMB_models = []
        if(compute_evolution):
            # Preload the ensemble SMB models to speed up simulations
            ensemble_SMB_models = glacier_evolution.preload_ensemble_SMB_models()
#        for thickness_idx in range(0,3):
        # 0 = original ice thickness / 1 = 1.3*ice thickness /  2 =  0.7*ice thickness 
        glacier_evolution.main(compute_evolution, ensemble_SMB_models, overwrite, counter_threshold, 0, filter_glacier) # thickness idx = 0 by default
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
        
    
