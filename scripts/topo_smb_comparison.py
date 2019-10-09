# -*- coding: utf-8 -*-
"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

COMPARISON OF SIMULATIONS BETWEEN TOPO PREDICTORS COMMING FROM MULTITEMPORAL
INVENTORIES AND F19 RASTER FILES

"""

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

workspace = Path(os.getcwd()).parent 
root = str(workspace.parent) + '\\'
workspace = str(workspace) + '\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 

# Training SMB and topo data
path_training_data = workspace + 'glacier_data\\glacier_evolution\\training\\'
path_training_glacier_info = path_training_data + 'glacier_info\\'

# Glacier evolution SMB and topo data
path_glacier_evolution = workspace + 'glacier_data\\glacier_evolution\\BACKUP\\'
path_glacier_area = path_glacier_evolution + 'glacier_area\\SAFRAN\\1\\'
path_glacier_volume = path_glacier_evolution + 'glacier_volume\\SAFRAN\\1\\'
path_glacier_zmean = path_glacier_evolution + 'glacier_zmean\\SAFRAN\\1\\'
path_glacier_slope20 = path_glacier_evolution + 'glacier_slope20\\SAFRAN\\1\\'
path_smb = workspace + 'glacier_data\\smb\\'
path_smb_simulations = path_smb + 'smb_simulations\\SAFRAN\\BACKUP\\1\\'


#######################################################################
#######################################################################

overall_plots = True

SMB_raw = genfromtxt(path_smb + 'SMB_raw_extended.csv', delimiter=';', dtype=float)
glims_rabatel = genfromtxt(path_glims + 'GLIMS_Rabatel_30_2015.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('slope20', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8'), ('slope20_evo', '<f8'),])        

train_SMBs, evo_SMBs, ref_SMBs = [],[],[]
train_slopes, evo_slopes, evo_avg_slopes = [],[],[]
train_zmeans, evo_zmeans = [],[]
train_areas, evo_areas = [],[]

fig_idx = 1
for glacier, ref_SMB in zip(glims_rabatel[:-1], SMB_raw):
    glacier_name = str(glacier['Glacier'].decode('ascii'))
    print("\nGlacier: " + glacier_name)
#    print("Area: " + str(glacier['Area']))
    area_ref = glacier['Area']
    glims_ID = glacier['GLIMS_ID'].decode('ascii')
    
    print("glims_ID: " + str(glims_ID))
    
    # Retrieve simulated data
    # Training    
    with open(path_training_glacier_info + 'glacier_info_' + str(glims_ID), 'rb') as train_gl_info_f:
            train_glacier_info = np.load(train_gl_info_f)[()]
    train_smb = genfromtxt(path_training_data + 'SMB\\' + str(glims_ID) + '_SMB.csv', delimiter=";")[20:, :]
    train_slope20 = genfromtxt(path_training_data + 'slope20\\' + str(glims_ID) + '_Slope_20.csv', delimiter=";")[20:, :]
    
#    import pdb; pdb.set_trace()
    
    train_zmean = train_glacier_info['mean_altitude'][19:-1]
    train_area = train_glacier_info['area'][19:-1]
    
    # Glacier evolution 
    evo_smb = genfromtxt(path_smb_simulations + str(glims_ID) + '_simu_SMB.csv', delimiter=";")[:-1,:]
    evo_zmean = genfromtxt(path_glacier_zmean + str(glims_ID) + '_zmean.csv', delimiter=";")[:-1,:]
    evo_slope20 = genfromtxt(path_glacier_slope20 + str(glims_ID) + '_slope20.csv', delimiter=";")[:-1,:]
    evo_area = genfromtxt(path_glacier_area + str(glims_ID) + '_area.csv', delimiter=";")[:-1,:]

    print("\nTraining slope20: " + str(train_slope20))
    print("Glacier evolution slope20: " + str(evo_slope20))
    
    print("\nTraining zmean: " + str(train_zmean))
    print("Glacier evolution zmean: " + str(evo_zmean[:,1]))
    
    print("\nTraining area: " + str(train_area))
    print("Glacier evolution area: " + str(evo_area[:,1]))
    
    print("\nTraining SMB: " + str(train_smb))
    print("Glacier evolution SMB: " + str(evo_smb))
    
    finite_mask_train = np.isfinite(train_smb[:,1])
    finite_mask_ref = np.isfinite(ref_SMB[-11:])
    
    # Store all data for all glaciers together
    train_SMBs = np.concatenate((train_SMBs, train_smb[:,1][finite_mask_train]), axis=None)
    evo_SMBs = np.concatenate((evo_SMBs, evo_smb[:,1][finite_mask_train]), axis=None)
    ref_SMBs = np.concatenate((ref_SMBs, ref_SMB[-11:][finite_mask_train]), axis=None)
    
    train_slopes = np.concatenate((train_slopes, train_slope20[:,1]), axis=None)
    train_zmeans = np.concatenate((train_zmeans, train_zmean), axis=None)
    train_areas = np.concatenate((train_areas, train_area), axis=None)
    evo_slopes = np.concatenate((evo_slopes, evo_slope20[:,1]), axis=None)
    evo_zmeans = np.concatenate((evo_zmeans, evo_zmean[:,1]), axis=None)
    evo_areas = np.concatenate((evo_areas, evo_area[:,1]), axis=None)
    
    evo_avg_slopes.append(np.mean(evo_slope20[:,1]))
    
    if(train_smb.size == evo_smb.size):    
        ####  PLOTS  ########
        mins = np.array([train_smb[:,1][finite_mask_train].min(), evo_smb[:,1].min(), ref_SMB[-11:][finite_mask_ref].min()])
        maxs = np.array([train_smb[:,1][finite_mask_train].max(), evo_smb[:,1].max(), ref_SMB[-11:][finite_mask_ref].max()])
        
#        import pdb; pdb.set_trace()
        
#        plt.subplot(1,2,1)
#        plt.title("Training SMB", fontsize=20)
#        plt.plot(train_smb[:,0], train_smb[:,1])
#        plt.plot(train_smb[:,0], ref_SMB[-11:])
#        plt.ylim(mins.min(), maxs.max())
#        fig_idx = fig_idx+1
#        
#        
#        plt.subplot(1,2,2)
#        plt.title("Glacier evolution SMB", fontsize=20)
#        plt.plot(evo_smb[:,0], evo_smb[:,1])
#        plt.plot(evo_smb[:,0], ref_SMB[-11:])
#        plt.ylim(mins.min(), maxs.max())
##        plt.show()
#        fig_idx = fig_idx+1
#        
#        plt.figure(fig_idx, figsize=(6,6))
#        plt.title(glacier_name + " comparison", fontsize=20)
#        plt.ylabel('SMB modeled with multitemporal glacier inventories (m.w.e)', fontsize=16)
#        plt.xlabel('SMB modeled in glacier evolution component (m.w.e)', fontsize=16)
#        plt.scatter(evo_smb[:, 1], train_smb[:, 1], s=100)
#        plt.tick_params(labelsize=14)
#        lineStart = evo_smb[:, 1].min() 
#        lineEnd = evo_smb[:, 1].max()  
#        plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
#        plt.xlim(lineStart, lineEnd)
#        plt.ylim(lineStart, lineEnd)
#        plt.clim(0,0.4)
#        fig_idx = fig_idx+1
        
#        plt.show()
    #    
    #    import pdb; pdb.set_trace()
    #    
    #    plt.figure(fig_idx, figsize=(6,6))
    #    plt.title("Training vs Glacier evolution z mean", fontsize=20)
    #    plt.ylabel('Zmean modeled with multitemporal glacier inventories (m)', fontsize=16)
    #    plt.xlabel('Zmean modeled in glacier evolution component (m)', fontsize=16)
    #    plt.scatter(train_zmean, evo_zmean[:,1], s=100)
    #    plt.tick_params(labelsize=14)
    #    lineStart = evo_zmean.min() 
    #    lineEnd = evo_zmean.max()  
    #    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
    #    plt.xlim(lineStart, lineEnd)
    #    plt.ylim(lineStart, lineEnd)
    #    plt.clim(0,0.4)
    #    plt.show()
    
# End of loop
    
train_SMBs = np.asarray(train_SMBs)
evo_SMBs = np.asarray(evo_SMBs)
ref_SMBs = np.asarray(ref_SMBs)

train_slopes = np.asarray(train_slopes)
evo_slopes = np.asarray(evo_slopes)
train_zmeans = np.asarray(train_zmeans)
evo_zmeans = np.asarray(evo_zmeans)
train_areas = np.asarray(train_areas)
evo_areas = np.asarray(evo_areas)

evo_avg_slopes = np.asarray(evo_avg_slopes)

# SMB performance results
# Train SMB
train_RMSE = math.sqrt(mean_squared_error(ref_SMBs, train_SMBs))
train_r2 = r2_score(ref_SMBs, train_SMBs)

# Glacier evolution SMB
evo_RMSE = math.sqrt(mean_squared_error(ref_SMBs, evo_SMBs))
evo_r2 = r2_score(ref_SMBs, evo_SMBs)

mae_train_evo_smb = np.abs(train_SMBs - evo_SMBs).mean()
mae_train_evo_slope = np.abs((train_slopes - evo_slopes)).mean()
mae_train_evo_zmeans = np.abs((train_zmeans - evo_zmeans)).mean()
mae_train_evo_areas = np.abs((train_areas - evo_areas)).mean()

print("\nRMSE train SMB: " + str(train_RMSE))
print("RMSE evo SMB: " + str(evo_RMSE))

print("\nr2 train SMB: " + str(train_r2))
print("r2 evo SMB: " + str(str(evo_r2)))

print("\nMAE between evo and train SMB: " + str(mae_train_evo_smb) + " (m.w.e.)")
print("\nMAE between evo and train slope20: " + str(mae_train_evo_slope) + " (°)")
print("\nMAE between evo and train Z mean: " + str(mae_train_evo_zmeans) + " (m)")
print("\nMAE between evo and train Area: " + str(mae_train_evo_areas) + " (km2)")

#plt.show()

#import pdb; pdb.set_trace() 

if(overall_plots):

    # SMB scatter plot
    plt.figure(fig_idx, figsize=(7,7))
    plt.ylabel('SMB multitemporal glacier inventories (m.w.e)', fontsize=16)
    plt.xlabel('SMB glacier evolution component (m.w.e)', fontsize=16)
    plt.scatter(evo_SMBs, train_SMBs, s=100, alpha=0.5)
    plt.tick_params(labelsize=14)
    lineStart = evo_SMBs.min() 
    lineEnd = evo_SMBs.max()  
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.clim(0,0.4)
    #plt.show()
    fig_idx = fig_idx+1
    
    # Slope 20% scatter plot
    plt.figure(fig_idx, figsize=(7,7))
    plt.ylabel('Slope multitemporal glacier inventories (m.w.e)', fontsize=16)
    plt.xlabel('Slope glacier evolution component (m.w.e)', fontsize=16)
    plt.scatter(evo_slopes, train_slopes, s=100, alpha=0.5, c='C2')
    plt.tick_params(labelsize=14)
    lineStart = evo_slopes.min() 
    lineEnd = evo_slopes.max()  
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.clim(0,0.4)
    #plt.show()
    fig_idx = fig_idx+1
    
    # Zmean scatter plot
    plt.figure(fig_idx, figsize=(7,7))
    plt.ylabel('Zmean multitemporal glacier inventories (m.w.e)', fontsize=16)
    plt.xlabel('Zmean glacier evolution component (m.w.e)', fontsize=16)
    plt.scatter(evo_zmeans, train_zmeans, s=100, alpha=0.5, c='C3')
    plt.tick_params(labelsize=14)
    lineStart = evo_zmeans.min() 
    lineEnd = evo_zmeans.max()  
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.clim(0,0.4)
    #plt.show()
    fig_idx = fig_idx+1
    
    # Area scatter plot
    plt.figure(fig_idx, figsize=(7,7))
    plt.ylabel('Area multitemporal glacier inventories (m.w.e)', fontsize=16)
    plt.xlabel('Area glacier evolution component (m.w.e)', fontsize=16)
    plt.scatter(evo_areas, train_areas, s=100, alpha=0.5, c='C4')
    plt.tick_params(labelsize=14)
    lineStart = evo_areas.min() 
    lineEnd = evo_areas.max()  
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.clim(0,0.4)
    plt.show()
    fig_idx = fig_idx+1
    
    #import pdb; pdb.set_trace() 