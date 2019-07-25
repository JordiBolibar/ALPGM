# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:06:19 2018

@author: bolibarj
"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import math
from pathlib import Path

######   FILE PATHS    #######
    
# Folders     
workspace = str(Path(os.getcwd()).parent) + '\\'
#path_obs = 'C:\\Jordi\\PhD\\Data\\Obs\\'
path_smb = workspace + 'glacier_data\\smb\\'
path_glacier_evolution = workspace + 'glacier_data\\glacier_evolution\\'
path_glacier_2003_shapefiles = workspace + 'glacier_data\\glacier_shapefiles\\2003\\'
path_glacier_2015_shapefiles = workspace + 'glacier_data\\glacier_shapefiles\\2015\\'
path_glacier_ID_rasters = workspace + 'glacier_data\\glacier_rasters\\glacier_thickness_Huss\\thickness_tif\\'
path_glacier_DEM_rasters = workspace + 'glacier_data\\glacier_rasters\\glacier_thickness_Huss\\dem_tif\\'
path_glacier_evolution_DEM_rasters = path_glacier_DEM_rasters + 'glacier_evolution\\' 
path_glacier_evolution_ID_rasters = path_glacier_ID_rasters + 'glacier_evolution\\'
path_smb_simulations = path_smb + 'smb_simulations\\'
path_glacier_evolution_plots = path_glacier_evolution + 'plots\\SAFRAN\\1\\'
path_glacier_area = path_glacier_evolution + 'glacier_area\\SAFRAN\\1\\'
path_glacier_volume = path_glacier_evolution + 'glacier_volume\\SAFRAN\\1\\'
path_glacier_zmean = path_glacier_evolution + 'glacier_zmean\\SAFRAN\\1\\'
path_glacier_slope20 = path_glacier_evolution + 'glacier_slope20\\SAFRAN\\1\\'
path_glacier_melt_years = path_glacier_evolution + 'glacier_melt_years\\SAFRAN\\1\\'
path_glacier_w_errors = path_glacier_evolution + 'glacier_w_errors\\SAFRAN\\1\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 
path_safran_forcings = 'C:\\Jordi\\PhD\\Data\\SAFRAN-Nivo-2017\\'
path_smb_function = path_smb + 'smb_function\\'
global path_smb_function_safran 
path_smb_safran = path_smb + 'smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\smb\\'
path_area_safran = path_smb + 'smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\area\\'

path_smb_glaciers = np.asarray(os.listdir(path_smb_safran))
path_area_glaciers = np.asarray(os.listdir(path_area_safran))

# Iterate the different forcings
rcp_idx = 0

all_glacier_smb  = []
annual_avg_smb = np.zeros(2015-1967)

    
fig1, ax1 = plt.subplots()
ax1.set_ylabel('Glacier-wide SMB (m.w.e)')
ax1.set_xlabel('Year')
ax1.set_title("Annual glacier-wide SMB of all French alpine glaciers")

fig2, ax2 = plt.subplots()
ax2.set_ylabel('Cumulative glacier-wide SMB (m.w.e)')
ax2.set_xlabel('Year')
ax2.set_title("Cumulative glacier-wide SMB of all French alpine glaciers")


# Iterate all glaciers with the full simulated period
glacier_idx = 0
glaciers_not_2015 = 0
for path_smb, path_area in zip(path_smb_glaciers, path_area_glaciers):
#    import pdb; pdb.set_trace()
    area_glacier = genfromtxt(path_area_safran + path_area, delimiter=';')
    area_glacier = area_glacier[:,1].flatten()
    smb_glacier = genfromtxt(path_smb_safran + path_smb, delimiter=';')
    smb_glacier = smb_glacier[:,1].flatten()
    
    if(np.cumsum(smb_glacier)[-1] > 0):
        print(path_smb)
        print(np.sum(smb_glacier))
    
    all_glacier_smb.append(np.asarray(smb_glacier))
    
    if(smb_glacier.size < 49):
        nan_tail = np.zeros(2015-2003)
        nan_tail[:] = np.nan
        smb_glacier = np.concatenate((smb_glacier, nan_tail))
        area_glacier_i = area_glacier[-1]
        glaciers_not_2015 = glaciers_not_2015+1
    else:
        area_glacier_i = area_glacier[-15]
        
    if(area_glacier_i < 0.5):
        alpha_i = 0.5
    elif(area_glacier_i < 0.1):
        alpha_i = 0.1
    else:
        alpha_i = 1.0
    
    line1, = ax1.plot(range(1967, 2016), smb_glacier, linewidth=0.5)
    line2, = ax2.plot(range(1967, 2016), np.cumsum(smb_glacier), linewidth=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    
    line1.set_alpha(alpha_i)
    line1.set_alpha(alpha_i)
    
    year_idx = 0
    for annual_smb in annual_avg_smb:
        annual_smb = annual_smb + smb_glacier[year_idx]
        year_idx = year_idx+1
    
    glacier_idx = glacier_idx+1

all_glacier_smb = np.asarray(all_glacier_smb) 
annual_avg_smb = np.asarray(annual_avg_smb)
annual_avg_smb = annual_avg_smb/glacier_idx

print("Number of glaciers disappeared between 2003 and 2015: " + str(glaciers_not_2015))

#plt.legend()
plt.show()
#plt.gca().invert_yaxis()

###############################  PLOTS  ###############################################
#
##### AREA ###
###########################################
#nfigure = 1
##plt.figure(nfigure, figsize=(10, 20))
#plt.figure(nfigure)
##plt.subplot(211)
#
#avg_yearly_glacier_area_rcp_26[0] = (avg_yearly_glacier_area_rcp_26[0]/avg_yearly_glacier_area_rcp_26[0][0])*100.0 -100
#avg_yearly_glacier_area_rcp_26[1] = (avg_yearly_glacier_area_rcp_26[1]/avg_yearly_glacier_area_rcp_26[1][0])*100.0 -100
#mean_glacier_area_rcp_26 = (mean_glacier_area_rcp_26/mean_glacier_area_rcp_26[0])*100.0 -100
#
#avg_yearly_glacier_area_rcp_45[0] = (avg_yearly_glacier_area_rcp_45[0]/avg_yearly_glacier_area_rcp_45[0][0])*100.0 -100
#avg_yearly_glacier_area_rcp_45[1] = (avg_yearly_glacier_area_rcp_45[1]/avg_yearly_glacier_area_rcp_45[1][0])*100.0 -100
#mean_glacier_area_rcp_45 = (mean_glacier_area_rcp_45/mean_glacier_area_rcp_45[0])*100.0 -100
#
#avg_yearly_glacier_area_rcp_85[0] = (avg_yearly_glacier_area_rcp_85[0]/avg_yearly_glacier_area_rcp_85[0][0])*100.0 -100
#avg_yearly_glacier_area_rcp_85[1] = (avg_yearly_glacier_area_rcp_85[1]/avg_yearly_glacier_area_rcp_85[1][0])*100.0 -100
#mean_glacier_area_rcp_85 = (mean_glacier_area_rcp_85/mean_glacier_area_rcp_85[0])*100.0 -100
#
#plt.ylabel('Relative glacier area change (%)')
#plt.xlabel('Year')
#plt.title("Area evolution of French alpine glaciers with forcing uncertainties", y=1.05)
#axes1.fill_between(range(2014, 2100), avg_yearly_glacier_area_rcp_26[0], avg_yearly_glacier_area_rcp_26[1], facecolor = "navy", alpha=0.1)
#plt.plot(range(2014, 2100), mean_glacier_area_rcp_26, label = "RCP 2.6", color = "navy")
#axes1.fill_between(range(2014, 2100), avg_yearly_glacier_area_rcp_45[0], avg_yearly_glacier_area_rcp_45[1], facecolor = "forestgreen", alpha=0.1)
#plt.plot(range(2014, 2100), mean_glacier_area_rcp_45, label = "RCP 4.5", color = "forestgreen")
#axes1.fill_between(range(2014, 2100), avg_yearly_glacier_area_rcp_85[0], avg_yearly_glacier_area_rcp_85[1], facecolor = "red", alpha=0.1)
#plt.plot(range(2014, 2100), mean_glacier_area_rcp_85, label = "RCP 8.5", color = "red")
#
#plt.legend()
#plt.show(block=False)
##plt.gca().invert_yaxis()








