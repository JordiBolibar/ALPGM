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

######   FILE PATHS    #######
    
# Folders     
workspace = 'C:\\Jordi\\PhD\\Python\\'
#path_obs = 'C:\\Jordi\\PhD\\Data\\Obs\\'
path_smb = workspace + 'ALPGM\\glacier_data\\smb\\'
#path_glacier_evolution = workspace + 'ALPGM\\glacier_data\\glacier_evolution\\'
path_glacier_evolution = 'C:\\Jordi\\PhD\\Simulations\\AGU 2018\\'
path_glacier_2003_shapefiles = workspace + 'ALPGM\\glacier_data\\glacier_shapefiles\\2003\\'
path_glacier_2015_shapefiles = workspace + 'ALPGM\\glacier_data\\glacier_shapefiles\\2015\\'
path_glacier_ID_rasters = workspace + 'ALPGM\\glacier_data\\glacier_rasters\\glacier_thickness_Huss\\thickness_tif\\'
path_glacier_DEM_rasters = workspace + 'ALPGM\\glacier_data\\glacier_rasters\\glacier_thickness_Huss\\dem_tif\\'
path_glacier_evolution_DEM_rasters = path_glacier_DEM_rasters + 'glacier_evolution\\' 
path_glacier_evolution_ID_rasters = path_glacier_ID_rasters + 'glacier_evolution\\'
path_smb_simulations = path_smb + 'smb_simulations\\'
path_glacier_evolution_plots = path_glacier_evolution + 'plots\\'
path_glacier_area = path_glacier_evolution + 'glacier_area\\'
path_glacier_volume = path_glacier_evolution + 'glacier_volume\\'
path_glacier_zmean = path_glacier_evolution + 'glacier_zmean\\'
path_glacier_slope20 = path_glacier_evolution + 'glacier_slope20\\'
path_glacier_melt_years = path_glacier_evolution + 'glacier_melt_years\\'
path_glacier_w_errors = path_glacier_evolution + 'glacier_w_errors\\'
path_glims = workspace + 'ALPGM\\glacier_data\\GLIMS\\' 
path_safran_forcings = 'C:\\Jordi\\PhD\\Data\\SAFRAN-Nivo-2017\\'
path_smb_function = path_smb + 'smb_function\\'
global path_smb_function_safran 
path_smb_function_safran = path_smb + 'smb_function\\SAFRAN\\'
global path_smb_function_adamont
path_smb_function_adamont = path_smb + 'smb_function\\ADAMONT\\'
path_adamont_forcings = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\FORCING_ADAMONT_IGE_BERGER\\'

#### We fetch all the data from the simulations

fig1, axes1 = plt.subplots()  
fig2, axes2 = plt.subplots()  
fig3, axes3 = plt.subplots()
fig4, axes4 = plt.subplots()  
fig5, axes5 = plt.subplots()  


path_area_forcings = np.asarray(os.listdir(path_glacier_area + "projections\\"))
path_melt_years_forcings = np.asarray(os.listdir(path_glacier_melt_years + "projections\\"))
path_slope20_forcings = np.asarray(os.listdir(path_glacier_slope20 + "projections\\"))
path_volume_forcings = np.asarray(os.listdir(path_glacier_volume + "projections\\"))
path_errors_forcings = np.asarray(os.listdir(path_glacier_w_errors + "projections\\"))
path_zmean_forcings = np.asarray(os.listdir(path_glacier_zmean + "projections\\"))

# Iterate the different forcings
rcp_idx = 0
# Arrays of the data for all glaciers for the whole period for the current forcing
avg_yearly_glacier_area_rcp_26, avg_yearly_glacier_volume_rcp_26, avg_yearly_glacier_zmean_rcp_26 = [],[],[]
avg_yearly_glacier_area_rcp_26_13, avg_yearly_glacier_volume_rcp_26_13, avg_yearly_glacier_zmean_rcp_26_13 = [],[],[]
avg_yearly_glacier_area_rcp_26_07, avg_yearly_glacier_volume_rcp_26_07, avg_yearly_glacier_zmean_rcp_26_07 = [],[],[]

avg_yearly_glacier_area_rcp_45, avg_yearly_glacier_volume_rcp_45, avg_yearly_glacier_zmean_rcp_45 = [],[],[]
avg_yearly_glacier_area_rcp_45_13, avg_yearly_glacier_volume_rcp_45_13, avg_yearly_glacier_zmean_rcp_45_13 = [],[],[]
avg_yearly_glacier_area_rcp_45_07, avg_yearly_glacier_volume_rcp_45_07, avg_yearly_glacier_zmean_rcp_45_07 = [],[],[]

avg_yearly_glacier_area_rcp_85, avg_yearly_glacier_volume_rcp_85, avg_yearly_glacier_zmean_rcp_85 = [],[],[]
avg_yearly_glacier_area_rcp_85_13, avg_yearly_glacier_volume_rcp_85_13, avg_yearly_glacier_zmean_rcp_85_13 = [],[],[]
avg_yearly_glacier_area_rcp_85_07, avg_yearly_glacier_volume_rcp_85_07, avg_yearly_glacier_zmean_rcp_85_07 = [],[],[]

for path_forcing_area, path_forcing_melt_years, path_forcing_slope20, path_forcing_volume, path_forcing_zmean, path_forcing_errors in zip(path_area_forcings, path_melt_years_forcings, path_slope20_forcings, path_volume_forcings, path_zmean_forcings, path_errors_forcings):
    path_area_glaciers = np.asarray(os.listdir(path_glacier_area + "projections\\" + path_forcing_area))
    path_melt_years_glaciers = np.asarray(os.listdir(path_glacier_melt_years + "projections\\" + path_forcing_melt_years))
    path_slope20_glaciers = np.asarray(os.listdir(path_glacier_slope20 + "projections\\" + path_forcing_slope20))
    path_volume_glaciers = np.asarray(os.listdir(path_glacier_volume + "projections\\" + path_forcing_volume))
    path_errors_glaciers = np.asarray(os.listdir(path_glacier_w_errors + "projections\\" + path_forcing_errors))
    path_zmean_glaciers = np.asarray(os.listdir(path_glacier_zmean + "projections\\" + path_forcing_zmean))
    
    print("\nCurrent forcing: " + path_forcing_area + "\n")
    
    # Arrays of the data for all glaciers for the whole period for the current forcing
    all_glacier_areas, all_glacier_melt_years, all_glacier_slope20, all_glacier_volumes, all_glacier_errors, all_glacier_zmean = [],[],[],[],[],[]
    all_glacier_areas_07, all_glacier_melt_years_07, all_glacier_slope20_07, all_glacier_volumes_07, all_glacier_errors_07, all_glacier_zmean_07 = [],[],[],[],[],[]
    all_glacier_areas_13, all_glacier_melt_years_13, all_glacier_slope20_13, all_glacier_volumes_13, all_glacier_errors_13, all_glacier_zmean_13 = [],[],[],[],[],[]

#    print("path_area_glaciers: " + str(path_area_glaciers))
##    print("path_melt_years_glaciers: " + str(path_melt_years_glaciers.size))
##    print("path_slope20_glaciers: " + str(path_slope20_glaciers.size))
#    print("path_volume_glaciers: " + str(path_volume_glaciers.size))
##    print("path_errors_glaciers: " + str(path_errors_glaciers.size))
#    print("path_zmean_glaciers: " + str(path_zmean_glaciers.size))
    scale_idx = 0
    # Iterate all glaciers with the full simulated period
    for path_area_scaled, path_volume_scaled, path_zmean_scaled in zip(path_area_glaciers, path_volume_glaciers, path_zmean_glaciers):
        
        path_area_glaciers_scaled = np.asarray(os.listdir(path_glacier_area + "projections\\" + path_forcing_area + "\\" + path_area_scaled))
        path_volume_glaciers_scaled = np.asarray(os.listdir(path_glacier_volume + "projections\\" + path_forcing_volume + "\\" +path_volume_scaled))
        path_zmean_glaciers_scaled = np.asarray(os.listdir(path_glacier_zmean + "projections\\" + path_forcing_zmean + "\\" + path_zmean_scaled))
#        print("path_area_glaciers_scaled: " + str(path_area_glaciers_scaled))
        for path_area, path_volume, path_zmean in zip(path_area_glaciers_scaled, path_volume_glaciers_scaled, path_zmean_glaciers_scaled):
#            print("path_area: " + str(path_area))
#            print(path_glacier_area + "projections\\" + path_forcing_area + "\\" + path_area_scaled + "\\" +  path_area)
            
            area_glacier = genfromtxt(path_glacier_area + "projections\\" + path_forcing_area + "\\" + path_area_scaled + "\\" + path_area, delimiter=';')
#            print("area_glacier: " + str(area_glacier))
            volume_glacier = genfromtxt(path_glacier_volume + "projections\\" + path_forcing_volume + "\\" + path_volume_scaled + "\\" + path_volume, delimiter=';')
            zmean_glacier = genfromtxt(path_glacier_zmean + "projections\\" + path_forcing_zmean + "\\" + path_zmean_scaled + "\\" + path_zmean, delimiter=';')
            
            if(scale_idx == 0):
                all_glacier_areas_07.append(np.asarray(area_glacier))
                all_glacier_volumes_07.append(np.asarray(volume_glacier))
                all_glacier_zmean_07.append(np.asarray(zmean_glacier))
            elif(scale_idx == 1):
                all_glacier_areas.append(np.asarray(area_glacier))
                all_glacier_volumes.append(np.asarray(volume_glacier))
                all_glacier_zmean.append(np.asarray(zmean_glacier))
            elif(scale_idx == 2):
                all_glacier_areas_13.append(np.asarray(area_glacier))
                all_glacier_volumes_13.append(np.asarray(volume_glacier))
                all_glacier_zmean_13.append(np.asarray(zmean_glacier))
                 
        scale_idx = scale_idx+1
    
    all_glacier_areas = np.asarray(all_glacier_areas) 
    all_glacier_volumes = np.asarray(all_glacier_volumes)
    all_glacier_zmean = np.asarray(all_glacier_zmean)
    all_glacier_areas_07 = np.asarray(all_glacier_areas_07)
    all_glacier_volumes_07 = np.asarray(all_glacier_volumes_07)
    all_glacier_zmean_07 = np.asarray(all_glacier_zmean_07)
    all_glacier_areas_13 = np.asarray(all_glacier_areas_13)
    all_glacier_volumes_13 = np.asarray(all_glacier_volumes_13)
    all_glacier_zmean_13 = np.asarray(all_glacier_zmean_13)
    
#    print("all_glacier_areas: " + str(all_glacier_areas))
#    print("all_glacier_volumes: " + str(all_glacier_volumes))
    
    avg_yearly_area, avg_yearly_volume, avg_yearly_zmean = np.zeros(86), np.zeros(86), np.empty(86)
    avg_yearly_area_07, avg_yearly_volume_07, avg_yearly_zmean_07 = np.zeros(86), np.zeros(86), np.empty(86)
    avg_yearly_area_13, avg_yearly_volume_13, avg_yearly_zmean_13 = np.zeros(86), np.zeros(86), np.empty(86)
    for glacier_area, glacier_volume, glacier_zmean in zip(all_glacier_areas, all_glacier_volumes, all_glacier_zmean):
        year_idx = 0
        for area_y, volume_y, zmean_y in zip(glacier_area, glacier_volume, glacier_zmean):
            if(not math.isnan(area_y)):
                avg_yearly_area[year_idx] = avg_yearly_area[year_idx] + area_y
            if(not math.isnan(volume_y)):
                avg_yearly_volume[year_idx] = avg_yearly_volume[year_idx] + volume_y
            if(not math.isnan(zmean_y)):
                avg_yearly_zmean[year_idx] = avg_yearly_zmean[year_idx] + zmean_y
            if(np.all(glacier_zmean == all_glacier_zmean[-1])):
    #                print("avg_yearly_zmean[year_idx]: " + str(avg_yearly_zmean[year_idx]))
                avg_yearly_zmean[year_idx] = np.nanmean(avg_yearly_zmean[year_idx])
    #                print("avg_yearly_zmean[year_idx]: " + str(avg_yearly_zmean[year_idx]))
            year_idx = year_idx+1
    
    for glacier_area_07, glacier_volume_07, glacier_zmean_07 in zip(all_glacier_areas_07, all_glacier_volumes_07, all_glacier_zmean_07):
        year_idx = 0
        for area_y_07, volume_y_07, zmean_y_07 in zip(glacier_area_07, glacier_volume_07, glacier_zmean_07):
            if(not math.isnan(area_y_07)):
                avg_yearly_area_07[year_idx] = avg_yearly_area_07[year_idx] + area_y_07
            if(not math.isnan(volume_y_07)):
                avg_yearly_volume_07[year_idx] = avg_yearly_volume_07[year_idx] + volume_y_07
            if(not math.isnan(zmean_y_07)):
                avg_yearly_zmean_07[year_idx] = avg_yearly_zmean_07[year_idx] + zmean_y_07
            if(np.all(glacier_zmean_07 == all_glacier_zmean_07[-1])):
#                print("avg_yearly_zmean[year_idx]: " + str(avg_yearly_zmean[year_idx]))
                avg_yearly_zmean_07[year_idx] = np.nanmean(avg_yearly_zmean_07[year_idx])
#                print("avg_yearly_zmean[year_idx]: " + str(avg_yearly_zmean[year_idx]))
            year_idx = year_idx+1
        
    for glacier_area_13, glacier_volume_13, glacier_zmean_13 in zip(all_glacier_areas_13, all_glacier_volumes_13, all_glacier_zmean_13):
        year_idx = 0
        for area_y_13, volume_y_13, zmean_y_13 in zip(glacier_area_13, glacier_volume_13, glacier_zmean_13):
            if(not math.isnan(area_y_13)):
                avg_yearly_area_13[year_idx] = avg_yearly_area_13[year_idx] + area_y_13
            if(not math.isnan(volume_y_13)):
                avg_yearly_volume_13[year_idx] = avg_yearly_volume_13[year_idx] + volume_y_13
            if(not math.isnan(zmean_y_13)):
                avg_yearly_zmean_13[year_idx] = avg_yearly_zmean_13[year_idx] + zmean_y_13
            if(np.all(glacier_zmean_13 == all_glacier_zmean_13[-1])):
#                print("avg_yearly_zmean[year_idx]: " + str(avg_yearly_zmean[year_idx]))
                avg_yearly_zmean_13[year_idx] = np.nanmean(avg_yearly_zmean_13[year_idx])
#                print("avg_yearly_zmean[year_idx]: " + str(avg_yearly_zmean[year_idx]))
            year_idx = year_idx+1
            
            
            
            # If needed, in the last iteration we compute the average yearly value for all glaciers
#            if(np.all(glacier_area == all_glacier_areas[-1])):
##                print("Computing avg area")
#                avg_yearly_area[year_idx] = np.nanmean(avg_yearly_area[year_idx])
#            if(np.all(glacier_volume == all_glacier_volumes[-1])):
##                print("Computing avg volume")
#                avg_yearly_volume[year_idx] = np.nanmean(avg_yearly_volume[year_idx])
           
#    print("avg_yearly_area: " + str(avg_yearly_area))
#    print("avg_yearly_volume: " + str(avg_yearly_volume))
        
        
    ### PLOTS FOR EACH VARIABLE FOR EACH FORCING ###
    
    if(rcp_idx < 1):
        print("Adding 2.6 forcing")
        avg_yearly_glacier_area_rcp_26.append(avg_yearly_area)
        avg_yearly_glacier_area_rcp_26_13.append(avg_yearly_area_13)
        avg_yearly_glacier_area_rcp_26_07.append(avg_yearly_area_07)
        avg_yearly_glacier_volume_rcp_26.append(avg_yearly_volume)
        avg_yearly_glacier_volume_rcp_26_13.append(avg_yearly_volume_13)
        avg_yearly_glacier_volume_rcp_26_07.append(avg_yearly_volume_07)
        rcp = "RCP 2.6"
        color = 'navy'
    elif(rcp_idx < 2):
        print("Adding 4.5 forcing")
        avg_yearly_glacier_area_rcp_45.append(avg_yearly_area)
        avg_yearly_glacier_area_rcp_45_13.append(avg_yearly_area_13)
        avg_yearly_glacier_area_rcp_45_07.append(avg_yearly_area_07)
        avg_yearly_glacier_volume_rcp_45.append(avg_yearly_volume)
        avg_yearly_glacier_volume_rcp_45_13.append(avg_yearly_volume_13)
        avg_yearly_glacier_volume_rcp_45_07.append(avg_yearly_volume_07)
        rcp = "RCP 4.5"
        color = 'forestgreen'
    elif(rcp_idx < 3):
        print("Adding 8.5 forcing")
        avg_yearly_glacier_area_rcp_85.append(avg_yearly_area)
        avg_yearly_glacier_area_rcp_85_13.append(avg_yearly_area_13)
        avg_yearly_glacier_area_rcp_85_07.append(avg_yearly_area_07)
        avg_yearly_glacier_volume_rcp_85.append(avg_yearly_volume)
        avg_yearly_glacier_volume_rcp_85_13.append(avg_yearly_volume_13)
        avg_yearly_glacier_volume_rcp_85_07.append(avg_yearly_volume_07)
        rcp = "RCP 8.5"
        color = 'red'
        
#    print("avg_yearly_area_07.shape: " + str(avg_yearly_area_07.shape))
#    print("avg_yearly_area_13.shape: " + str(avg_yearly_area_13.shape))
#    print("avg_yearly_area.shape: " + str(avg_yearly_area.shape))
        
    #### AREA ###
    nfigure = 1
    #plt.figure(nfigure, figsize=(10, 20))
    plt.figure(nfigure)
    #plt.subplot(211)
    
    plt.ylabel('Total glacier area (km2)')
    plt.xlabel('Year')
    plt.title("Total area of French alpine glaciers")
#    axes1.fill_between(range(2014, 2100), avg_yearly_area_07, avg_yearly_area_13, facecolor = color, alpha=0.1)
#    plt.plot(range(2014, 2100), avg_yearly_area, color = color, linewidth=0.5)
    
    plt.legend()
    plt.show(block=False)
    #plt.gca().invert_yaxis()
    
    ### VOLUME ###
    nfigure = 3
    #plt.figure(nfigure, figsize=(10, 20))
    plt.figure(nfigure)
    #plt.subplot(211)
    
    plt.ylabel('Total glacier volume')
    plt.xlabel('Year')
    plt.title("Total volume of French alpine glaciers")
#    axes2.fill_between(range(2014, 2100), avg_yearly_volume_07, avg_yearly_volume_13, facecolor = color, alpha=0.1)
#    plt.plot(range(2014, 2100), avg_yearly_volume, color = color, linewidth=0.5)
    
    plt.legend()
    plt.show(block=False)
    #plt.gca().invert_yaxis()
    
    rcp_idx = rcp_idx+0.5


        
# We compute the average multi-model trajectories
# RCP 2.6
avg_yearly_glacier_area_rcp_26 = np.asarray(avg_yearly_glacier_area_rcp_26)
mean_glacier_area_rcp_26 = np.mean(avg_yearly_glacier_area_rcp_26, axis=0)
avg_yearly_glacier_area_rcp_26_13 = np.asarray(avg_yearly_glacier_area_rcp_26_13)
mean_glacier_area_rcp_26_13 = np.mean(avg_yearly_glacier_area_rcp_26_13, axis=0)
avg_yearly_glacier_area_rcp_26_07 = np.asarray(avg_yearly_glacier_area_rcp_26_07)
mean_glacier_area_rcp_26_07 = np.mean(avg_yearly_glacier_area_rcp_26_07, axis=0)
avg_yearly_glacier_volume_rcp_26 = np.asarray(avg_yearly_glacier_volume_rcp_26)
mean_glacier_volume_rcp_26 = np.mean(avg_yearly_glacier_volume_rcp_26, axis=0)
avg_yearly_glacier_volume_rcp_26_13 = np.asarray(avg_yearly_glacier_volume_rcp_26_13)
mean_glacier_volume_rcp_26_13 = np.mean(avg_yearly_glacier_volume_rcp_26_13, axis=0)
avg_yearly_glacier_volume_rcp_26_07 = np.asarray(avg_yearly_glacier_volume_rcp_26_07)
mean_glacier_volume_rcp_26_07 = np.mean(avg_yearly_glacier_volume_rcp_26_07, axis=0)

# RCP 4.5
avg_yearly_glacier_area_rcp_45 = np.asarray(avg_yearly_glacier_area_rcp_45)
mean_glacier_area_rcp_45 = np.mean(avg_yearly_glacier_area_rcp_45, axis=0)
avg_yearly_glacier_area_rcp_45_13 = np.asarray(avg_yearly_glacier_area_rcp_45_13)
mean_glacier_area_rcp_45_13 = np.mean(avg_yearly_glacier_area_rcp_45_13, axis=0)
avg_yearly_glacier_area_rcp_45_07 = np.asarray(avg_yearly_glacier_area_rcp_45_07)
mean_glacier_area_rcp_45_07 = np.mean(avg_yearly_glacier_area_rcp_45_07, axis=0)
avg_yearly_glacier_volume_rcp_45 = np.asarray(avg_yearly_glacier_volume_rcp_45)
mean_glacier_volume_rcp_45 = np.mean(avg_yearly_glacier_volume_rcp_45, axis=0)
avg_yearly_glacier_volume_rcp_45_13 = np.asarray(avg_yearly_glacier_volume_rcp_45_13)
mean_glacier_volume_rcp_45_13 = np.mean(avg_yearly_glacier_volume_rcp_45_13, axis=0)
avg_yearly_glacier_volume_rcp_45_07 = np.asarray(avg_yearly_glacier_volume_rcp_45_07)
mean_glacier_volume_rcp_45_07 = np.mean(avg_yearly_glacier_volume_rcp_45_07, axis=0)

# RCP 8.5
avg_yearly_glacier_area_rcp_85 = np.asarray(avg_yearly_glacier_area_rcp_85)
mean_glacier_area_rcp_85 = np.mean(avg_yearly_glacier_area_rcp_85, axis=0)
avg_yearly_glacier_area_rcp_85_13 = np.asarray(avg_yearly_glacier_area_rcp_85_13)
mean_glacier_area_rcp_85_13 = np.mean(avg_yearly_glacier_area_rcp_85_13, axis=0)
avg_yearly_glacier_area_rcp_85_07 = np.asarray(avg_yearly_glacier_area_rcp_85_07)
mean_glacier_area_rcp_85_07 = np.mean(avg_yearly_glacier_area_rcp_85_07, axis=0)
avg_yearly_glacier_volume_rcp_85 = np.asarray(avg_yearly_glacier_volume_rcp_85)
mean_glacier_volume_rcp_85 = np.mean(avg_yearly_glacier_volume_rcp_85, axis=0)
avg_yearly_glacier_volume_rcp_85_13 = np.asarray(avg_yearly_glacier_volume_rcp_85_13)
mean_glacier_volume_rcp_85_13 = np.mean(avg_yearly_glacier_volume_rcp_85_13, axis=0)
avg_yearly_glacier_volume_rcp_85_07 = np.asarray(avg_yearly_glacier_volume_rcp_85_07)
mean_glacier_volume_rcp_85_07 = np.mean(avg_yearly_glacier_volume_rcp_85_07, axis=0)

###############################  PLOTS  ###############################################

#### AREA ###
##########################################
nfigure = 1
#plt.figure(nfigure, figsize=(10, 20))
plt.figure(nfigure)
#plt.subplot(211)

avg_yearly_glacier_area_rcp_26[0] = (avg_yearly_glacier_area_rcp_26[0]/avg_yearly_glacier_area_rcp_26[0][0])*100.0 -100
avg_yearly_glacier_area_rcp_26[1] = (avg_yearly_glacier_area_rcp_26[1]/avg_yearly_glacier_area_rcp_26[1][0])*100.0 -100
mean_glacier_area_rcp_26 = (mean_glacier_area_rcp_26/mean_glacier_area_rcp_26[0])*100.0 -100

avg_yearly_glacier_area_rcp_45[0] = (avg_yearly_glacier_area_rcp_45[0]/avg_yearly_glacier_area_rcp_45[0][0])*100.0 -100
avg_yearly_glacier_area_rcp_45[1] = (avg_yearly_glacier_area_rcp_45[1]/avg_yearly_glacier_area_rcp_45[1][0])*100.0 -100
mean_glacier_area_rcp_45 = (mean_glacier_area_rcp_45/mean_glacier_area_rcp_45[0])*100.0 -100

avg_yearly_glacier_area_rcp_85[0] = (avg_yearly_glacier_area_rcp_85[0]/avg_yearly_glacier_area_rcp_85[0][0])*100.0 -100
avg_yearly_glacier_area_rcp_85[1] = (avg_yearly_glacier_area_rcp_85[1]/avg_yearly_glacier_area_rcp_85[1][0])*100.0 -100
mean_glacier_area_rcp_85 = (mean_glacier_area_rcp_85/mean_glacier_area_rcp_85[0])*100.0 -100

plt.ylabel('Relative glacier area change (%)')
plt.xlabel('Year')
plt.title("Area evolution of French alpine glaciers with forcing uncertainties", y=1.05)
axes1.fill_between(range(2014, 2100), avg_yearly_glacier_area_rcp_26[0], avg_yearly_glacier_area_rcp_26[1], facecolor = "navy", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_area_rcp_26, label = "RCP 2.6", color = "navy")
axes1.fill_between(range(2014, 2100), avg_yearly_glacier_area_rcp_45[0], avg_yearly_glacier_area_rcp_45[1], facecolor = "forestgreen", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_area_rcp_45, label = "RCP 4.5", color = "forestgreen")
axes1.fill_between(range(2014, 2100), avg_yearly_glacier_area_rcp_85[0], avg_yearly_glacier_area_rcp_85[1], facecolor = "red", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_area_rcp_85, label = "RCP 8.5", color = "red")

plt.legend()
plt.show(block=False)
#plt.gca().invert_yaxis()

############################################

mean_glacier_area_rcp_26_07 = (mean_glacier_area_rcp_26_07/mean_glacier_area_rcp_26_07[0])*100.0 -100
mean_glacier_area_rcp_26_13 = (mean_glacier_area_rcp_26_13/mean_glacier_area_rcp_26_13[0])*100.0 -100

mean_glacier_area_rcp_45_07 = (mean_glacier_area_rcp_45_07/mean_glacier_area_rcp_45_07[0])*100.0 -100
mean_glacier_area_rcp_45_13 = (mean_glacier_area_rcp_45_13/mean_glacier_area_rcp_45_13[0])*100.0 -100

mean_glacier_area_rcp_85_07 = (mean_glacier_area_rcp_85_07/mean_glacier_area_rcp_85_07[0])*100.0 -100
mean_glacier_area_rcp_85_13 = (mean_glacier_area_rcp_85_13/mean_glacier_area_rcp_85_13[0])*100.0 -100


nfigure = 2
#plt.figure(nfigure, figsize=(10, 20))
plt.figure(nfigure)
#plt.subplot(211)

plt.ylabel('Relative glacier area change (%)')
plt.xlabel('Year')
plt.title("Area evolution of French alpine glaciers with ice thickness uncertainties", y=1.05)
axes2.fill_between(range(2014, 2100), mean_glacier_area_rcp_26_07, mean_glacier_area_rcp_26_13, facecolor = "navy", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_area_rcp_26, label = "RCP 2.6", color = "navy")
axes2.fill_between(range(2014, 2100), mean_glacier_area_rcp_45_07, mean_glacier_area_rcp_45_13, facecolor = "forestgreen", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_area_rcp_45, label = "RCP 4.5", color = "forestgreen")
axes2.fill_between(range(2014, 2100), mean_glacier_area_rcp_85_07, mean_glacier_area_rcp_85_13, facecolor = "red", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_area_rcp_85, label = "RCP 8.5", color = "red")

plt.legend()
plt.show(block=False)
#plt.gca().invert_yaxis()

############################################

avg_yearly_glacier_volume_rcp_26[0] = (avg_yearly_glacier_volume_rcp_26[0]/avg_yearly_glacier_volume_rcp_26[0][0])*100.0 -100
avg_yearly_glacier_volume_rcp_26[1] = (avg_yearly_glacier_volume_rcp_26[1]/avg_yearly_glacier_volume_rcp_26[1][0])*100.0 -100
mean_glacier_volume_rcp_26 = (mean_glacier_volume_rcp_26/mean_glacier_volume_rcp_26[0])*100.0 -100

avg_yearly_glacier_volume_rcp_45[0] = (avg_yearly_glacier_volume_rcp_45[0]/avg_yearly_glacier_volume_rcp_45[0][0])*100.0 -100
avg_yearly_glacier_volume_rcp_45[1] = (avg_yearly_glacier_volume_rcp_45[1]/avg_yearly_glacier_volume_rcp_45[1][0])*100.0 -100
mean_glacier_volume_rcp_45 = (mean_glacier_volume_rcp_45/mean_glacier_volume_rcp_45[0])*100.0 -100

avg_yearly_glacier_volume_rcp_85[0] = (avg_yearly_glacier_volume_rcp_85[0]/avg_yearly_glacier_volume_rcp_85[0][0])*100.0 -100
avg_yearly_glacier_volume_rcp_85[1] = (avg_yearly_glacier_volume_rcp_85[1]/avg_yearly_glacier_volume_rcp_85[1][0])*100.0 -100
mean_glacier_volume_rcp_85 = (mean_glacier_volume_rcp_85/mean_glacier_volume_rcp_85[0])*100.0 -100

### VOLUME ###
nfigure = 3
#plt.figure(nfigure, figsize=(10, 20))
plt.figure(nfigure)
#plt.subplot(211)

plt.ylabel('Relative glacier volume change (%)')
plt.xlabel('Year')
plt.title("Volume evolution of French alpine glaciers with forcing uncertainties", y=1.05)
axes3.fill_between(range(2014, 2100), avg_yearly_glacier_volume_rcp_26[0], avg_yearly_glacier_volume_rcp_26[1], facecolor = "navy", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_volume_rcp_26, label = "RCP 2.6", color = "navy")
axes3.fill_between(range(2014, 2100), avg_yearly_glacier_volume_rcp_45[0], avg_yearly_glacier_volume_rcp_45[1], facecolor = "forestgreen", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_volume_rcp_45, label = "RCP 4.5", color = "forestgreen")
axes3.fill_between(range(2014, 2100), avg_yearly_glacier_volume_rcp_85[0], avg_yearly_glacier_volume_rcp_85[1], facecolor = "red", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_volume_rcp_85, label = "RCP 8.5", color = "red")

plt.legend()
plt.show(block=False)
#plt.gca().invert_yaxis()

###############################################

mean_glacier_volume_rcp_26_07 = (mean_glacier_volume_rcp_26_07/mean_glacier_volume_rcp_26_07[0])*100.0 -100
mean_glacier_volume_rcp_26_13 = (mean_glacier_volume_rcp_26_13/mean_glacier_volume_rcp_26_13[0])*100.0 -100

mean_glacier_volume_rcp_45_07 = (mean_glacier_volume_rcp_45_07/mean_glacier_volume_rcp_45_07[0])*100.0 -100
mean_glacier_volume_rcp_45_13 = (mean_glacier_volume_rcp_45_13/mean_glacier_volume_rcp_45_13[0])*100.0 -100

mean_glacier_volume_rcp_85_07 = (mean_glacier_volume_rcp_85_07/mean_glacier_volume_rcp_85_07[0])*100.0 -100
mean_glacier_volume_rcp_85_13 = (mean_glacier_volume_rcp_85_13/mean_glacier_volume_rcp_85_13[0])*100.0 -100

nfigure = 4
#plt.figure(nfigure, figsize=(10, 20))
plt.figure(nfigure)
#plt.subplot(211)

plt.ylabel('Relative glacier volume change (%)')
plt.xlabel('Year')
plt.title("Volume evolution of French alpine glaciers with ice thickness uncertainties", y=1.05)
axes4.fill_between(range(2014, 2100), mean_glacier_volume_rcp_26_07, mean_glacier_volume_rcp_26_13, facecolor = "navy", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_volume_rcp_26, label = "RCP 2.6", color = "navy")
axes4.fill_between(range(2014, 2100), mean_glacier_volume_rcp_45_07, mean_glacier_volume_rcp_45_13, facecolor = "forestgreen", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_volume_rcp_45, label = "RCP 4.5", color = "forestgreen")
axes4.fill_between(range(2014, 2100), mean_glacier_volume_rcp_85_07, mean_glacier_volume_rcp_85_13, facecolor = "red", alpha=0.1)
plt.plot(range(2014, 2100), mean_glacier_volume_rcp_85, label = "RCP 8.5", color = "red")

plt.legend()
plt.show(block=False)
#plt.gca().invert_yaxis()

#########################################

### ZMEAN ###
nfigure = 5
#plt.figure(nfigure, figsize=(10, 20))
plt.figure(nfigure)
#plt.subplot(211)
    
plt.ylabel('Average glacier mean altitude')
plt.xlabel('Year')
plt.title("Average yearly mean altitude of French alpine glaciers", y=1.05)
axes5.fill_between(range(2014, 2100), avg_yearly_zmean_07, avg_yearly_zmean_13, facecolor = color, alpha=0.1)
plt.plot(range(2014, 2100), avg_yearly_zmean, label = rcp, color = color)

plt.legend()
plt.show(block=False)
#plt.gca().invert_yaxis()

#########################################







