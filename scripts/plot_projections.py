# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:30:22 2020

@author: bolibarj
"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path

######   FILE PATHS    #######
    
# Folders     
workspace = str(Path(os.getcwd()).parent) + '\\'
#path_obs = 'C:\\Jordi\\PhD\\Data\\Obs\\'
path_smb = workspace + 'ALPGM\\glacier_data\\smb\\'
path_glacier_evolution = workspace + 'glacier_data\\glacier_evolution\\'
#path_glacier_evolution = 'C:\\Jordi\\PhD\\Simulations\\AGU 2018\\'
path_smb_simulations = path_smb + 'smb_simulations\\'
path_glacier_evolution_plots = path_glacier_evolution + 'plots\\'
path_glacier_area = path_glacier_evolution + 'glacier_area\\'
path_glacier_volume = path_glacier_evolution + 'glacier_volume\\'
path_glacier_zmean = path_glacier_evolution + 'glacier_zmean\\'
path_glacier_slope20 = path_glacier_evolution + 'glacier_slope20\\'
path_glacier_melt_years = path_glacier_evolution + 'glacier_melt_years\\'
path_glacier_w_errors = path_glacier_evolution + 'glacier_w_errors\\'
path_glacier_CPDDs = path_glacier_evolution + 'glacier_CPDDs\\'
path_glacier_snowfall = path_glacier_evolution + 'glacier_snowfall\\'

#### We fetch all the data from the simulations
path_area_root = np.asarray(os.listdir(path_glacier_area + "projections\\"))
path_melt_years_root = np.asarray(os.listdir(path_glacier_melt_years + "projections\\"))
path_slope20_root = np.asarray(os.listdir(path_glacier_slope20 + "projections\\"))
path_volume_root = np.asarray(os.listdir(path_glacier_volume + "projections\\"))
path_errors_root = np.asarray(os.listdir(path_glacier_w_errors + "projections\\"))
path_zmean_root = np.asarray(os.listdir(path_glacier_zmean + "projections\\"))
path_CPDDs_root = np.asarray(os.listdir(path_glacier_CPDDs + "projections\\"))
path_snowfall_root = np.asarray(os.listdir(path_glacier_snowfall + "projections\\"))

proj_blob = {'year':[], 'data':[]}
annual_mean = {'area':copy.deepcopy(proj_blob), 'volume':copy.deepcopy(proj_blob), 'zmean':copy.deepcopy(proj_blob), 'slope20':copy.deepcopy(proj_blob), 'CPDD':copy.deepcopy(proj_blob), 'snowfall':copy.deepcopy(proj_blob)}
# Data structure composed by annual values clusters
RCP_data = {'26':copy.deepcopy(annual_mean), '45':copy.deepcopy(annual_mean), '85':copy.deepcopy(annual_mean)}
first_26, first_45, first_85 = True, True, True
# Data structure composed by member clusters
RCP_members = copy.deepcopy(RCP_data)

# Data reading and processing
print("n\Reading files and creating data structures...")

# Iterate different RCP-GCM-RCM combinations
member_idx = 0
for path_forcing_area, path_forcing_melt_years, path_forcing_slope20, path_forcing_volume, path_forcing_zmean, path_forcing_CPDDs, path_forcing_snowfall in zip(path_area_root, path_melt_years_root, path_slope20_root, path_volume_root, path_zmean_root, path_CPDDs_root, path_snowfall_root):
    print("\nProcessing " + str(path_forcing_area))
    path_area_glaciers = np.asarray(os.listdir(path_glacier_area + "projections\\" + path_forcing_area))
    path_melt_years_glaciers = np.asarray(os.listdir(path_glacier_melt_years + "projections\\" + path_forcing_melt_years))
    path_slope20_glaciers = np.asarray(os.listdir(path_glacier_slope20 + "projections\\" + path_forcing_slope20))
    path_volume_glaciers = np.asarray(os.listdir(path_glacier_volume + "projections\\" + path_forcing_volume))
    path_zmean_glaciers = np.asarray(os.listdir(path_glacier_zmean + "projections\\" + path_forcing_zmean))
    path_CPDDs_glaciers = np.asarray(os.listdir(path_glacier_CPDDs + "projections\\" + path_forcing_CPDDs))
    path_snowfall_glaciers = np.asarray(os.listdir(path_glacier_snowfall + "projections\\" + path_forcing_snowfall))
    
    current_RCP = path_forcing_area[-28:-26]
#    print("current_RCP: " + str(current_RCP))
    
    # Iterate volume scaling folders
    for path_area_scaled, path_volume_scaled, path_zmean_scaled, path_slope20_scaled, path_CPDDs_scaled, path_snowfall_scaled in zip(path_area_glaciers, path_volume_glaciers, path_zmean_glaciers, path_slope20_glaciers, path_CPDDs_glaciers, path_snowfall_glaciers):
        
        path_area_glaciers_scaled = np.asarray(os.listdir(path_glacier_area + "projections\\" + path_forcing_area + "\\" + path_area_scaled))
        path_volume_glaciers_scaled = np.asarray(os.listdir(path_glacier_volume + "projections\\" + path_forcing_volume + "\\" +path_volume_scaled))
        path_zmean_glaciers_scaled = np.asarray(os.listdir(path_glacier_zmean + "projections\\" + path_forcing_zmean + "\\" + path_zmean_scaled))
        path_slope20_glaciers_scaled = np.asarray(os.listdir(path_glacier_slope20 + "projections\\" + path_forcing_slope20 + "\\" + path_slope20_scaled))
        path_CPDDs_glaciers_scaled = np.asarray(os.listdir(path_glacier_CPDDs + "projections\\" + path_forcing_CPDDs + "\\" + path_CPDDs_scaled))
        path_snowfall_glaciers_scaled = np.asarray(os.listdir(path_glacier_snowfall + "projections\\" + path_forcing_snowfall + "\\" + path_snowfall_scaled))
        
        for path_area, path_volume, path_zmean, path_slope20, path_CPDD, path_snowfall in zip(path_area_glaciers_scaled, path_volume_glaciers_scaled, path_zmean_glaciers_scaled, path_slope20_glaciers_scaled, path_CPDDs_glaciers_scaled, path_snowfall_glaciers_scaled):
            
            area_glacier = genfromtxt(path_glacier_area + "projections\\" + path_forcing_area + "\\" + path_area_scaled + "\\" + path_area, delimiter=';')
            volume_glacier = genfromtxt(path_glacier_volume + "projections\\" + path_forcing_volume + "\\" + path_volume_scaled + "\\" + path_volume, delimiter=';')
            zmean_glacier = genfromtxt(path_glacier_zmean + "projections\\" + path_forcing_zmean + "\\" + path_zmean_scaled + "\\" + path_zmean, delimiter=';')
            slope20_glacier = genfromtxt(path_glacier_slope20 + "projections\\" + path_forcing_slope20 + "\\" + path_slope20_scaled + "\\" + path_slope20, delimiter=';')
            CPDD_glacier = genfromtxt(path_glacier_CPDDs + "projections\\" + path_forcing_CPDDs + "\\" + path_CPDDs_scaled + "\\" + path_CPDD, delimiter=';')
            snowfall_glacier = genfromtxt(path_glacier_snowfall + "projections\\" + path_forcing_snowfall + "\\" + path_snowfall_scaled + "\\" + path_snowfall, delimiter=';')
            
            # Initialize data structure
            if((current_RCP == '26' and first_26) or (current_RCP == '45' and first_45) or (current_RCP == '85' and first_85)):
                for year in area_glacier:
                    RCP_data[current_RCP]['area']['data'].append([])
                    RCP_data[current_RCP]['volume']['data'].append([])
                    RCP_data[current_RCP]['zmean']['data'].append([])
                    RCP_data[current_RCP]['slope20']['data'].append([])
                    RCP_data[current_RCP]['CPDD']['data'].append([])
                    RCP_data[current_RCP]['snowfall']['data'].append([])
                    
                    RCP_data[current_RCP]['area']['year'].append(year[0])
                    RCP_data[current_RCP]['volume']['year'].append(year[0])
                    RCP_data[current_RCP]['zmean']['year'].append(year[0])
                    RCP_data[current_RCP]['slope20']['year'].append(year[0])
                    RCP_data[current_RCP]['CPDD']['year'].append(year[0])
                    RCP_data[current_RCP]['snowfall']['year'].append(year[0])
                
                for member in path_area_root:
                    RCP_members[current_RCP]['area']['data'].append([])
                    RCP_members[current_RCP]['volume']['data'].append([])
                    RCP_members[current_RCP]['zmean']['data'].append([])
                    RCP_members[current_RCP]['slope20']['data'].append([])
                    RCP_members[current_RCP]['CPDD']['data'].append([])
                    RCP_members[current_RCP]['snowfall']['data'].append([])
                    
                    RCP_members[current_RCP]['area']['year'].append(year[0])
                    RCP_members[current_RCP]['volume']['year'].append(year[0])
                    RCP_members[current_RCP]['zmean']['year'].append(year[0])
                    RCP_members[current_RCP]['slope20']['year'].append(year[0])
                    RCP_members[current_RCP]['CPDD']['year'].append(year[0])
                    RCP_members[current_RCP]['snowfall']['year'].append(year[0])
                    
                # Fix to avoid initializing data structure only until 2098
                if(len(area_glacier) == 80):
#                    import pdb; pdb.set_trace()
                    RCP_data[current_RCP]['area']['data'].append([])
                    RCP_data[current_RCP]['volume']['data'].append([])
                    RCP_data[current_RCP]['zmean']['data'].append([])
                    RCP_data[current_RCP]['slope20']['data'].append([])
                    RCP_data[current_RCP]['CPDD']['data'].append([])
                    RCP_data[current_RCP]['snowfall']['data'].append([])
                    RCP_data[current_RCP]['area']['year'].append(2099)
                    RCP_data[current_RCP]['volume']['year'].append(2099)
                    RCP_data[current_RCP]['zmean']['year'].append(2099)
                    RCP_data[current_RCP]['slope20']['year'].append(2099)
                    RCP_data[current_RCP]['CPDD']['year'].append(2099)
                    RCP_data[current_RCP]['snowfall']['year'].append(2099)
            
            # Add glacier data to blob separated by year
            year_idx = 0
            for area_y, volume_y, zmean_y, slope20_y, CPDD_y, snowfall_y in zip(area_glacier, volume_glacier, zmean_glacier, slope20_glacier, CPDD_glacier, snowfall_glacier):
                RCP_data[current_RCP]['area']['data'][year_idx].append(area_y[1])
                RCP_data[current_RCP]['volume']['data'][year_idx].append(volume_y[1])
                RCP_data[current_RCP]['zmean']['data'][year_idx].append(zmean_y[1])
                RCP_data[current_RCP]['slope20']['data'][year_idx].append(slope20_y[1])
                RCP_data[current_RCP]['CPDD']['data'][year_idx].append(CPDD_y[1])
                RCP_data[current_RCP]['snowfall']['data'][year_idx].append(snowfall_y[1])
#                print("year_idx: " + str(year_idx))
                year_idx = year_idx+1
            
                # Add data to blob separated by RCP-GCM-RCM members
                RCP_members[current_RCP]['area']['data'][member_idx].append(area_y[1])
                RCP_members[current_RCP]['volume']['data'][member_idx].append(volume_y[1])
                RCP_members[current_RCP]['zmean']['data'][member_idx].append(zmean_y[1])
                RCP_members[current_RCP]['slope20']['data'][member_idx].append(slope20_y[1])
                RCP_members[current_RCP]['CPDD']['data'][member_idx].append(CPDD_y[1])
                RCP_members[current_RCP]['snowfall']['data'][member_idx].append(snowfall_y[1])
#                print("member_idx: " + str(member_idx))
#                print("current_RCP: " + str(current_RCP))
#                import pdb; pdb.set_trace()
            
            if(current_RCP == '26'):
                first_26 = False
            elif(current_RCP == '45'):
                first_45 = False
            elif(current_RCP == '85'):
                first_85 = False
    
    member_idx=member_idx+1
print("Post-processing data...")

# Compute overall average values per year
proj_blob = {'year':[], 'data':[]}
overall_annual_mean = {'area':copy.deepcopy(proj_blob), 'volume':copy.deepcopy(proj_blob), 'zmean':copy.deepcopy(proj_blob), 'slope20':copy.deepcopy(proj_blob)}          
RCP_means = {'26':copy.deepcopy(annual_mean), '45':copy.deepcopy(annual_mean), '85':copy.deepcopy(annual_mean)}

for RCP in ['26', '45', '85']:
#    import pdb; pdb.set_trace()
    for annual_area, annual_volume, annual_zmean, annual_slope20, annual_CPDD, annual_snowfall in zip(RCP_data[RCP]['area']['data'], RCP_data[RCP]['volume']['data'], RCP_data[RCP]['zmean']['data'], RCP_data[RCP]['slope20']['data'], RCP_data[RCP]['CPDD']['data'], RCP_data[RCP]['snowfall']['data']):
    #    import pdb; pdb.set_trace()
        RCP_means[RCP]['area']['data'].append(np.average(annual_area))
        RCP_means[RCP]['area']['year'] = np.array(RCP_data[RCP]['area']['year'], dtype=int)
        RCP_means[RCP]['volume']['data'].append(np.average(annual_volume))
        RCP_means[RCP]['volume']['year'] = np.array(RCP_data[RCP]['volume']['year'], dtype=int)
        RCP_means[RCP]['zmean']['data'].append(np.average(annual_zmean))
        RCP_means[RCP]['zmean']['year'] = np.array(RCP_data[RCP]['zmean']['year'], dtype=int)
        RCP_means[RCP]['slope20']['data'].append(np.average(annual_slope20))
        RCP_means[RCP]['slope20']['year'] = np.array(RCP_data[RCP]['slope20']['year'], dtype=int)
        RCP_means[RCP]['CPDD']['data'].append(np.average(annual_CPDD))
        RCP_means[RCP]['CPDD']['year'] = np.array(RCP_data[RCP]['CPDD']['year'], dtype=int)
        RCP_means[RCP]['snowfall']['data'].append(np.average(annual_snowfall))
        RCP_means[RCP]['snowfall']['year'] = np.array(RCP_data[RCP]['snowfall']['year'], dtype=int)
        
#print(overall_annual_mean)
#import pdb; pdb.set_trace()

##########    PLOTS    #######################

#############       Plot each one of the RCP-GCM-RCM combinations       #############################################
fig1, (ax11, ax12) = plt.subplots(1,2)
fig1.suptitle("Tré-la-Tête glacier projections under climate change")
ax11.set_ylabel('Volume')
ax11.set_xlabel('Year')

for member_26 in RCP_members['26']['volume']['data']:
    if(len(member_26) > 0):
        if(len(RCP_means['26']['volume']['year']) > len(member_26)):
            ax11.plot(RCP_means['26']['volume']['year'][:-1], member_26, linewidth=0.2, c='blue')
        else:
            ax11.plot(RCP_means['26']['volume']['year'], member_26, linewidth=0.2, c='blue')
for member_45 in RCP_members['45']['volume']['data']:
    if(len(member_45) > 0):
        if(len(RCP_means['45']['volume']['year']) > len(member_45)):
            ax11.plot(RCP_means['45']['volume']['year'][:-1], member_45, linewidth=0.2, c='green')
        else:
            ax11.plot(RCP_means['45']['volume']['year'], member_45, linewidth=0.2, c='green')
for member_85 in RCP_members['85']['volume']['data']:
    if(len(member_85) > 0):
        if(len(RCP_means['85']['volume']['year']) > len(member_85)):
            ax11.plot(RCP_means['85']['volume']['year'][:-1], member_85, linewidth=0.2, c='red')
        else:
            ax11.plot(RCP_means['85']['volume']['year'], member_85, linewidth=0.2, c='red')
    
# Plot the average of each RCP
line111, = ax11.plot(RCP_means['26']['volume']['year'][:-1], RCP_means['26']['volume']['data'][:-1], linewidth=3, label='RCP 2.6', c='blue')
line112, = ax11.plot(RCP_means['45']['volume']['year'][:-1], RCP_means['45']['volume']['data'][:-1], linewidth=3, label='RCP 4.5', c='green')
line113, = ax11.plot(RCP_means['85']['volume']['year'][:-1], RCP_means['85']['volume']['data'][:-1], linewidth=3, label='RCP 8.5', c='red')
ax11.legend()

ax12.set_ylabel('Area (km$^2$)')
ax12.set_xlabel('Year')

for member_26 in RCP_members['26']['area']['data']:
    if(len(member_26) > 0):
        if(len(RCP_means['26']['volume']['year']) > len(member_26)):
            ax12.plot(RCP_means['26']['area']['year'][:-1], member_26, linewidth=0.2, c='blue')
        else:
            ax12.plot(RCP_means['26']['area']['year'], member_26, linewidth=0.2, c='blue')
for member_45 in RCP_members['45']['area']['data']:
    if(len(member_45) > 0):
        if(len(RCP_means['45']['area']['year']) > len(member_45)):
            ax12.plot(RCP_means['45']['area']['year'][:-1], member_45, linewidth=0.2, c='green')
        else:
            ax12.plot(RCP_means['45']['area']['year'], member_45, linewidth=0.2, c='green')
for member_85 in RCP_members['85']['area']['data']:
    if(len(member_85) > 0):
        if(len(RCP_means['85']['area']['year']) > len(member_85)):
            ax12.plot(RCP_means['85']['area']['year'][:-1], member_85, linewidth=0.2, c='red')
        else:
            ax12.plot(RCP_means['85']['area']['year'], member_85, linewidth=0.2, c='red')

line121, = ax12.plot(RCP_means['26']['area']['year'][:-1], RCP_means['26']['area']['data'][:-1], linewidth=3, label='RCP 2.6', c='blue')
line122, = ax12.plot(RCP_means['45']['area']['year'][:-1], RCP_means['45']['area']['data'][:-1], linewidth=3, label='RCP 4.5', c='green')
line123, = ax12.plot(RCP_means['85']['area']['year'][:-1], RCP_means['85']['area']['data'][:-1], linewidth=3, label='RCP 8.5', c='red')
ax12.legend()

###############     Plot the evolution of topographical parameters    ####################################
fig2, (ax21, ax22) = plt.subplots(1,2)
fig2.suptitle("Tré-la-Tête glacier projections under climate change")
ax21.set_ylabel('Mean glacier altitude (m)')
ax21.set_xlabel('Year')

# Mean altitude
line211, = ax21.plot(RCP_means['26']['zmean']['year'][:-1], RCP_means['26']['zmean']['data'][:-1], linewidth=3, label='RCP 2.6', c='blue')
line212, = ax21.plot(RCP_means['45']['zmean']['year'][:-1], RCP_means['45']['zmean']['data'][:-1], linewidth=3, label='RCP 4.5', c='green')
line213, = ax21.plot(RCP_means['85']['zmean']['year'][:-1], RCP_means['85']['zmean']['data'][:-1], linewidth=3, label='RCP 8.5', c='red')
ax21.legend()

# Slope 20% altitudinal range
ax22.set_ylabel('Slope of 20% altitudinal range (°)')
ax22.set_xlabel('Year')
line221, = ax22.plot(RCP_means['26']['slope20']['year'][:-1], RCP_means['26']['slope20']['data'][:-1], linewidth=3, label='RCP 2.6', c='blue')
line222, = ax22.plot(RCP_means['45']['slope20']['year'][:-1], RCP_means['45']['slope20']['data'][:-1], linewidth=3, label='RCP 4.5', c='green')
line223, = ax22.plot(RCP_means['85']['slope20']['year'][:-1], RCP_means['85']['slope20']['data'][:-1], linewidth=3, label='RCP 8.5', c='red')
ax22.legend()

###############     Plot the evolution of temperature and snowfall    ####################################
fig3, (ax31, ax32) = plt.subplots(1,2)
fig3.suptitle("Tré-la-Tête climate projections")
ax31.set_ylabel('Cumulative positive degree days anomaly (1984-2015)')
ax31.set_xlabel('Year')

# CPDD
for member_26 in RCP_members['26']['CPDD']['data']:
    if(len(member_26) > 0):
        if(len(RCP_means['26']['CPDD']['year']) > len(member_26)):
            ax31.plot(RCP_means['26']['CPDD']['year'][:-1], member_26, linewidth=0.2, c='blue')
        else:
            ax31.plot(RCP_means['26']['CPDD']['year'], member_26, linewidth=0.2, c='blue')
for member_45 in RCP_members['45']['CPDD']['data']:
    if(len(member_45) > 0):
        if(len(RCP_means['45']['CPDD']['year']) > len(member_45)):
            ax31.plot(RCP_means['45']['CPDD']['year'][:-1], member_45, linewidth=0.2, c='green')
        else:
            ax31.plot(RCP_means['45']['CPDD']['year'], member_45, linewidth=0.2, c='green')
for member_85 in RCP_members['85']['CPDD']['data']:
    if(len(member_85) > 0):
        if(len(RCP_means['85']['CPDD']['year']) > len(member_85)):
            ax31.plot(RCP_means['85']['CPDD']['year'][:-1], member_85, linewidth=0.2, c='red')
        else:
            ax31.plot(RCP_means['85']['CPDD']['year'], member_85, linewidth=0.2, c='red')

line311, = ax31.plot(RCP_means['26']['CPDD']['year'][:-1], RCP_means['26']['CPDD']['data'][:-1], linewidth=3, label='RCP 2.6', c='blue')
line312, = ax31.plot(RCP_means['45']['CPDD']['year'][:-1], RCP_means['45']['CPDD']['data'][:-1], linewidth=3, label='RCP 4.5', c='green')
line313, = ax31.plot(RCP_means['85']['CPDD']['year'][:-1], RCP_means['85']['CPDD']['data'][:-1], linewidth=3, label='RCP 8.5', c='red')
ax31.legend()

# Snowfall
for member_26 in RCP_members['26']['snowfall']['data']:
    if(len(member_26) > 0):
        if(len(RCP_means['26']['snowfall']['year']) > len(member_26)):
            ax32.plot(RCP_means['26']['snowfall']['year'][:-1], member_26, linewidth=0.2, c='blue')
        else:
            ax32.plot(RCP_means['26']['snowfall']['year'], member_26, linewidth=0.2, c='blue')
for member_45 in RCP_members['45']['snowfall']['data']:
    if(len(member_45) > 0):
        if(len(RCP_means['45']['snowfall']['year']) > len(member_45)):
            ax32.plot(RCP_means['45']['snowfall']['year'][:-1], member_45, linewidth=0.2, c='green')
        else:
            ax32.plot(RCP_means['45']['snowfall']['year'], member_45, linewidth=0.2, c='green')
for member_85 in RCP_members['85']['snowfall']['data']:
    if(len(member_85) > 0):
        if(len(RCP_means['85']['snowfall']['year']) > len(member_85)):
            ax32.plot(RCP_means['85']['snowfall']['year'][:-1], member_85, linewidth=0.2, c='red')
        else:
            ax32.plot(RCP_means['85']['snowfall']['year'], member_85, linewidth=0.2, c='red')

ax32.set_ylabel('Annual cumulative snowfall anomaly (1984-2015)')
ax32.set_xlabel('Year')
line321, = ax32.plot(RCP_means['26']['snowfall']['year'][:-1], RCP_means['26']['snowfall']['data'][:-1], linewidth=3, label='RCP 2.6', c='blue')
line322, = ax32.plot(RCP_means['45']['snowfall']['year'][:-1], RCP_means['45']['snowfall']['data'][:-1], linewidth=3, label='RCP 4.5', c='green')
line323, = ax32.plot(RCP_means['85']['snowfall']['year'][:-1], RCP_means['85']['snowfall']['data'][:-1], linewidth=3, label='RCP 8.5', c='red')
ax32.legend()

plt.show()