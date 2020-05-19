# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:30:22 2020

@author: Jordi Bolibar
"""

## Dependencies: ##
import matplotlib.pyplot as plt
import proplot as plot
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path

###### FLAGS  #########
with_26 = False
filter_glacier = False
#filter_member = False
# mer de glace
#glacier_ID_filter = "G006934E45883N"
# argentiere
glacier_ID_filter = "G006985E45951N"
# Tré-la-Tête
#glacier_ID_filter = "G006784E45784N"

# Member index to be filtered
# Set to -1 to turn filtering off
filtered_member = -1

# Year to start the projected plots
year_start = 2015

######   FILE PATHS    #######

# Folders     
workspace = str(Path(os.getcwd()).parent) 
path_glims = os.path.join(workspace, 'glacier_data', 'GLIMS') 
#path_obs = 'C:\\Jordi\\PhD\\Data\\Obs\\'
path_smb = os.path.join(workspace, 'glacier_data', 'smb')
path_glacier_evolution = os.path.join(workspace, 'glacier_data', 'glacier_evolution')
#path_glacier_evolution = 'C:\\Jordi\\PhD\\Simulations\\AGU 2018\\'
path_smb_simulations = os.path.join(path_smb, 'smb_simulations')
path_glacier_evolution_plots = os.path.join(path_glacier_evolution, 'plots')
path_glacier_area = os.path.join(path_glacier_evolution, 'glacier_area')
path_glacier_volume = os.path.join(path_glacier_evolution, 'glacier_volume')
path_glacier_zmean = os.path.join(path_glacier_evolution, 'glacier_zmean')
path_glacier_slope20 = os.path.join(path_glacier_evolution, 'glacier_slope20')
path_glacier_melt_years = os.path.join(path_glacier_evolution, 'glacier_melt_years')
path_glacier_w_errors = os.path.join(path_glacier_evolution, 'glacier_w_errors')
path_glacier_CPDDs = os.path.join(path_glacier_evolution, 'glacier_CPDDs')
path_glacier_s_CPDDs = os.path.join(path_glacier_evolution, 'glacier_summer_CPDDs')
path_glacier_w_CPDDs = os.path.join(path_glacier_evolution, 'glacier_winter_CPDDs')
path_glacier_snowfall = os.path.join(path_glacier_evolution, 'glacier_snowfall')
path_glacier_s_snowfall = os.path.join(path_glacier_evolution, 'glacier_summer_snowfall')
path_glacier_w_snowfall = os.path.join(path_glacier_evolution, 'glacier_winter_snowfall')
path_glacier_s_rain = os.path.join(path_glacier_evolution, 'glacier_summer_rain')
path_glacier_w_rain = os.path.join(path_glacier_evolution, 'glacier_winter_rain')
path_glacier_discharge = os.path.join(path_glacier_evolution, 'glacier_meltwater_discharge')

#### We fetch all the data from the simulations
path_area_root = np.asarray(os.listdir(os.path.join(path_glacier_area, "projections")))
path_melt_years_root = np.asarray(os.listdir(os.path.join(path_glacier_melt_years, "projections")))
path_slope20_root = np.asarray(os.listdir(os.path.join(path_glacier_slope20, "projections")))
path_volume_root = np.asarray(os.listdir(os.path.join(path_glacier_volume, "projections")))
path_errors_root = np.asarray(os.listdir(os.path.join(path_glacier_w_errors, "projections")))
path_zmean_root = np.asarray(os.listdir(os.path.join(path_glacier_zmean, "projections")))
path_s_CPDDs_root = np.asarray(os.listdir(os.path.join(path_glacier_s_CPDDs, "projections")))
path_w_CPDDs_root = np.asarray(os.listdir(os.path.join(path_glacier_w_CPDDs, "projections")))
path_s_snowfall_root = np.asarray(os.listdir(os.path.join(path_glacier_s_snowfall, "projections")))
path_w_snowfall_root = np.asarray(os.listdir(os.path.join(path_glacier_w_snowfall, "projections")))
path_s_rain_root = np.asarray(os.listdir(os.path.join(path_glacier_s_rain, "projections")))
path_w_rain_root = np.asarray(os.listdir(os.path.join(path_glacier_w_rain, "projections")))
path_SMB_root = np.asarray(os.listdir(os.path.join(path_smb_simulations, "projections")))

glims_2003 = genfromtxt(os.path.join(path_glims, 'GLIMS_2003.csv'), delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), 
                        ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), 
                        ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8')])
glacier_name_filter = glims_2003['Glacier'][glims_2003['GLIMS_ID'] == glacier_ID_filter.encode('UTF-8')]
glacier_name_filter = glacier_name_filter[0].decode('UTF-8')
#print("\nFiltered glacier name: " + str(glacier_name_filter))

proj_blob = {'year':[], 'data':[]}
annual_mean = {'SMB':copy.deepcopy(proj_blob), 'area':copy.deepcopy(proj_blob), 'volume':copy.deepcopy(proj_blob), 
               'zmean':copy.deepcopy(proj_blob), 'slope20':copy.deepcopy(proj_blob), 
               'CPDD':copy.deepcopy(proj_blob), 'summer_CPDD':copy.deepcopy(proj_blob), 'winter_CPDD':copy.deepcopy(proj_blob), 
               'snowfall':copy.deepcopy(proj_blob), 'summer_snowfall':copy.deepcopy(proj_blob), 'winter_snowfall':copy.deepcopy(proj_blob),
               'rain':copy.deepcopy(proj_blob), 'summer_rain':copy.deepcopy(proj_blob), 'winter_rain':copy.deepcopy(proj_blob),
               'discharge':copy.deepcopy(proj_blob)}

# Data structure composed by annual values clusters
RCP_data = {'26':copy.deepcopy(annual_mean), '45':copy.deepcopy(annual_mean), '85':copy.deepcopy(annual_mean)}
multiple_RCP_data = {'26':[], '45':[], '85':[]}
first_26, first_45, first_85 = True, True, True
# Data structure composed by member clusters
RCP_members = copy.deepcopy(multiple_RCP_data)
RCP_member_means = copy.deepcopy(multiple_RCP_data)

# Array of indexes of the data structure to iterate
data_idxs = ['SMB','area','volume','zmean','slope20', 'CPDD', 'summer_CPDD', 'winter_CPDD', 'snowfall', 'summer_snowfall', 'winter_snowfall', 'rain', 'summer_rain', 'winter_rain', 'discharge']

members_with_26 = np.array(['KNMI-RACMO22E_MOHC-HadGEM2-ES', 'MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR', 'SMHI-RCA4_ICHEC-EC-EARTH'])

#######################    FUNCTIONS    ##########################################################

def save_plot_as_pdf(fig, variables, with_26):
    # Save as PDF
    if(with_26):
        fig.savefig(os.path.join(path_glacier_evolution_plots, 'summary', 'pdf', 'glacier_' + str(variables) + '_evolution_with_26.pdf'))
        fig.savefig(os.path.join(path_glacier_evolution_plots, 'summary', 'jpeg', 'glacier_' + str(variables) + '_evolution_with_26.jpeg'))
    else:
        fig.savefig(os.path.join(path_glacier_evolution_plots, 'summary', 'pdf', 'glacier_' + str(variables) + '_evolution.pdf'))
        fig.savefig(os.path.join(path_glacier_evolution_plots, 'summary', 'jpeg', 'glacier_' + str(variables) + '_evolution.jpeg'))
        
        
# Store the RCP means in CSV files
def store_RCP_mean(path_variable, variable, RCP_means):
    
    path_RCP_means = os.path.join(path_variable, "RCP_means")
    if not os.path.exists(path_RCP_means):
            os.makedirs(path_RCP_means)
    RCPs = ['26', '45', '85']
    for RCP in RCPs:
        if((with_26 and RCP == '26') or RCP != '26'):
            data = np.asarray(RCP_means[RCP][variable]['data'][:-1])
            years = np.asarray(RCP_means[RCP][variable]['year'][:-1])
            data_years = np.transpose(np.stack((data,years)))
            
            if(with_26):
                np.savetxt(os.path.join(path_RCP_means, "RCP" + str(RCP) + "_glacier_with_26_" + str(variable) + "_" + str(years[0])+ "_" + str(years[-1]) + '.csv'), 
                           data_years, delimiter=";", fmt="%s")
            else:
                np.savetxt(os.path.join(path_RCP_means, "RCP" + str(RCP) + "_glacier_" + str(variable) + "_" + str(years[0])+ "_" + str(years[-1]) + '.csv'), 
                           data_years, delimiter=";", fmt="%s")
                
def plot_individual_members(ax, RCP_member_means, RCP_means, variable, filtered_member, alpha=0.2):
    member_idx = 0
    for member_26 in RCP_member_means['26']:
        data_26 = member_26[variable]['data']
        if(len(data_26) > 0 and (member_idx == filtered_member or filtered_member == -1)):
            if(len(RCP_means['26'][variable]['year']) > len(data_26)):
                ax.plot(RCP_means['26'][variable]['year'][:-1], data_26, linewidth=0.1, alpha=alpha, c='steelblue')
            else:
                ax.plot(RCP_means['26'][variable]['year'], data_26, linewidth=0.1, alpha=alpha, c='steelblue')
        member_idx=member_idx+1
    member_idx = 0
    for member_45 in RCP_member_means['45']:
        data_45 = member_45[variable]['data']
        if(len(data_45) > 0 and (member_idx == filtered_member or filtered_member == -1)):
            if(len(RCP_means['45'][variable]['year']) > len(data_45)):
                ax.plot(RCP_means['45'][variable]['year'][:-1], data_45, linewidth=0.1, alpha=alpha+0.1, c='brown orange')
            else:
                ax.plot(RCP_means['45'][variable]['year'], data_45, linewidth=0.1, alpha=alpha+0.1, c='brown orange')
        member_idx=member_idx+1
    member_idx = 0
    for member_85 in RCP_member_means['85']:
        data_85 = member_85[variable]['data']
        if(len(data_85) > 0 and (member_idx == filtered_member or filtered_member == -1)):
            if(len(RCP_means['85'][variable]['year']) > len(data_85)):
                ax.plot(RCP_means['85'][variable]['year'][:-1], data_85, linewidth=0.1, alpha=alpha, c='darkred')
            else:
                ax.plot(RCP_means['85'][variable]['year'], data_85, linewidth=0.1, alpha=alpha, c='darkred')
        member_idx=member_idx+1
        
def plot_RCP_means(ax, RCP_means, variable, with_26, legend=True, linewidth=2):
    if(legend):
        legend_pos='ur'
    else:
        legend_pos=''
    if(with_26):
        ax.plot(RCP_means['26'][variable]['year'][:-1], np.asarray(RCP_means['26'][variable]['data'][:-1]), linewidth=linewidth, label='RCP 2.6', c='steelblue', legend=legend_pos)
    ax.plot(RCP_means['45'][variable]['year'][:-1], np.asarray(RCP_means['45'][variable]['data'][:-1]), linewidth=linewidth, label='RCP 4.5', c='brown orange', legend=legend_pos)
    ax.plot(RCP_means['85'][variable]['year'][:-1], np.asarray(RCP_means['85'][variable]['data'][:-1]), linewidth=linewidth, label='RCP 8.5', c='darkred', legend=legend_pos)
    
##################################################################################################
        
        
###############################################################################
###                           MAIN                                          ###
###############################################################################

# Data reading and processing
print("\nReading files and creating data structures...")

# Iterate different RCP-GCM-RCM combinations
member_26_idx, member_45_idx, member_85_idx = 0, 0, 0
root_paths = zip(path_SMB_root, path_area_root, path_melt_years_root, path_slope20_root, path_volume_root, path_zmean_root, 
                 path_s_CPDDs_root, path_w_CPDDs_root,
                 path_s_snowfall_root, path_w_snowfall_root,
                 path_s_rain_root, path_w_rain_root)
for path_forcing_SMB, path_forcing_area, path_forcing_melt_years, path_forcing_slope20, path_forcing_volume, path_forcing_zmean, path_forcing_s_CPDDs, path_forcing_w_CPDDs, path_forcing_s_snowfall, path_forcing_w_snowfall, path_forcing_s_rain, path_forcing_w_rain in root_paths:
    
    current_RCP = path_forcing_area[-28:-26]
    current_member = path_forcing_area[8:-32]
    
    # Filter members depending if we want to include RCP 2.6 or not
    if((with_26 and np.any(current_member == members_with_26)) or (not with_26 and current_RCP != '26')):
#    if(current_member == 'CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5'):
        print("\nProcessing " + str(path_forcing_area))
        
        # Assign the right member idx
        if(current_RCP == '26'):
            member_idx = copy.deepcopy(member_26_idx)
        elif(current_RCP == '45'):
            member_idx = copy.deepcopy(member_45_idx)
        if(current_RCP == '85'):
            member_idx = copy.deepcopy(member_85_idx)
            
#        print("member_idx: " + str(member_idx))
            
        path_area_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_area, "projections", path_forcing_area)))
        path_melt_years_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_melt_years, "projections", path_forcing_melt_years)))
        path_slope20_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_slope20, "projections", path_forcing_slope20)))
        path_volume_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_volume, "projections", path_forcing_volume)))
        path_zmean_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_zmean, "projections", path_forcing_zmean)))
        path_s_CPDDs_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_s_CPDDs, "projections", path_forcing_s_CPDDs)))
        path_w_CPDDs_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_w_CPDDs, "projections", path_forcing_w_CPDDs)))
        path_s_snowfall_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_s_snowfall, "projections", path_forcing_s_snowfall)))
        path_w_snowfall_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_w_snowfall, "projections", path_forcing_w_snowfall)))
        path_s_rain_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_s_rain, "projections", path_forcing_s_rain)))
        path_w_rain_glaciers = np.asarray(os.listdir(os.path.join(path_glacier_w_rain, "projections", path_forcing_w_rain)))
        path_SMB_glaciers = np.asarray(os.listdir(os.path.join(path_smb_simulations, "projections", path_forcing_SMB)))
        
        # Initialize data structures
        # We add a new member to the RCP group
        RCP_members[current_RCP].append(copy.deepcopy(annual_mean))
        RCP_member_means[current_RCP].append(copy.deepcopy(annual_mean))
                            
        for year in range(year_start, 2100):
            for data_idx in data_idxs:
                RCP_members[current_RCP][member_idx][data_idx]['data'].append([])
                RCP_members[current_RCP][member_idx][data_idx]['year'].append(year)
                RCP_member_means[current_RCP][member_idx][data_idx]['year'].append(year)
        
        bump_member = False
        # Iterate volume scaling folders
        volume_scale_paths = zip(path_SMB_glaciers, path_area_glaciers, path_volume_glaciers, path_zmean_glaciers, path_slope20_glaciers, 
                                 path_s_CPDDs_glaciers, path_w_CPDDs_glaciers,
                                 path_s_snowfall_glaciers, path_w_snowfall_glaciers,
                                 path_s_rain_glaciers, path_w_rain_glaciers)
        for path_SMB_scaled, path_area_scaled, path_volume_scaled, path_zmean_scaled, path_slope20_scaled, path_s_CPDDs_scaled, path_w_CPDDs_scaled, path_s_snowfall_scaled, path_w_snowfall_scaled, path_s_rain_scaled, path_w_rain_scaled in volume_scale_paths:
            
            path_area_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_area, "projections", path_forcing_area, path_area_scaled)))
            path_volume_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_volume, "projections", path_forcing_volume, path_volume_scaled)))
            path_zmean_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_zmean, "projections", path_forcing_zmean, path_zmean_scaled)))
            path_slope20_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_slope20, "projections", path_forcing_slope20, path_slope20_scaled)))
            path_s_CPDDs_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_s_CPDDs, "projections", path_forcing_s_CPDDs, path_s_CPDDs_scaled)))
            path_w_CPDDs_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_w_CPDDs, "projections", path_forcing_w_CPDDs, path_w_CPDDs_scaled)))
            path_s_snowfall_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_s_snowfall, "projections", path_forcing_s_snowfall, path_s_snowfall_scaled)))
            path_w_snowfall_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_w_snowfall, "projections", path_forcing_w_snowfall, path_w_snowfall_scaled)))
            path_s_rain_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_s_rain, "projections", path_forcing_s_rain, path_s_rain_scaled)))
            path_w_rain_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_glacier_w_rain, "projections", path_forcing_w_rain, path_w_rain_scaled)))
            path_SMB_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_smb_simulations, "projections", path_forcing_SMB, path_SMB_scaled)))
            
            glacier_count = 0
#            if(path_area_scaled == '1'):
            if(path_area_scaled == '1' and path_s_CPDDs_glaciers_scaled.size > 369):
                bump_member = True
                glacier_paths = zip(path_SMB_glaciers_scaled, path_area_glaciers_scaled, path_volume_glaciers_scaled, path_zmean_glaciers_scaled, path_slope20_glaciers_scaled, 
                                    path_s_CPDDs_glaciers_scaled, path_w_CPDDs_glaciers_scaled,
                                    path_s_snowfall_glaciers_scaled, path_w_snowfall_glaciers_scaled,
                                    path_s_rain_glaciers_scaled, path_w_rain_glaciers_scaled)
                for path_SMB, path_area, path_volume, path_zmean, path_slope20, path_s_CPDD, path_w_CPDD, path_s_snowfall, path_w_snowfall, path_s_rain, path_w_rain in glacier_paths:
                    
                    
    #                print("path_SMB[:13]: " + str(path_SMB[:13]))
    #                print("glacier_ID_filter: " + str(glacier_ID_filter))
    #                if(path_SMB[:14] == glacier_ID_filter):
                    if((filter_glacier and glacier_ID_filter == path_SMB[:14]) or not filter_glacier):
                        
                        area_glacier = genfromtxt(os.path.join(path_glacier_area, "projections", path_forcing_area, path_area_scaled, path_area), delimiter=';')
                        volume_glacier = genfromtxt(os.path.join(path_glacier_volume, "projections", path_forcing_volume, path_volume_scaled, path_volume), delimiter=';')
                        zmean_glacier = genfromtxt(os.path.join(path_glacier_zmean, "projections", path_forcing_zmean, path_zmean_scaled, path_zmean), delimiter=';')
                        slope20_glacier = genfromtxt(os.path.join(path_glacier_slope20, "projections", path_forcing_slope20, path_slope20_scaled, path_slope20), delimiter=';')
                        s_CPDD_glacier = genfromtxt(os.path.join(path_glacier_s_CPDDs, "projections", path_forcing_s_CPDDs, path_s_CPDDs_scaled, path_s_CPDD), delimiter=';')
                        w_CPDD_glacier = genfromtxt(os.path.join(path_glacier_w_CPDDs, "projections", path_forcing_w_CPDDs, path_w_CPDDs_scaled, path_w_CPDD), delimiter=';')
                        s_snowfall_glacier = genfromtxt(os.path.join(path_glacier_s_snowfall, "projections", path_forcing_s_snowfall, path_s_snowfall_scaled, path_s_snowfall), delimiter=';')
                        w_snowfall_glacier = genfromtxt(os.path.join(path_glacier_w_snowfall, "projections", path_forcing_w_snowfall, path_w_snowfall_scaled, path_w_snowfall), delimiter=';')
                        s_rain_glacier = genfromtxt(os.path.join(path_glacier_s_rain, "projections", path_forcing_s_rain, path_s_rain_scaled, path_s_rain), delimiter=';')
                        w_rain_glacier = genfromtxt(os.path.join(path_glacier_w_rain, "projections", path_forcing_w_rain, path_w_rain_scaled, path_w_rain), delimiter=';')
                        SMB_glacier = genfromtxt(os.path.join(path_smb_simulations, "projections", path_forcing_SMB, path_SMB_scaled, path_SMB), delimiter=';')
                        
                        if(len(SMB_glacier.shape) > 1):
                            for year in range(year_start, 2100):
                                for data_idx in data_idxs:
                                    if((current_RCP == '26' and first_26) or (current_RCP == '45' and first_45) or (current_RCP == '85' and first_85)):
                                        RCP_data[current_RCP][data_idx]['data'].append([])
                                        RCP_data[current_RCP][data_idx]['year'].append(year)
                            
                            # Add glacier data to blob separated by year
                            year_idx = 0
                            
#                            print("\nmember_idx: " + str(member_idx))
                            years_path = zip(SMB_glacier, area_glacier, volume_glacier, zmean_glacier, slope20_glacier, 
                                             s_CPDD_glacier, w_CPDD_glacier,
                                             s_snowfall_glacier, w_snowfall_glacier,
                                             s_rain_glacier, w_rain_glacier)
                            for SMB_y, area_y, volume_y, zmean_y, slope20_y, s_CPDD_y, w_CPDD_y, s_snowfall_y, w_snowfall_y, s_rain_y, w_rain_y in years_path:
                                RCP_data[current_RCP]['SMB']['data'][year_idx].append(SMB_y[1])
                                RCP_data[current_RCP]['area']['data'][year_idx].append(area_y[1])
                                RCP_data[current_RCP]['volume']['data'][year_idx].append(volume_y[1])
                                RCP_data[current_RCP]['zmean']['data'][year_idx].append(zmean_y[1])
                                RCP_data[current_RCP]['slope20']['data'][year_idx].append(slope20_y[1])
                                
                                RCP_data[current_RCP]['CPDD']['data'][year_idx].append(s_CPDD_y[1] + w_CPDD_y[1])
                                RCP_data[current_RCP]['summer_CPDD']['data'][year_idx].append(s_CPDD_y[1])
                                RCP_data[current_RCP]['winter_CPDD']['data'][year_idx].append(w_CPDD_y[1])
                                RCP_data[current_RCP]['snowfall']['data'][year_idx].append(s_snowfall_y[1] + w_snowfall_y[1])
                                RCP_data[current_RCP]['summer_snowfall']['data'][year_idx].append(s_snowfall_y[1])
                                RCP_data[current_RCP]['winter_snowfall']['data'][year_idx].append(w_snowfall_y[1])
                                RCP_data[current_RCP]['rain']['data'][year_idx].append(s_rain_y[1] + w_rain_y[1])
                                RCP_data[current_RCP]['summer_rain']['data'][year_idx].append(s_rain_y[1])
                                RCP_data[current_RCP]['winter_rain']['data'][year_idx].append(w_rain_y[1])
                                annual_discharge = -1*area_y[1]*SMB_y[1]
                                if(annual_discharge < 0):
                                    annual_discharge = 0
                                RCP_data[current_RCP]['discharge']['data'][year_idx].append(annual_discharge)
                            
                                # Add data to blob separated by RCP-GCM-RCM members
                                RCP_members[current_RCP][member_idx]['SMB']['data'][year_idx].append(SMB_y[1])
                                RCP_members[current_RCP][member_idx]['area']['data'][year_idx].append(area_y[1])
                                RCP_members[current_RCP][member_idx]['volume']['data'][year_idx].append(volume_y[1])
                                RCP_members[current_RCP][member_idx]['zmean']['data'][year_idx].append(zmean_y[1])
                                RCP_members[current_RCP][member_idx]['slope20']['data'][year_idx].append(slope20_y[1])
                                
                                RCP_members[current_RCP][member_idx]['CPDD']['data'][year_idx].append(s_CPDD_y[1] + w_CPDD_y[1])
                                RCP_members[current_RCP][member_idx]['summer_CPDD']['data'][year_idx].append(s_CPDD_y[1])
                                RCP_members[current_RCP][member_idx]['winter_CPDD']['data'][year_idx].append(w_CPDD_y[1])
                                RCP_members[current_RCP][member_idx]['snowfall']['data'][year_idx].append(s_snowfall_y[1] + w_snowfall_y[1])
                                RCP_members[current_RCP][member_idx]['summer_snowfall']['data'][year_idx].append(s_snowfall_y[1])
                                RCP_members[current_RCP][member_idx]['winter_snowfall']['data'][year_idx].append(w_snowfall_y[1])
                                RCP_members[current_RCP][member_idx]['rain']['data'][year_idx].append(s_rain_y[1] + w_rain_y[1])
                                RCP_members[current_RCP][member_idx]['summer_rain']['data'][year_idx].append(s_rain_y[1])
                                RCP_members[current_RCP][member_idx]['winter_rain']['data'][year_idx].append(w_rain_y[1])
                                RCP_members[current_RCP][member_idx]['discharge']['data'][year_idx].append(annual_discharge)
                                
                                year_idx = year_idx+1
                            
                            if(current_RCP == '26'):
                                first_26 = False
                            elif(current_RCP == '45'):
                                first_45 = False
                            elif(current_RCP == '85'):
                                first_85 = False
                                
                            glacier_count = glacier_count+1
                            
        ### End of if RCP-GCM-RCM is part of the subgroup
        # Bump the right member idx 
        if(bump_member):
            if(current_RCP == '26'):
                member_26_idx = member_26_idx+1
            elif(current_RCP == '45'):
                member_45_idx = member_45_idx+1
            if(current_RCP == '85'):
                member_85_idx = member_85_idx+1
        
print("\nPost-processing data...")

# Compute overall average values per year
RCP_means = {'26':copy.deepcopy(annual_mean), '45':copy.deepcopy(annual_mean), '85':copy.deepcopy(annual_mean)}

year_size = copy.deepcopy(year_idx)

if(with_26):
    RCP_array = ['26', '45', '85']
else:
    RCP_array = ['45', '85']
    
for RCP in RCP_array:
    print("\nCurrent RCP: " + str(RCP))
    if(RCP == '26'):
        member_idx = copy.deepcopy(member_26_idx)
    elif(RCP == '45'):
        member_idx = copy.deepcopy(member_45_idx)
    if(RCP == '85'):
        member_idx = copy.deepcopy(member_85_idx)
    
    year_range = np.asarray(range(year_start, 2100))
    for year_idx in range(0, year_range.size):
        
        # RCP_means
        RCP_means[RCP]['SMB']['data'].append(np.nanmean(RCP_data[RCP]['SMB']['data'][year_idx]))
        RCP_means[RCP]['SMB']['year'] = np.array(RCP_data[RCP]['SMB']['year'], dtype=int)
        RCP_means[RCP]['zmean']['data'].append(np.nanmean(RCP_data[RCP]['zmean']['data'][year_idx]))
        RCP_means[RCP]['zmean']['year'] = np.array(RCP_data[RCP]['zmean']['year'], dtype=int)
        RCP_means[RCP]['slope20']['data'].append(np.nanmean(RCP_data[RCP]['slope20']['data'][year_idx]))
        RCP_means[RCP]['slope20']['year'] = np.array(RCP_data[RCP]['slope20']['year'], dtype=int)
        
        RCP_means[RCP]['CPDD']['data'].append(np.nanmean(RCP_data[RCP]['CPDD']['data'][year_idx]))
        RCP_means[RCP]['CPDD']['year'] = np.array(RCP_data[RCP]['CPDD']['year'], dtype=int)
        RCP_means[RCP]['summer_CPDD']['data'].append(np.nanmean(RCP_data[RCP]['summer_CPDD']['data'][year_idx]))
        RCP_means[RCP]['summer_CPDD']['year'] = np.array(RCP_data[RCP]['summer_CPDD']['year'], dtype=int)
        RCP_means[RCP]['winter_CPDD']['data'].append(np.nanmean(RCP_data[RCP]['winter_CPDD']['data'][year_idx]))
        RCP_means[RCP]['winter_CPDD']['year'] = np.array(RCP_data[RCP]['winter_CPDD']['year'], dtype=int)
        RCP_means[RCP]['snowfall']['data'].append(np.nanmean(RCP_data[RCP]['snowfall']['data'][year_idx]))
        RCP_means[RCP]['snowfall']['year'] = np.array(RCP_data[RCP]['snowfall']['year'], dtype=int)
        RCP_means[RCP]['summer_snowfall']['data'].append(np.nanmean(RCP_data[RCP]['summer_snowfall']['data'][year_idx]))
        RCP_means[RCP]['summer_snowfall']['year'] = np.array(RCP_data[RCP]['summer_snowfall']['year'], dtype=int)
        RCP_means[RCP]['winter_snowfall']['data'].append(np.nanmean(RCP_data[RCP]['winter_snowfall']['data'][year_idx]))
        RCP_means[RCP]['winter_snowfall']['year'] = np.array(RCP_data[RCP]['winter_snowfall']['year'], dtype=int)
        RCP_means[RCP]['rain']['data'].append(np.nanmean(RCP_data[RCP]['rain']['data'][year_idx]))
        RCP_means[RCP]['rain']['year'] = np.array(RCP_data[RCP]['rain']['year'], dtype=int)
        RCP_means[RCP]['summer_rain']['data'].append(np.nanmean(RCP_data[RCP]['summer_rain']['data'][year_idx]))
        RCP_means[RCP]['summer_rain']['year'] = np.array(RCP_data[RCP]['summer_rain']['year'], dtype=int)
        RCP_means[RCP]['winter_rain']['data'].append(np.nanmean(RCP_data[RCP]['winter_rain']['data'][year_idx]))
        RCP_means[RCP]['winter_rain']['year'] = np.array(RCP_data[RCP]['winter_rain']['year'], dtype=int)
        
        annual_discharge = -1*np.nansum(np.asarray(RCP_data[RCP]['SMB']['data'][year_idx])*np.asarray(RCP_data[RCP]['area']['data'][year_idx]))/member_idx
        annual_discharge = np.where(annual_discharge < 0, 0, annual_discharge)
        RCP_means[RCP]['discharge']['data'].append(annual_discharge)
        RCP_means[RCP]['discharge']['year'] = np.array(RCP_data[RCP]['snowfall']['year'], dtype=int)
#        
        # RCP_member_means
        if(filter_glacier):
            member_idx = len(RCP_members[RCP])
#        print("len(RCP_members[RCP]): " + str(len(RCP_members[RCP])))
#        print("member_idx: " + str(member_idx))
#        for member in range(0, member_idx-1):
        for member in range(0, member_idx):
#            if(year_idx == 83):
#                print("member: " + str(member))
#            print("year_idx: " + str(year_idx))
#            if(member == 12):
#                import pdb; pdb.set_trace()
            
            RCP_member_means[RCP][member]['SMB']['year'] = np.array(RCP_members[RCP][member]['SMB']['year'], dtype=int)
            RCP_member_means[RCP][member]['area']['year'] = np.array(RCP_members[RCP][member]['area']['year'], dtype=int)
            RCP_member_means[RCP][member]['volume']['year'] = np.array(RCP_members[RCP][member]['volume']['year'], dtype=int)
            RCP_member_means[RCP][member]['zmean']['year'] = np.array(RCP_members[RCP][member]['zmean']['year'], dtype=int)
            RCP_member_means[RCP][member]['slope20']['year'] = np.array(RCP_members[RCP][member]['slope20']['year'], dtype=int)
            
            RCP_member_means[RCP][member]['CPDD']['year'] = np.array(RCP_members[RCP][member]['CPDD']['year'], dtype=int)
            RCP_member_means[RCP][member]['summer_CPDD']['year'] = np.array(RCP_members[RCP][member]['summer_CPDD']['year'], dtype=int)
            RCP_member_means[RCP][member]['winter_CPDD']['year'] = np.array(RCP_members[RCP][member]['winter_CPDD']['year'], dtype=int)
            RCP_member_means[RCP][member]['snowfall']['year'] = np.array(RCP_members[RCP][member]['snowfall']['year'], dtype=int)
            RCP_member_means[RCP][member]['summer_snowfall']['year'] = np.array(RCP_members[RCP][member]['summer_snowfall']['year'], dtype=int)
            RCP_member_means[RCP][member]['winter_snowfall']['year'] = np.array(RCP_members[RCP][member]['winter_snowfall']['year'], dtype=int)
            RCP_member_means[RCP][member]['rain']['year'] = np.array(RCP_members[RCP][member]['rain']['year'], dtype=int)
            RCP_member_means[RCP][member]['summer_rain']['year'] = np.array(RCP_members[RCP][member]['summer_rain']['year'], dtype=int)
            RCP_member_means[RCP][member]['winter_rain']['year'] = np.array(RCP_members[RCP][member]['winter_rain']['year'], dtype=int)
            
            RCP_member_means[RCP][member]['discharge']['year'] = np.array(RCP_members[RCP][member]['snowfall']['year'], dtype=int)
            
            if(len(RCP_members[RCP][member]['SMB']['data'][year_idx]) > 0):
                RCP_member_means[RCP][member]['SMB']['data'].append(np.nanmean(RCP_members[RCP][member]['SMB']['data'][year_idx]))
                RCP_member_means[RCP][member]['area']['data'].append(np.nansum(RCP_members[RCP][member]['area']['data'][year_idx]))
                RCP_member_means[RCP][member]['volume']['data'].append(np.nansum(RCP_members[RCP][member]['volume']['data'][year_idx]))
                RCP_member_means[RCP][member]['zmean']['data'].append(np.nanmean(RCP_members[RCP][member]['zmean']['data'][year_idx]))
                RCP_member_means[RCP][member]['slope20']['data'].append(np.nanmean(RCP_members[RCP][member]['slope20']['data'][year_idx]))
                
                RCP_member_means[RCP][member]['CPDD']['data'].append(np.nanmean(RCP_members[RCP][member]['CPDD']['data'][year_idx]))
                RCP_member_means[RCP][member]['summer_CPDD']['data'].append(np.nanmean(RCP_members[RCP][member]['summer_CPDD']['data'][year_idx]))
                RCP_member_means[RCP][member]['winter_CPDD']['data'].append(np.nanmean(RCP_members[RCP][member]['winter_CPDD']['data'][year_idx]))
                RCP_member_means[RCP][member]['snowfall']['data'].append(np.nanmean(RCP_members[RCP][member]['snowfall']['data'][year_idx]))
                RCP_member_means[RCP][member]['summer_snowfall']['data'].append(np.nanmean(RCP_members[RCP][member]['summer_snowfall']['data'][year_idx]))
                RCP_member_means[RCP][member]['winter_snowfall']['data'].append(np.nanmean(RCP_members[RCP][member]['winter_snowfall']['data'][year_idx]))
                RCP_member_means[RCP][member]['rain']['data'].append(np.nanmean(RCP_members[RCP][member]['rain']['data'][year_idx]))
                RCP_member_means[RCP][member]['summer_rain']['data'].append(np.nanmean(RCP_members[RCP][member]['summer_rain']['data'][year_idx]))
                RCP_member_means[RCP][member]['winter_rain']['data'].append(np.nanmean(RCP_members[RCP][member]['winter_rain']['data'][year_idx]))
                member_annual_discharge = -1*np.nansum(np.asarray(RCP_members[RCP][member]['SMB']['data'][year_idx])*np.asarray(RCP_members[RCP][member]['area']['data'][year_idx]))
                member_annual_discharge = np.where(member_annual_discharge < 0, 0, member_annual_discharge)
                RCP_member_means[RCP][member]['discharge']['data'].append(member_annual_discharge)
                
            else:
                RCP_member_means[RCP][member]['SMB']['data'].append(np.nan)
                RCP_member_means[RCP][member]['area']['data'].append(np.nan)
                RCP_member_means[RCP][member]['volume']['data'].append(np.nan)
                RCP_member_means[RCP][member]['zmean']['data'].append(np.nan)
                RCP_member_means[RCP][member]['slope20']['data'].append(np.nan)
                
                RCP_member_means[RCP][member]['CPDD']['data'].append(np.nan)
                RCP_member_means[RCP][member]['summer_CPDD']['data'].append(np.nan)
                RCP_member_means[RCP][member]['winter_CPDD']['data'].append(np.nan)
                RCP_member_means[RCP][member]['snowfall']['data'].append(np.nan)
                RCP_member_means[RCP][member]['summer_snowfall']['data'].append(np.nan)
                RCP_member_means[RCP][member]['winter_snowfall']['data'].append(np.nan)
                RCP_member_means[RCP][member]['rain']['data'].append(np.nan)
                RCP_member_means[RCP][member]['summer_rain']['data'].append(np.nan)
                RCP_member_means[RCP][member]['winter_rain']['data'].append(np.nan)
                RCP_member_means[RCP][member]['discharge']['data'].append(np.nan)
                
#            if(year_idx == 83):
#                print("\nFinal volume: " + str(np.nansum(RCP_members[RCP][member]['volume']['data'][year_idx])))

    for year_idx in range(0, year_range.size):
        area_year, volume_year = [],[]
        if(filter_glacier):
            member_idx = len(RCP_members[RCP])
        #        print("len(RCP_members[RCP]): " + str(len(RCP_members[RCP])))
        #        print("member_idx: " + str(member_idx))
        for member in range(0, member_idx):
#        for member in range(0, member_idx-1):
            # RCP_means
            area_year.append(np.nansum(RCP_members[RCP][member]['area']['data'][year_idx]))
            volume_year.append(np.nansum(RCP_members[RCP][member]['volume']['data'][year_idx]))
        
        RCP_means[RCP]['area']['data'].append(np.nanmean(copy.deepcopy(area_year)))
        RCP_means[RCP]['area']['year'] = np.array(RCP_data[RCP]['area']['year'], dtype=int)
        RCP_means[RCP]['volume']['data'].append(np.nanmean(copy.deepcopy(volume_year)))
        RCP_means[RCP]['volume']['year'] = np.array(RCP_data[RCP]['volume']['year'], dtype=int)
 
#print(overall_annual_mean)

##########    PLOTS    #######################
if(filter_glacier):
    header = glacier_ID_filter + "_"
else:
    header = "french_alps_avg_"

############  Area and volume   #####################################################################################

#############       Plot each one of the RCP-GCM-RCM combinations       #############################################
    
fig1, ax1 = plot.subplots(ncols=2, nrows=1, axwidth=4, share=0)
if(filter_glacier):
    fig1.suptitle("Glacier " + glacier_name_filter + " glacier projections under climate change")
else:
    fig1.suptitle("Regional average French alpine glacier projections under climate change")
ax1[0].set_ylabel('Volume (m$^3$ 10$^6$)')
ax1[0].set_xlabel('Year')

ax1.format(
        abc=True, abcloc='ul',
        ygridminor=True,
        ytickloc='both', yticklabelloc='left'
)

# Volume
plot_individual_members(ax1[0], RCP_member_means, RCP_means, 'volume', filtered_member)
plot_RCP_means(ax1[0], RCP_means, 'volume', with_26)

ax1[1].set_ylabel('Area (km$^2$)')
ax1[1].set_xlabel('Year')

# Area
plot_individual_members(ax1[1], RCP_member_means, RCP_means, 'area', filtered_member)
plot_RCP_means(ax1[1], RCP_means, 'area', with_26)

# Save as PDF
save_plot_as_pdf(fig1, header + 'volume_area', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_glacier_area, 'area', RCP_means)
store_RCP_mean(path_glacier_volume, 'volume', RCP_means)



###############     Plot the evolution of topographical parameters    ####################################
###############     Zmean and slope      #################################################################

fig2, (ax21, ax22) = plot.subplots(ncols=2, nrows=1, axwidth=4, share=0)
if(filter_glacier):
    fig2.suptitle("Glacier " + glacier_name_filter + " glacier projections under climate change")
else:
    fig2.suptitle("Regional average French alpine glacier projections under climate change")
ax21.set_ylabel('Mean glacier altitude (m)')
ax21.set_xlabel('Year')

ax22.set_ylabel('Mean glacier slope of the lowermost 20% altitudinal range (°)')
ax22.set_xlabel('Year')

# Mean altitude
plot_individual_members(ax21, RCP_member_means, RCP_means, 'zmean', filtered_member)
plot_RCP_means(ax21, RCP_means, 'zmean', with_26)

# Slope 20% altitudinal range
plot_individual_members(ax22, RCP_member_means, RCP_means, 'slope20', filtered_member)
plot_RCP_means(ax22, RCP_means, 'slope20', with_26)

# Save as PDF
save_plot_as_pdf(fig2, header + 'zmean_slope', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_glacier_zmean, 'zmean', RCP_means)
store_RCP_mean(path_glacier_slope20, 'slope20', RCP_means)


###############     Plot the evolution of temperature and snowfall    ####################################
###############     Winter and summer snowfall, rain and temperature  ####################################

fig3, axs3 = plot.subplots(ncols=3, nrows=3, aspect=2, axwidth=2, spany=0)
if(filter_glacier):
    fig3.suptitle("Glacier " + glacier_name_filter + " climate projections")
else:
    fig3.suptitle("French Alpine glaciers average regional climate signal evolution")

axs3.format(
        abc=True, abcloc='ul',
        ygridminor=True,
        ytickloc='both', yticklabelloc='left',
        xlabel='Year'
)

titles = ['Annual CPDD', 'Summer CPDD', 'Winter CPDD', 'Annual snowfall', 'Summer snowfall', 'Winter snowfall', 'Annual rainfall', 'Summer rainfall', 'Winter rainfall']
ylabels = ['PDD', 'PDD', 'PDD', 'mm', 'mm', 'mm', 'mm', 'mm', 'mm']
for i, s_title, s_ylabel in zip(range(0, 9), titles, ylabels):
    if(s_title[0:6] == 'Winter'):
        season_color = 'midnightblue'
    elif(s_title[0:6] == 'Summer'):
       season_color = 'darkred'
    else:
        season_color = 'k'
    
    axs3[i].set_title(s_title, color=season_color)  
    axs3[i].format(ylabel=s_ylabel)

# CPDD
plot_individual_members(axs3[0], RCP_member_means, RCP_means, 'CPDD', filtered_member, alpha=0.2)
plot_RCP_means(axs3[0], RCP_means, 'CPDD', with_26, legend=False, linewidth=2)

# Summer CPDD
plot_individual_members(axs3[1], RCP_member_means, RCP_means, 'summer_CPDD', filtered_member, alpha=0.2)
plot_RCP_means(axs3[1], RCP_means, 'summer_CPDD', with_26, legend=False, linewidth=2)

# Winter CPDD
plot_individual_members(axs3[2], RCP_member_means, RCP_means, 'winter_CPDD', filtered_member, alpha=0.2)
plot_RCP_means(axs3[2], RCP_means, 'winter_CPDD', with_26, legend=True, linewidth=2)

# Snowfall
plot_individual_members(axs3[3], RCP_member_means, RCP_means, 'snowfall', filtered_member, alpha=0.2)
plot_RCP_means(axs3[3], RCP_means, 'snowfall', with_26, legend=False, linewidth=2)

# Summer Snowfall
plot_individual_members(axs3[4], RCP_member_means, RCP_means, 'summer_snowfall', filtered_member, alpha=0.2)
plot_RCP_means(axs3[4], RCP_means, 'summer_snowfall', with_26, legend=False, linewidth=2)

# Winter Snowfall
plot_individual_members(axs3[5], RCP_member_means, RCP_means, 'winter_snowfall', filtered_member, alpha=0.2)
plot_RCP_means(axs3[5], RCP_means, 'winter_snowfall', with_26, legend=True, linewidth=2)

# Rainfall
plot_individual_members(axs3[6], RCP_member_means, RCP_means, 'rain', filtered_member, alpha=0.2)
plot_RCP_means(axs3[6], RCP_means, 'rain', with_26, legend=False, linewidth=2)

# Summer rainfall
plot_individual_members(axs3[7], RCP_member_means, RCP_means, 'summer_rain', filtered_member, alpha=0.2)
plot_RCP_means(axs3[7], RCP_means, 'summer_rain', with_26, legend=False, linewidth=2)

# Winter rainfall
plot_individual_members(axs3[8], RCP_member_means, RCP_means, 'winter_rain', filtered_member, alpha=0.2)
plot_RCP_means(axs3[8], RCP_means, 'winter_rain', with_26, legend=True, linewidth=2)

# Save as PDF
save_plot_as_pdf(fig3, header + 'CPDD_snowfall_rain', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_glacier_CPDDs, 'CPDD', RCP_means)
store_RCP_mean(path_glacier_snowfall, 'snowfall', RCP_means)

###############     Plot the glacier-wide SMB   ####################################
fig4, (ax41) = plot.subplots(ncols=1, nrows=1, axwidth=5, aspect=2)
ax41.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
if(filter_glacier):
    fig4.suptitle("Glacier " + glacier_name_filter + " glacier-wide SMB evolution under climate change")
else:
    fig4.suptitle("Average glacier-wide SMB projections of French alpine glaciers under climate change")
ax41.set_ylabel('Glacier-wide SMB (m.w.e. a$^-1$)')
ax41.set_xlabel('Year')

# Glacier-wide SMB
plot_individual_members(ax41, RCP_member_means, RCP_means, 'SMB', filtered_member)
plot_RCP_means(ax41, RCP_means, 'SMB', with_26)

# Save as PDF
save_plot_as_pdf(fig4, header + 'SMB', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_smb_simulations, 'SMB', RCP_means)

###############     Plot the glacier meltwater discharge   ####################################
fig5, (ax51) = plot.subplots(ncols=1, nrows=1, axwidth=5, aspect=2)
ax51.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
if(filter_glacier):
    fig5.suptitle("Glacier " + glacier_name_filter + " meltwater discharge evolution under climate change")
else:
    fig5.suptitle("Average meltwater discharge projections of French alpine glacier under climate change")
ax51.set_ylabel('Meltwater discharge (m$^3$ 10$^6$)')
ax51.set_xlabel('Year')

# Glacier meltwater discharge
plot_individual_members(ax51, RCP_member_means, RCP_means, 'discharge', filtered_member)
plot_RCP_means(ax51, RCP_means, 'discharge', with_26)

# Save as PDF
save_plot_as_pdf(fig5, header + 'meltwater_discharge', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_glacier_discharge, 'discharge', RCP_means)


#plt.show()