# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:30:22 2020

@author: Jordi Bolibar

PROCESSING AND PLOTTING DATA OF CLIMATE AND TOPOGRAPHICAL VARIABLES OF PROJECTIONS 
OF FRENCH ALPINE GLACIERS (2015-2100)

"""

## Dependencies: ##
import matplotlib.pyplot as plt
import proplot as plot
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path
import pickle
import pandas as pd
import xarray as xr
import numpy.polynomial.polynomial as poly

###### FLAGS  #########
with_26 = True
filter_glacier = False
static_geometry = False
load_dictionaries = True
other_plots = False
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

##################### Lasso glacier evolution simulations  ######################
path_glacier_evolution = os.path.join(workspace, 'glacier_data', 'glacier_evolution', 'lasso')
path_MB = os.path.join(workspace, 'glacier_data', 'SMB')
path_MB_simulations = os.path.join(path_MB, 'SMB_simulations', 'lasso')

path_glims = os.path.join(workspace, 'glacier_data', 'GLIMS') 

path_RCP_means_all = os.path.join(path_glacier_evolution, "RCP_means")

path_glogem = "C:\\Jordi\\PhD\\Data\\SMB\\GloGEM\\"

####################  Full glacier evolution simulations   ###########################

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

# Listing all subfolders
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
path_MB_root = np.asarray(os.listdir(os.path.join(path_MB_simulations, "projections")))


glims_2003 = genfromtxt(os.path.join(path_glims, 'GLIMS_2003.csv'), delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), 
                        ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), 
                        ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8')])
glacier_name_filter = glims_2003['Glacier'][glims_2003['GLIMS_ID'] == glacier_ID_filter.encode('UTF-8')]
glacier_name_filter = glacier_name_filter[0].decode('UTF-8')
#print("\nFiltered glacier name: " + str(glacier_name_filter))

proj_blob = {'year':[], 'data':[]}
annual_mean = {'MB':copy.deepcopy(proj_blob), 'area':copy.deepcopy(proj_blob), 'volume':copy.deepcopy(proj_blob), 
               'zmean':copy.deepcopy(proj_blob), 'slope20':copy.deepcopy(proj_blob), 'avg_area':copy.deepcopy(proj_blob),
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

# Dictionary of glacier evolution data

raw_members = os.listdir(os.path.join(path_glacier_area, "projections"))
raw_IDs = os.listdir(os.path.join(path_glacier_area, "projections", "FORCING_CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5_RCP45_alp_2005080106_2100080106", "1"))
members, glims_IDs = np.array([]), np.array([])

# Fetch unique model members
for member in raw_members:
    if(not np.any(members == member[8:-32])):
        members = np.append(members, member[8:-32])
        
for ID in raw_IDs:
    if(not np.any(glims_IDs == ID[:-9])):
        glims_IDs = np.append(glims_IDs, ID[:-9])
        
massif_IDs = np.array([1,3,6,8,9,10,11,12,13,15,16,19,21])       
        
index_list = {'GLIMS_ID':glims_IDs,
              'massif_ID': massif_IDs,
              'RCP': np.array(['26', '45', '85']),
              'member': np.asarray(members),
              'year': np.asarray(range(year_start, 2100))}

dummy_coords = np.empty((index_list['GLIMS_ID'].size, index_list['massif_ID'].size, index_list['RCP'].size, index_list['member'].size, index_list['year'].size))
dummy_coords[:] = np.nan

glacier_projections_dict = {'MB':copy.deepcopy(dummy_coords), 'area':copy.deepcopy(dummy_coords), 'volume':copy.deepcopy(dummy_coords), 
                            'zmean':copy.deepcopy(dummy_coords), 'slope20':copy.deepcopy(dummy_coords), 
                            'CPDD':copy.deepcopy(dummy_coords), 's_CPDD':copy.deepcopy(dummy_coords), 'w_CPDD':copy.deepcopy(dummy_coords), 
                            'snowfall':copy.deepcopy(dummy_coords), 'w_snowfall':copy.deepcopy(dummy_coords), 's_snowfall':copy.deepcopy(dummy_coords),
                            'rain':copy.deepcopy(dummy_coords), 's_rain':copy.deepcopy(dummy_coords), 'w_rain':copy.deepcopy(dummy_coords), 
                            'discharge':copy.deepcopy(dummy_coords)}

# Array of indexes of the data structure to iterate
data_idxs = ['MB','area','volume','zmean','slope20', 'avg_area',
             'CPDD', 'summer_CPDD', 'winter_CPDD', 
             'snowfall', 'summer_snowfall', 'winter_snowfall', 
             'rain', 'summer_rain', 'winter_rain', 'discharge']

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
                
                config_dictionary_file = os.path.join(path_RCP_means, "RCP" + str(RCP) + "_glacier_with_26_" + str(variable) + "_" + str(years[0])+ "_" + str(years[-1]))
                with open('config.dictionary', 'wb') as config_dictionary_file:
                    pickle.dump(RCP_means, config_dictionary_file)
            else:
                np.savetxt(os.path.join(path_RCP_means, "RCP" + str(RCP) + "_glacier_" + str(variable) + "_" + str(years[0])+ "_" + str(years[-1]) + '.csv'), 
                           data_years, delimiter=";", fmt="%s")
                
                config_dictionary_file = os.path.join(path_RCP_means, "RCP" + str(RCP) + "_glacier_" + str(variable) + "_" + str(years[0])+ "_" + str(years[-1]))
                with open('config.dictionary', 'wb') as config_dictionary_file:
                    pickle.dump(RCP_means, config_dictionary_file)
                
def plot_individual_members(ax, RCP_member_means, RCP_means, variable, filtered_member, alpha=0.3):
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
        
def plot_individual_members_diff(ax, RCP_member_means, RCP_means, variable, static_variable, filtered_member, alpha=0.15):
    member_idx = 0
    for member_85 in RCP_member_means['85']:
        data_85 = np.asarray(member_85[variable]['data'])
        static_data_85 = np.asarray(member_85[static_variable]['data'])
        if(len(data_85) > 0 and (member_idx == filtered_member or filtered_member == -1)):
            if(len(RCP_means['85'][variable]['year']) > len(data_85)):
                ax.plot(RCP_means['85'][variable]['year'][:-1], data_85 - static_data_85, linewidth=0.1, alpha=alpha, c='dark slate blue')
            else:
                ax.plot(RCP_means['85'][variable]['year'], data_85 - static_data_85, linewidth=0.1, alpha=alpha, c='dark slate blue')
    member_idx=member_idx+1
    member_idx = 0
    for member_26 in RCP_member_means['26']:
        data_26 = np.asarray(member_26[variable]['data'])
        static_data_26 = np.asarray(member_26[static_variable]['data'])
        if(len(data_26) > 0 and (member_idx == filtered_member or filtered_member == -1)):
            if(len(RCP_means['26'][variable]['year']) > len(data_26)):
                ax.plot(RCP_means['26'][variable]['year'][:-1], data_26 - static_data_26, linewidth=0.1, alpha=alpha, c='steelblue')
            else:
                ax.plot(RCP_means['26'][variable]['year'], data_26 - static_data_26, linewidth=0.1, alpha=alpha, c='steelblue')
        member_idx=member_idx+1
    member_idx = 0
    for member_45 in RCP_member_means['45']:
        data_45 = np.asarray(member_45[variable]['data'])
        static_data_45 = np.asarray(member_45[static_variable]['data'])
        if(len(data_45) > 0 and (member_idx == filtered_member or filtered_member == -1)):
            if(len(RCP_means['45'][variable]['year']) > len(data_45)):
                ax.plot(RCP_means['45'][variable]['year'][:-1], data_45 - static_data_45, linewidth=0.1, alpha=alpha+0.1, c='cyan7')
            else:
                ax.plot(RCP_means['45'][variable]['year'], data_45 - static_data_45, linewidth=0.1, alpha=alpha+0.1, c='cyan7')
        member_idx=member_idx+1
        
def plot_RCP_means(ax, RCP_means, variable, with_26, legend=False, linewidth=2, linestyle='-'):
    if(legend):
        legend_pos='ur'
    else:
        legend_pos=''
        
    h1 = ''
    if(with_26):
        h1 = ax.plot(RCP_means['26'][variable]['year'][:-1], np.asarray(RCP_means['26'][variable]['data'][:-1]), linewidth=linewidth, linestyle=linestyle, label='RCP 2.6', c='steelblue', legend=legend_pos)
    h3 = ax.plot(RCP_means['85'][variable]['year'][:-1], np.asarray(RCP_means['85'][variable]['data'][:-1]), linewidth=linewidth, linestyle=linestyle, label='RCP 8.5', c='darkred', legend=legend_pos)
    h2 = ax.plot(RCP_means['45'][variable]['year'][:-1], np.asarray(RCP_means['45'][variable]['data'][:-1]), linewidth=linewidth, linestyle=linestyle, label='RCP 4.5', c='brown orange', legend=legend_pos)

    
    return ((h1, h2, h3))
    
def plot_RCP_means_diff(ax, RCP_means, variable, static_variable, with_26, legend=False, linewidth=2, linestyle='-'):
    if(legend):
        if(variable == 'winter_CPDD' or variable == 'winter_rain'):
            legend_pos = 'lr'
        else:
            legend_pos='ur'
    else:
        legend_pos=''
    h1 = ''
    h3 = ax.plot(RCP_means['85'][variable]['year'][:-1], np.asarray(RCP_means['85'][variable]['data'][:-1]) - np.asarray(RCP_means['85'][static_variable]['data'][:-1]), linewidth=linewidth, linestyle=linestyle, label='RCP 8.5', c='dark slate blue', legend=legend_pos)
    h2 = ax.plot(RCP_means['45'][variable]['year'][:-1], np.asarray(RCP_means['45'][variable]['data'][:-1]) - np.asarray(RCP_means['45'][static_variable]['data'][:-1]), linewidth=linewidth, linestyle=linestyle, label='RCP 4.5', c='cyan7', legend=legend_pos)
    if(with_26):
        h1 = ax.plot(RCP_means['26'][variable]['year'][:-1], np.asarray(RCP_means['26'][variable]['data'][:-1]) - np.asarray(RCP_means['26'][static_variable]['data'][:-1]), linewidth=linewidth, linestyle=linestyle, label='RCP 2.6', c='steelblue', legend=legend_pos)
           
    return ((h1, h2, h3))
    
def return_indexes(index_list, glims_ID, massif_ID, RCP, current_member, year):
    
    glims_idx = np.where(index_list['GLIMS_ID'] == glims_ID)
    massif_idx = np.where(index_list['massif_ID'] == massif_ID)
    RCP_idx = np.where(index_list['RCP'] == RCP)
    member_idx = np.where(index_list['member'] == current_member)
    year_idx = np.where(index_list['year'] == year)
    
    return ((glims_idx, massif_idx, RCP_idx, member_idx, year_idx))

    
##################################################################################################
        
        
###############################################################################
###                           MAIN                                          ###
###############################################################################

# Data reading and processing
    
if(not load_dictionaries):

    print("\nReading files and creating data structures...")
    
    # Iterate different RCP-GCM-RCM combinations
    member_26_idx, member_45_idx, member_85_idx = 0, 0, 0
    first = True
    root_paths = zip(path_MB_root, path_area_root, path_melt_years_root, path_slope20_root, path_volume_root, path_zmean_root, 
                     path_s_CPDDs_root, path_w_CPDDs_root,
                     path_s_snowfall_root, path_w_snowfall_root,
                     path_s_rain_root, path_w_rain_root)
    for path_forcing_MB, path_forcing_area, path_forcing_melt_years, path_forcing_slope20, path_forcing_volume, path_forcing_zmean, path_forcing_s_CPDDs, path_forcing_w_CPDDs, path_forcing_s_snowfall, path_forcing_w_snowfall, path_forcing_s_rain, path_forcing_w_rain in root_paths:
        
        current_RCP = path_forcing_area[-28:-26]
        current_member = path_forcing_area[8:-32]
        
        # Filter members depending if we want to include RCP 2.6 or not
#        if((with_26 and np.any(current_member == members_with_26)) or (not with_26 and current_RCP != '26')):
        if(True):
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
            
            ### Full glacier evolution projections  ####
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
            path_MB_glaciers = np.asarray(os.listdir(os.path.join(path_MB_simulations, "projections", path_forcing_MB)))
            
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
            volume_scale_paths = zip(path_MB_glaciers, path_area_glaciers, path_volume_glaciers, path_zmean_glaciers, path_slope20_glaciers, 
                                     path_s_CPDDs_glaciers, path_w_CPDDs_glaciers,
                                     path_s_snowfall_glaciers, path_w_snowfall_glaciers,
                                     path_s_rain_glaciers, path_w_rain_glaciers)
            for path_MB_scaled, path_area_scaled, path_volume_scaled, path_zmean_scaled, path_slope20_scaled, path_s_CPDDs_scaled, path_w_CPDDs_scaled, path_s_snowfall_scaled, path_w_snowfall_scaled, path_s_rain_scaled, path_w_rain_scaled in volume_scale_paths:
                
                ### Full glacier evolution projections  ###
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
                path_MB_glaciers_scaled = np.asarray(os.listdir(os.path.join(path_MB_simulations, "projections", path_forcing_MB, path_MB_scaled)))
                
                glacier_count = 0
    #            if(path_area_scaled == '1'):
                if(path_area_scaled == '1' and path_s_CPDDs_glaciers_scaled.size > 369):
                    bump_member = True
                    glacier_paths = zip(path_MB_glaciers_scaled, path_area_glaciers_scaled, path_volume_glaciers_scaled, path_zmean_glaciers_scaled, path_slope20_glaciers_scaled, 
                                        path_s_CPDDs_glaciers_scaled, path_w_CPDDs_glaciers_scaled,
                                        path_s_snowfall_glaciers_scaled, path_w_snowfall_glaciers_scaled,
                                        path_s_rain_glaciers_scaled, path_w_rain_glaciers_scaled)
                    for path_MB, path_area, path_volume, path_zmean, path_slope20, path_s_CPDD, path_w_CPDD, path_s_snowfall, path_w_snowfall, path_s_rain, path_w_rain in glacier_paths:
                        
                        
        #                print("path_MB[:13]: " + str(path_MB[:13]))
        #                print("glacier_ID_filter: " + str(glacier_ID_filter))
        #                if(path_MB[:14] == glacier_ID_filter):
                        
                        # We extract the GLIMS ID for the glacier
                        if(path_MB[16] == '2' or path_MB[16] == '3' or path_MB[16] == '4'):
                            glims_ID = path_MB[:17]
                        else:
                            glims_ID = path_MB[:14]
                            
                         # We retrieve its SAFRAN massif ID
                        glims_2003_idx = np.where(glims_2003['GLIMS_ID'] == glims_ID.encode('UTF-8'))
                        
                        if(len(glims_2003_idx) > 1):
                            glims_2003_idx = glims_2003_idx[0]
                        massif_ID = glims_2003['Massif_SAFRAN'][glims_2003_idx]
                        massif_ID = massif_ID[0]
                        
                        if((filter_glacier and glacier_ID_filter == glims_ID) or not filter_glacier):
                            
                            ### Full glacier evolution projections  ###
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
                            MB_glacier = genfromtxt(os.path.join(path_MB_simulations, "projections", path_forcing_MB, path_MB_scaled, path_MB), delimiter=';')
                            
                            if(len(MB_glacier.shape) > 1):
                                for year in range(year_start, 2100):
                                    for data_idx in data_idxs:
                                        if((current_RCP == '26' and first_26) or (current_RCP == '45' and first_45) or (current_RCP == '85' and first_85)):
                                            RCP_data[current_RCP][data_idx]['data'].append([])
                                            RCP_data[current_RCP][data_idx]['year'].append(year)
                                
                                # Add glacier data to blob separated by year
                                year_idx = 0
                                
    #                            print("\nmember_idx: " + str(member_idx))
                                years_path = zip(MB_glacier, area_glacier, volume_glacier, zmean_glacier, slope20_glacier, 
                                                 s_CPDD_glacier, w_CPDD_glacier,
                                                 s_snowfall_glacier, w_snowfall_glacier,
                                                 s_rain_glacier, w_rain_glacier)
                                
                                year=year_start
                                for MB_y, area_y, volume_y, zmean_y, slope20_y, s_CPDD_y, w_CPDD_y, s_snowfall_y, w_snowfall_y, s_rain_y, w_rain_y in years_path:
                                    
                                    annual_discharge = -1*area_glacier[0][1]*MB_y[1]
                                    if(annual_discharge < 0):
                                        annual_discharge = 0
                                    
                                    # Fill the individual glacier projections dictionary
                                    # We get the coordinates of the N-dimensional space for the current iteration
                                    current_idxs = return_indexes(index_list, glims_ID, massif_ID, current_RCP, current_member, year)
                                    
                                    glacier_projections_dict['MB'][current_idxs] = MB_y[1]
                                    glacier_projections_dict['area'][current_idxs] = area_y[1]
                                    glacier_projections_dict['volume'][current_idxs] = volume_y[1]
                                    glacier_projections_dict['zmean'][current_idxs] = zmean_y[1]
                                    glacier_projections_dict['slope20'][current_idxs] = slope20_y[1]
                                    glacier_projections_dict['CPDD'][current_idxs] = s_CPDD_y[1] + w_CPDD_y[1]
                                    glacier_projections_dict['s_CPDD'][current_idxs] = s_CPDD_y[1]
                                    glacier_projections_dict['w_CPDD'][current_idxs] = w_CPDD_y[1]
                                    glacier_projections_dict['snowfall'][current_idxs] = s_snowfall_y[1] + w_snowfall_y[1]
                                    glacier_projections_dict['s_snowfall'][current_idxs] = s_snowfall_y[1]
                                    glacier_projections_dict['w_snowfall'][current_idxs] = w_snowfall_y[1]
                                    glacier_projections_dict['rain'][current_idxs] = s_rain_y[1] + w_rain_y[1]
                                    glacier_projections_dict['s_rain'][current_idxs] = s_rain_y[1]
                                    glacier_projections_dict['w_rain'][current_idxs] = w_rain_y[1]
                                    glacier_projections_dict['discharge'][current_idxs] = annual_discharge
                                        
                                    ### Full glacier evolution projections  ###
                                    RCP_data[current_RCP]['MB']['data'][year_idx].append(MB_y[1])
                                    RCP_data[current_RCP]['area']['data'][year_idx].append(area_y[1])
                                    RCP_data[current_RCP]['volume']['data'][year_idx].append(volume_y[1])
                                    RCP_data[current_RCP]['zmean']['data'][year_idx].append(zmean_y[1])
                                    RCP_data[current_RCP]['slope20']['data'][year_idx].append(slope20_y[1])
                                    RCP_data[current_RCP]['avg_area']['data'][year_idx].append(area_y[1])
                                    
                                    RCP_data[current_RCP]['CPDD']['data'][year_idx].append(s_CPDD_y[1] + w_CPDD_y[1])
                                    RCP_data[current_RCP]['summer_CPDD']['data'][year_idx].append(s_CPDD_y[1])
                                    RCP_data[current_RCP]['winter_CPDD']['data'][year_idx].append(w_CPDD_y[1])
                                    RCP_data[current_RCP]['snowfall']['data'][year_idx].append(s_snowfall_y[1] + w_snowfall_y[1])
                                    RCP_data[current_RCP]['summer_snowfall']['data'][year_idx].append(s_snowfall_y[1])
                                    RCP_data[current_RCP]['winter_snowfall']['data'][year_idx].append(w_snowfall_y[1])
                                    RCP_data[current_RCP]['rain']['data'][year_idx].append(s_rain_y[1] + w_rain_y[1])
                                    RCP_data[current_RCP]['summer_rain']['data'][year_idx].append(s_rain_y[1])
                                    RCP_data[current_RCP]['winter_rain']['data'][year_idx].append(w_rain_y[1])
                                    RCP_data[current_RCP]['discharge']['data'][year_idx].append(annual_discharge)
                                    
                                    ################### Add data to blob separated by RCP-GCM-RCM members  #######################
                                    ### Full glacier evolution projections  ###
                                    RCP_members[current_RCP][member_idx]['MB']['data'][year_idx].append(MB_y[1])
                                    RCP_members[current_RCP][member_idx]['area']['data'][year_idx].append(area_y[1])
                                    RCP_members[current_RCP][member_idx]['volume']['data'][year_idx].append(volume_y[1])
                                    RCP_members[current_RCP][member_idx]['zmean']['data'][year_idx].append(zmean_y[1])
                                    RCP_members[current_RCP][member_idx]['slope20']['data'][year_idx].append(slope20_y[1])
                                    RCP_members[current_RCP][member_idx]['avg_area']['data'][year_idx].append(area_y[1])
                                    
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
                                    year = year+1
                                
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
            ### Full glacier evolution projections  ###
            RCP_means[RCP]['MB']['data'].append(np.nanmean(RCP_data[RCP]['MB']['data'][year_idx]))
            RCP_means[RCP]['MB']['year'] = np.array(RCP_data[RCP]['MB']['year'], dtype=int)
            RCP_means[RCP]['zmean']['data'].append(np.nanmean(RCP_data[RCP]['zmean']['data'][year_idx]))
            RCP_means[RCP]['zmean']['year'] = np.array(RCP_data[RCP]['zmean']['year'], dtype=int)
            RCP_means[RCP]['slope20']['data'].append(np.nanmean(RCP_data[RCP]['slope20']['data'][year_idx]))
            RCP_means[RCP]['slope20']['year'] = np.array(RCP_data[RCP]['slope20']['year'], dtype=int)
            RCP_means[RCP]['avg_area']['data'].append(np.nanmean(RCP_data[RCP]['avg_area']['data'][year_idx]))
            RCP_means[RCP]['avg_area']['year'] = np.array(RCP_data[RCP]['avg_area']['year'], dtype=int)
            
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
            
            annual_discharge = -1*np.nansum(np.asarray(RCP_data[RCP]['MB']['data'][year_idx])*np.asarray(RCP_data[RCP]['area']['data'][year_idx]))/member_idx
            annual_discharge = np.where(annual_discharge < 0, 0, annual_discharge)
            RCP_means[RCP]['discharge']['data'].append(annual_discharge)
            RCP_means[RCP]['discharge']['year'] = np.array(RCP_data[RCP]['snowfall']['year'], dtype=int)
            
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
                
                ### Full glacier evolution projections  ###
                RCP_member_means[RCP][member]['MB']['year'] = np.array(RCP_members[RCP][member]['MB']['year'], dtype=int)
                RCP_member_means[RCP][member]['area']['year'] = np.array(RCP_members[RCP][member]['area']['year'], dtype=int)
                RCP_member_means[RCP][member]['volume']['year'] = np.array(RCP_members[RCP][member]['volume']['year'], dtype=int)
                RCP_member_means[RCP][member]['zmean']['year'] = np.array(RCP_members[RCP][member]['zmean']['year'], dtype=int)
                RCP_member_means[RCP][member]['slope20']['year'] = np.array(RCP_members[RCP][member]['slope20']['year'], dtype=int)
                RCP_member_means[RCP][member]['avg_area']['year'] = np.array(RCP_members[RCP][member]['avg_area']['year'], dtype=int)
                
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
                
                if(len(RCP_members[RCP][member]['MB']['data'][year_idx]) > 0):
                    ### Full glacier evolution projections  ###
                    RCP_member_means[RCP][member]['MB']['data'].append(np.nanmean(RCP_members[RCP][member]['MB']['data'][year_idx]))
                    RCP_member_means[RCP][member]['area']['data'].append(np.nansum(RCP_members[RCP][member]['area']['data'][year_idx]))
                    RCP_member_means[RCP][member]['volume']['data'].append(np.nansum(RCP_members[RCP][member]['volume']['data'][year_idx]))
                    RCP_member_means[RCP][member]['zmean']['data'].append(np.nanmean(RCP_members[RCP][member]['zmean']['data'][year_idx]))
                    RCP_member_means[RCP][member]['slope20']['data'].append(np.nanmean(RCP_members[RCP][member]['slope20']['data'][year_idx]))
                    RCP_member_means[RCP][member]['avg_area']['data'].append(np.nanmean(RCP_members[RCP][member]['avg_area']['data'][year_idx]))
                    
                    RCP_member_means[RCP][member]['CPDD']['data'].append(np.nanmean(RCP_members[RCP][member]['CPDD']['data'][year_idx]))
                    RCP_member_means[RCP][member]['summer_CPDD']['data'].append(np.nanmean(RCP_members[RCP][member]['summer_CPDD']['data'][year_idx]))
                    RCP_member_means[RCP][member]['winter_CPDD']['data'].append(np.nanmean(RCP_members[RCP][member]['winter_CPDD']['data'][year_idx]))
                    RCP_member_means[RCP][member]['snowfall']['data'].append(np.nanmean(RCP_members[RCP][member]['snowfall']['data'][year_idx]))
                    RCP_member_means[RCP][member]['summer_snowfall']['data'].append(np.nanmean(RCP_members[RCP][member]['summer_snowfall']['data'][year_idx]))
                    RCP_member_means[RCP][member]['winter_snowfall']['data'].append(np.nanmean(RCP_members[RCP][member]['winter_snowfall']['data'][year_idx]))
                    RCP_member_means[RCP][member]['rain']['data'].append(np.nanmean(RCP_members[RCP][member]['rain']['data'][year_idx]))
                    RCP_member_means[RCP][member]['summer_rain']['data'].append(np.nanmean(RCP_members[RCP][member]['summer_rain']['data'][year_idx]))
                    RCP_member_means[RCP][member]['winter_rain']['data'].append(np.nanmean(RCP_members[RCP][member]['winter_rain']['data'][year_idx]))
                    member_annual_discharge = -1*np.nansum(np.asarray(RCP_members[RCP][member]['MB']['data'][year_idx])*np.asarray(RCP_members[RCP][member]['area']['data'][year_idx]))
                    member_annual_discharge = np.where(member_annual_discharge < 0, 0, member_annual_discharge)
                    RCP_member_means[RCP][member]['discharge']['data'].append(member_annual_discharge)
                    
                else:
                    ### Full glacier evolution projections  ###
                    RCP_member_means[RCP][member]['MB']['data'].append(np.nan)
                    RCP_member_means[RCP][member]['area']['data'].append(np.nan)
                    RCP_member_means[RCP][member]['volume']['data'].append(np.nan)
                    RCP_member_means[RCP][member]['zmean']['data'].append(np.nan)
                    RCP_member_means[RCP][member]['slope20']['data'].append(np.nan)
                    RCP_member_means[RCP][member]['avg_area']['data'].append(np.nan)
                    
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
        print("member_idx: " + str(member_idx))
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
        

    ### We save the dictionaries in order to avoid reprocessing them every time
    
            with open(os.path.join(path_RCP_means_all, "RCP_means_lasso.txt"), 'wb') as rcpmeans_f:
                                np.save(rcpmeans_f, RCP_means)
            with open(os.path.join(path_RCP_means_all, "RCP_member_means_lasso.txt"), 'wb') as rcpmmeans_f:
                                np.save(rcpmmeans_f, RCP_member_means)
                                
else:
    # We load the previously stored data dictionaries
    with open(os.path.join(path_RCP_means_all, "RCP_means_lasso.txt"), 'rb') as rcpmeans_f:
            RCP_means = np.load(rcpmeans_f,  allow_pickle=True)[()]
            
    with open(os.path.join(path_RCP_means_all, "RCP_member_means_lasso.txt"), 'rb') as rcpmmeans_f:
            RCP_member_means = np.load(rcpmmeans_f,  allow_pickle=True)[()]
 
#print(overall_annual_mean)

# Transfer dictionary to xarray dataset   
if(not load_dictionaries):    
    ds_glacier_projections_lasso = xr.Dataset(data_vars={'MB': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['MB']),
                                                   'area': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['area']),
                                                   'volume': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['volume']),
                                                   'zmean': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['zmean']),
                                                   'slope20': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['slope20']),
                                                   'CPDD': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['CPDD']),
                                                   's_CPDD': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['s_CPDD']),
                                                   'w_CPDD': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['w_CPDD']),
                                                   'snowfall': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['snowfall']),
                                                   'w_snowfall': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['w_snowfall']),
                                                   's_snowfall': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['s_snowfall']),
                                                   'rain': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['rain']),
                                                   'w_rain': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['w_rain']),
                                                   's_rain': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['s_rain']),
                                                   'discharge': (('GLIMS_ID', 'massif_ID', 'RCP', 'member', 'year'), glacier_projections_dict['discharge'])},
                                                coords={'GLIMS_ID': index_list['GLIMS_ID'], 
                                                        'massif_ID': index_list['massif_ID'], 
                                                        'RCP': index_list['RCP'], 
                                                        'member': index_list['member'],
                                                        'year': index_list['year']},
                                                attrs={'README': "This dataset contains highly sparse data among NaN values. massif_IDs do not contain all GLIMS_IDs. Luckily, xarray filters all NaN values for data processing.",
                                                       'Content': "French Alpine glacier evolution projections (2015-2100) from the ALpine Parametrized Glacier Model",
                                                       'Author': "Jordi Bolibar",
                                                       'Affiliation': "Institute of Environmental Geosciences (University Grenoble Alpes / INRAE)"})
    
    
    
    # We save the whole dataset in a single netCDF file
    # We compress the files due to its sparsity
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_glacier_projections_lasso.data_vars}
    ds_glacier_projections_lasso.to_netcdf(os.path.join(path_glacier_evolution, 'glacier_evolution_lasso_' + str(year_start) + '_2100.nc'), encoding=encoding)
    
    # We transform the dataset to dataframe without data gaps
    df_glacier_projections = ds_glacier_projections_lasso.to_dataframe().dropna(how='any')
    df_glacier_projections.to_csv(os.path.join(path_glacier_evolution, 'glacier_evolution_lasso_' + str(year_start) + '_2100.csv'), sep=";")
    
else:
    
    ################################################
    #### Analyzing Deep learning - Lasso runs  ######
    #################################################
    
    ds_glacier_projections_lasso = xr.open_dataset(os.path.join(path_glacier_evolution, 'glacier_evolution_lasso_' + str(year_start) + '_2100.nc'))
    
    path_glacier_evolution = os.path.join(workspace, 'glacier_data', 'glacier_evolution')
    ds_glacier_projections = xr.open_dataset(os.path.join(path_glacier_evolution, 'glacier_evolution_' + str(year_start) + '_2100.nc'))
    
#    import pdb; pdb.set_trace()
    
    MB_26_lasso = ds_glacier_projections_lasso.MB.sel(RCP='26').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    MB_45_lasso = ds_glacier_projections_lasso.MB.sel(RCP='45').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    MB_85_lasso = ds_glacier_projections_lasso.MB.sel(RCP='85').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    
    MB_26_nn = ds_glacier_projections.MB.sel(RCP='26', member=ds_glacier_projections_lasso.member.values).mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    MB_45_nn = ds_glacier_projections.MB.sel(RCP='45', member=ds_glacier_projections_lasso.member.values).mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    MB_85_nn = ds_glacier_projections.MB.sel(RCP='85', member=ds_glacier_projections_lasso.member.values).mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    
#    import pdb; pdb.set_trace()
       
    p26 = poly.Polynomial.fit(MB_26_lasso.year.values[:-1], (MB_26_nn.values - MB_26_lasso.values)[:-1], 3)
    poly26 = np.asarray(p26.linspace(n=MB_45_lasso.year.values[:-1].size))[1,:]
    p45 = poly.Polynomial.fit(MB_45_lasso.year.values[:-1], (MB_45_nn.values - MB_45_lasso.values)[:-1], 3)
    poly45 = np.asarray(p45.linspace(n=MB_45_lasso.year.values[:-1].size))[1,:]
    p85 = poly.Polynomial.fit(MB_85_lasso.year.values[:-1], (MB_85_nn.values - MB_85_lasso.values)[:-1], 3)
    poly85 = np.asarray(p85.linspace(n=MB_85_lasso.year.values[:-1].size))[1,:]
    
    #########################################
    #### Analyzing ALPGM - GloGEM runs ######
    #########################################
    
    MB_26_alpgm = ds_glacier_projections.MB.sel(RCP='26').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    MB_45_alpgm = ds_glacier_projections.MB.sel(RCP='45').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    MB_85_alpgm = ds_glacier_projections.MB.sel(RCP='85').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    
    volume_26_alpgm = ds_glacier_projections.volume.sel(RCP='26').mean(dim=['massif_ID', 'member']).sum(dim=['GLIMS_ID'])[:-1]/1000
    volume_45_alpgm = ds_glacier_projections.volume.sel(RCP='45').mean(dim=['massif_ID', 'member']).sum(dim=['GLIMS_ID'])[:-1]/1000
    volume_85_alpgm = ds_glacier_projections.volume.sel(RCP='85').mean(dim=['massif_ID', 'member']).sum(dim=['GLIMS_ID'])[:-1]/1000
    
    zmean_26_alpgm = ds_glacier_projections.zmean.sel(RCP='26').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    zmean_45_alpgm = ds_glacier_projections.zmean.sel(RCP='45').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    zmean_85_alpgm = ds_glacier_projections.zmean.sel(RCP='85').mean(dim=['GLIMS_ID', 'massif_ID', 'member'])
    
    zeko_MB_26 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_massbalance_rcp26-selection.csv'), sep=',')
    zeko_MB_45 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_massbalance_rcp45-selection.csv'), sep=',')
    zeko_MB_85 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_massbalance_rcp85-selection.csv'), sep=',')
    
    zeko_volume_26 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_volume_rcp26-selection.csv'), sep=',')
    zeko_volume_45 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_volume_rcp45-selection.csv'), sep=',')
    zeko_volume_85 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_volume_rcp85-selection.csv'), sep=',')
    
    zeko_zmean_26 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_meanaltitude_rcp26-selection.csv'), sep=',')
    zeko_zmean_45 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_meanaltitude_rcp45-selection.csv'), sep=',')
    zeko_zmean_85 = pd.read_csv(os.path.join(path_glogem, 'ZekollariHussFarinotti_TC2019_meanaltitude_rcp85-selection.csv'), sep=',')
    
    zeko_avg_MB_26 = zeko_MB_26.mean(axis=0).values[1:-1]
    zeko_avg_MB_45 = zeko_MB_45.mean(axis=0).values[1:-1]
    zeko_avg_MB_85 = zeko_MB_85.mean(axis=0).values[1:-1]
    zeko_avg_volume_26 = zeko_volume_26.sum(axis=0).values[1:-1]
    zeko_avg_volume_45 = zeko_volume_45.sum(axis=0).values[1:-1]
    zeko_avg_volume_85 = zeko_volume_85.sum(axis=0).values[1:-1]
    zeko_avg_zmean_26 = zeko_zmean_26.mean(axis=0).values[1:-1]
    zeko_avg_zmean_45 = zeko_zmean_45.mean(axis=0).values[1:-1]
    zeko_avg_zmean_85 = zeko_zmean_85.mean(axis=0).values[1:-1]
    
#    import pdb; pdb.set_trace()
    
    p26_algpgm = poly.Polynomial.fit(MB_26_alpgm.year.values[2:], (MB_26_alpgm.values[2:] - zeko_avg_MB_26), 3)
    poly26_algpgm = np.concatenate(([np.nan,np.nan], np.asarray(p26_algpgm.linspace(n=MB_26_alpgm.year.values[2:].size))[1,:]))
    p45_algpgm = poly.Polynomial.fit(MB_45_alpgm.year.values[2:], (MB_45_alpgm.values[2:] - zeko_avg_MB_45), 3)
    poly45_algpgm = np.concatenate(([np.nan,np.nan], np.asarray(p45_algpgm.linspace(n=MB_45_alpgm.year.values[2:].size))[1,:]))
    p85_algpgm = poly.Polynomial.fit(MB_85_alpgm.year.values[2:], (MB_85_alpgm.values[2:] - zeko_avg_MB_85), 3)
    poly85_algpgm = np.concatenate(([np.nan,np.nan], np.asarray(p85_algpgm.linspace(n=MB_85_alpgm.year.values[2:].size))[1,:]))
    
    mb_cum_diff_26 = np.concatenate(([np.nan,np.nan], np.cumsum(MB_26_alpgm.values[2:] - zeko_avg_MB_26)))
    mb_cum_diff_45 = np.concatenate(([np.nan,np.nan], np.cumsum(MB_45_alpgm.values[2:] - zeko_avg_MB_45)))
    mb_cum_diff_85 = np.concatenate(([np.nan,np.nan], np.cumsum(MB_85_alpgm.values[2:] - zeko_avg_MB_85)))
    
    zeko_avg_MB_26 = np.concatenate(([np.nan,np.nan], zeko_avg_MB_26))
    zeko_avg_MB_45 = np.concatenate(([np.nan,np.nan], zeko_avg_MB_45))
    zeko_avg_MB_85 = np.concatenate(([np.nan,np.nan], zeko_avg_MB_85))
    zeko_avg_volume_26 = np.concatenate(([np.nan,np.nan], zeko_avg_volume_26))
    zeko_avg_volume_45 = np.concatenate(([np.nan,np.nan], zeko_avg_volume_45))
    zeko_avg_volume_85 = np.concatenate(([np.nan,np.nan], zeko_avg_volume_85))
    zeko_avg_zmean_26 = np.concatenate(([np.nan,np.nan], zeko_avg_zmean_26))
    zeko_avg_zmean_45 = np.concatenate(([np.nan,np.nan], zeko_avg_zmean_45))
    zeko_avg_zmean_85 = np.concatenate(([np.nan,np.nan], zeko_avg_zmean_85))

#    import pdb; pdb.set_trace()
    
    ######  PLOT   ############
    ###########################
    
    ##### LASSO VS DEEP LEARNING #######
    
    fig1, ax1 = plot.subplots(ncols=2, nrows=3, axwidth=2, aspect=1.5, sharey=1, hspace='2em')

    ax1.format(
            abc=True, abcloc='ul', abcstyle='A',
            ygridminor=True,
            ytickloc='both', yticklabelloc='left',
            xlabel='Year',
            rightlabels=['RCP 2.6', 'RCP 4.5', 'RCP 8.5'],
            collabels=['Annual nonlinear difference', 'Cumulative nonlinear difference']
    )
    
    # Non-cumulative
    ax1[0].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax1[2].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax1[4].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    
    h1 = ax1[0].plot(MB_26_lasso.year.values, MB_26_nn.values - MB_26_lasso.values, c='slate blue', linewidth=1)
    h1 = ax1[0].plot(MB_26_lasso.year.values[:-1], poly26, c='dark blue', linewidth=4, alpha=0.5)
    ax1[0].set_ylim([-1,0.75])
#    ax1[0].format(ylabel = "Lasso - Deep learning ($\Delta$ m.w.e. a$^{-1}$)")
    h2 = ax1[2].plot(MB_45_lasso.year.values, MB_45_nn.values - MB_45_lasso.values, c='sienna', linewidth=1)
    h2 = ax1[2].plot(MB_45_lasso.year.values[:-1], poly45, c='dark orange', linewidth=4, alpha=0.5)
    ax1[2].set_ylim([-1,0.75])
    ax1[2].format(ylabel = "Deep learning - Lasso ($\Delta$ m.w.e. a$^{-1}$)")
    h3 = ax1[4].plot(MB_85_lasso.year.values, MB_85_nn.values - MB_85_lasso.values, c='light maroon', linewidth=1)
    h3 = ax1[4].plot(MB_85_lasso.year.values[:-1], poly85, c='dark red', linewidth=4, alpha=0.5)
    ax1[4].set_ylim([-1,0.75])
#    ax1[4].format(ylabel = "Lasso - Deep learning ($\Delta$ m.w.e. a$^{-1}$)")
    
    # Cumulative
    ax1[1].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax1[3].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax1[5].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    
    h1 = ax1[1].plot(MB_26_lasso.year.values, np.cumsum(MB_26_nn.values - MB_26_lasso.values), c='slate blue', linewidth=2)
    ax1[1].set_ylim([-4,27])
#    ax1[1].format(ylabel = "Lasso - Deep learning  ($\Delta$ m.w.e.)")
    h2 = ax1[3].plot(MB_45_lasso.year.values, np.cumsum(MB_45_nn.values - MB_45_lasso.values), c='sienna', linewidth=2)
    ax1[3].set_ylim([-4,27])
    ax1[3].format(ylabel = "Deep learning - Lasso ($\Delta$ m.w.e.)")
    h3 = ax1[5].plot(MB_85_lasso.year.values, np.cumsum(MB_85_nn.values - MB_85_lasso.values), c='light maroon', linewidth=2)
    ax1[5].set_ylim([-4,27])
#    ax1[5].format(ylabel = "Lasso - Deep learning ($\Delta$ m.w.e.)")
    
    #######################################
     ##### ALPGM VS GLOGEM #######
    
    fig2, ax2 = plot.subplots(ncols=2, nrows=3, axwidth=2, aspect=1.5, sharey=1, hspace='2em')

    ax2.format(
            abc=True, abcloc='ll', abcstyle='A',
            ygridminor=True,
            ytickloc='both', yticklabelloc='left',
            xlabel='Year',
            rightlabels=['RCP 2.6', 'RCP 4.5', 'RCP 8.5'],
            collabels=['Annual difference', 'Cumulative difference']
    )
    
    # Non-cumulative
    ax2[0].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax2[2].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax2[4].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    
    h1 = ax2[0].plot(MB_26_alpgm.year.values, MB_26_alpgm.values - zeko_avg_MB_26, c='slate blue', linewidth=1)
    h1 = ax2[0].plot(MB_26_alpgm.year.values, poly26_algpgm, c='dark blue', linewidth=4, alpha=0.5)
    ax2[0].set_ylim([-2,1])
    h2 = ax2[2].plot(MB_45_alpgm.year.values, MB_45_alpgm.values - zeko_avg_MB_45, c='sienna', linewidth=1)
    h2 = ax2[2].plot(MB_45_alpgm.year.values, poly45_algpgm, c='dark orange', linewidth=4, alpha=0.5)
    ax2[2].set_ylim([-2,1])
    ax2[2].format(ylabel = "ALPGM - GloGEMflow ($\Delta$ m.w.e. a$^{-1}$)")
    h3 = ax2[4].plot(MB_85_alpgm.year.values, MB_85_alpgm.values - zeko_avg_MB_85, c='light maroon', linewidth=1)
    h3 = ax2[4].plot(MB_85_alpgm.year.values, poly85_algpgm, c='dark red', linewidth=4, alpha=0.5)
    ax2[4].set_ylim([-2,1])
    
    # Cumulative
    ax2[1].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax2[3].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax2[5].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    
    h1 = ax2[1].plot(MB_26_alpgm.year.values, mb_cum_diff_26, c='slate blue', linewidth=2)
    ax2[1].set_ylim([-35,5])
#    ax1[1].format(ylabel = "Lasso - Deep learning  ($\Delta$ m.w.e.)")
    h2 = ax2[3].plot(MB_45_alpgm.year.values, mb_cum_diff_45, c='sienna', linewidth=2)
    ax2[3].set_ylim([-35,5])
    ax2[3].format(ylabel = "ALPGM - GloGEMflow ($\Delta$ m.w.e. a$^{-1}$)")
    h3 = ax2[5].plot(MB_85_alpgm.year.values, mb_cum_diff_85, c='light maroon', linewidth=2)
    ax2[5].set_ylim([-35,5])
#    ax1[5].format(ylabel = "Lasso - Deep learning ($\Delta$ m.w.e.)")
    
    ######################
    ####  ZMEAN  #########
    
    fig3, ax3 = plot.subplots(ncols=1, nrows=3, axwidth=2, aspect=1.5, sharey=1, hspace='2em')

    ax3.format(
            abc=True, abcloc='ul', abcstyle='A',
            ygridminor=True,
            ytickloc='both', yticklabelloc='left',
            xlabel='Year',
            ylabel='ALPGM - GloGEMflow glacier mean altitude ($\Delta$m)',
            rightlabels=['RCP 2.6', 'RCP 4.5', 'RCP 8.5'],
    )
    
    ax3[0].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax3[1].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax3[2].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    h1 = ax3[0].plot(zmean_26_alpgm.year.values, zmean_26_alpgm.values - zeko_avg_zmean_26, c='slate blue', linewidth=2)
    h2 = ax3[1].plot(zmean_45_alpgm.year.values, zmean_45_alpgm.values - zeko_avg_zmean_45, c='sienna', linewidth=2)
    h3 = ax3[2].plot(zmean_85_alpgm.year.values, zmean_85_alpgm.values - zeko_avg_zmean_85, c='light maroon', linewidth=2)
    
    ######################
    ####  VOLUME  #########
    
    fig4, ax4 = plot.subplots(ncols=1, nrows=3, axwidth=2, aspect=1.5, share=1, hspace='2em')

    ax3.format(
            abc=True, abcloc='ul', abcstyle='A',
            ygridminor=True,
            ytickloc='both', yticklabelloc='left',
            xlabel='Year',
            rightlabels=['RCP 2.6', 'RCP 4.5', 'RCP 8.5'],
    )
    
    h11 = ax4[0].plot(zmean_26_alpgm.year.values[:-1], volume_26_alpgm.values, c='midnightblue', linewidth=2, label='ALPGM', legend='r', legend_kw={'ncols':1,'frame':True})
    h12 = ax4[0].plot(zmean_26_alpgm.year.values, zeko_avg_volume_26, c='skyblue', linewidth=2, label='GloGEMflow', legend='r', legend_kw={'ncols':1,'frame':True})
    ax4[0].set_ylim([0,volume_26_alpgm.max()])
    ax4[0].format(ylabel='Total glacier volume (km$^{3}$)')
    
    h21 = ax4[1].plot(zmean_45_alpgm.year.values[:-1], volume_45_alpgm.values, c='bronze', linewidth=2, label='ALPGM', legend='r', legend_kw={'ncols':1,'frame':True})
    h22 = ax4[1].plot(zmean_45_alpgm.year.values, zeko_avg_volume_45, c='goldenrod', linewidth=2, label='GloGEMflow', legend='r', legend_kw={'ncols':1,'frame':True})
    ax4[1].set_ylim([0,volume_45_alpgm.max()])
    ax4[1].format(ylabel='Total glacier volume (km$^{3}$)')
    
    h31 = ax4[2].plot(zmean_85_alpgm.year.values[:-1], volume_85_alpgm.values, c='darkred', linewidth=2, label='ALPGM', legend='r', legend_kw={'ncols':1,'frame':True})
    h32 = ax4[2].plot(zmean_85_alpgm.year.values, zeko_avg_volume_85, c='tomato', linewidth=2, label='GloGEMflow', legend='r', legend_kw={'ncols':1,'frame':True})
    ax4[2].set_ylim([0,volume_85_alpgm.max()])
    ax4[2].format(ylabel='Total glacier volume (km$^{3}$)')
    
#    ax4[0].legend((h11,h12), loc='r', ncols=1, frame=True)
#    ax4[1].legend((h21,h22), loc='r', ncols=1, frame=True)
#    ax4[2].legend((h31,h32), loc='r', ncols=1, frame=True)
    
    plt.show()
    
if(other_plots):
        
    #    import pdb; pdb.set_trace()
        
    ##########    PLOTS    #######################
    if(filter_glacier):
        header = glacier_ID_filter + "_"
    else:
        header = "lasso_"
    
    ############  Area and volume   #####################################################################################
    
    #############       Plot each one of the RCP-GCM-RCM combinations       #############################################
        
    fig1, ax1 = plot.subplots([[1, 1], [2, 3]], ncols=2, nrows=2, axwidth=4, aspect=2.5, share=0)
    #if(filter_glacier):
    #    fig1.suptitle("Glacier " + glacier_name_filter + " glacier projections under climate change")
    #else:
    #    fig1.suptitle("Regional average French alpine glacier projections under climate change")
    
    ax1.format(
            abc=True, abcloc='ur', abcstyle='A',
            ygridminor=True,
            ytickloc='both', yticklabelloc='left'
    )
    
    # Glacier-wide MB
    ax1[0].set_ylabel('Glacier-wide MB (m.w.e. a$^{-1}$)')
    ax1[0].set_xlabel('Year')
    
    plot_individual_members(ax1[0], RCP_member_means, RCP_means, 'MB', filtered_member)
    h = plot_RCP_means(ax1[0], RCP_means, 'MB', with_26)
    ax1[0].axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    
    # Volume
    ax1[1].set_ylabel('Volume (m$^3$ 10$^6$)')
    ax1[1].set_xlabel('Year')
    ax1[1].set_ylim(0, np.max(RCP_means['45']['volume']['data']))
    ax1b = ax1[1].twinx()  # instantiate a second axes that shares the same x-axis
    ax1b.set_ylim(0, 100)
    ax1b.set_ylabel('Remaining fraction (%)')
    
    plot_individual_members(ax1[1], RCP_member_means, RCP_means, 'volume', filtered_member)
    h = plot_RCP_means(ax1[1], RCP_means, 'volume', with_26)
    
    ax1[2].set_ylabel('Area (km$^{2}$)')
    ax1[2].set_xlabel('Year')
    ax1[2].set_ylim(0, np.max(RCP_means['45']['area']['data']))
    ax1c = ax1[2].twinx()  # instantiate a second axes that shares the same x-axis
    ax1c.set_ylim(0, 100)
    ax1c.set_ylabel('Remaining fraction (%)')
    
    # Area
    plot_individual_members(ax1[2], RCP_member_means, RCP_means, 'area', filtered_member)
    h = plot_RCP_means(ax1[2], RCP_means, 'area', with_26)
    
    fig1.legend(h, loc='r', ncols=1, frame=True)
    
    # Save as PDF
    save_plot_as_pdf(fig1, header + 'mb_volume_area', with_26)
    
    # Store RCP means in CSV file
    store_RCP_mean(path_glacier_area, 'area', RCP_means)
    store_RCP_mean(path_glacier_volume, 'volume', RCP_means)
    store_RCP_mean(path_MB_simulations, 'MB', RCP_means)
    
    ###############     Plot the evolution of topographical parameters    ####################################
    ###############     Zmean and slope      #################################################################
    
    fig2, (ax21, ax22, ax23) = plot.subplots([[1, 1], [2, 3]], ncols=2, nrows=2, axwidth=4, aspect=2.5, share=0)
    
    for ax2 in (ax21, ax22, ax23):
        ax2.format(
                abc=True, abcloc='ul', abcstyle='A',
                ygridminor=True,
                ytickloc='both', yticklabelloc='left'
        )
    
    #if(filter_glacier):
    #    fig2.suptitle("Glacier " + glacier_name_filter + " glacier projections under climate change")
    #else:
    #    fig2.suptitle("Regional average French alpine glacier projections under climate change")
        
    ax21.set_ylabel('Mean glacier altitude (m)')
    ax21.set_xlabel('Year')
    
    ax22.set_ylabel('Mean glacier tongue slope (°)')
    ax22.set_xlabel('Year')
    
    ax23.set_ylabel('Mean glacier area (km$^{2}$)')
    ax23.set_xlabel('Year')
    
    # Mean altitude
    plot_individual_members(ax21, RCP_member_means, RCP_means, 'zmean', filtered_member)
    h = plot_RCP_means(ax21, RCP_means, 'zmean', with_26)
    
    # Slope 20% altitudinal range
    plot_individual_members(ax22, RCP_member_means, RCP_means, 'slope20', filtered_member)
    h = plot_RCP_means(ax22, RCP_means, 'slope20', with_26)
    
    # Mean glacier area
    plot_individual_members(ax23, RCP_member_means, RCP_means, 'avg_area', filtered_member)
    h = plot_RCP_means(ax23, RCP_means, 'avg_area', with_26)
    
    fig2.legend(h, loc='r', ncols=1, frame=True)
    
    # Save as PDF
    save_plot_as_pdf(fig2, header + 'zmean_slope_avgArea', with_26)
    
    # Store RCP means in CSV file
    store_RCP_mean(path_glacier_zmean, 'zmean', RCP_means)
    store_RCP_mean(path_glacier_slope20, 'slope20', RCP_means)
    
    
    ###############     Plot the evolution of temperature and snowfall    ####################################
    ###############     Winter and summer snowfall, rain and temperature  ####################################
    
    fig3, axs3 = plot.subplots(ncols=3, nrows=3, aspect=2, axwidth=2, spany=0)
    
    #if(filter_glacier):
    #    fig3.suptitle("Glacier " + glacier_name_filter + " climate projections")
    #else:
    #    fig3.suptitle("French Alpine glaciers average regional climate signal evolution")
    
    axs3.format(
            abc=True, abcloc='ul', abcstyle='A',
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
        
    alpha = 0.15
    
    # CPDD
    plot_individual_members(axs3[0], RCP_member_means, RCP_means, 'CPDD', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[0], RCP_means, 'CPDD', with_26, legend=False, linewidth=2)
    axs3[0].set_ylim([0,1500])
    
    # Summer CPDD
    plot_individual_members(axs3[1], RCP_member_means, RCP_means, 'summer_CPDD', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[1], RCP_means, 'summer_CPDD', with_26, legend=False, linewidth=2)
    
    # Winter CPDD
    plot_individual_members(axs3[2], RCP_member_means, RCP_means, 'winter_CPDD', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[2], RCP_means, 'winter_CPDD', with_26, legend=False, linewidth=2)
    
    # Snowfall
    plot_individual_members(axs3[3], RCP_member_means, RCP_means, 'snowfall', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[3], RCP_means, 'snowfall', with_26, legend=False, linewidth=2)
    
    # Summer Snowfall
    plot_individual_members(axs3[4], RCP_member_means, RCP_means, 'summer_snowfall', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[4], RCP_means, 'summer_snowfall', with_26, legend=False, linewidth=2)
    
    # Winter Snowfall
    plot_individual_members(axs3[5], RCP_member_means, RCP_means, 'winter_snowfall', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[5], RCP_means, 'winter_snowfall', with_26, legend=False, linewidth=2)
    
    # Rainfall
    plot_individual_members(axs3[6], RCP_member_means, RCP_means, 'rain', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[6], RCP_means, 'rain', with_26, legend=False, linewidth=2)
    axs3[6].set_ylim([0,1000])
    
    # Summer rainfall
    plot_individual_members(axs3[7], RCP_member_means, RCP_means, 'summer_rain', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[7], RCP_means, 'summer_rain', with_26, legend=False, linewidth=2)
    
    # Winter rainfall
    plot_individual_members(axs3[8], RCP_member_means, RCP_means, 'winter_rain', filtered_member, alpha=alpha)
    h = plot_RCP_means(axs3[8], RCP_means, 'winter_rain', with_26, legend=False, linewidth=2)
    
    fig3.legend(h, loc='r', ncols=1, frame=True)
    
    # Save as PDF
    save_plot_as_pdf(fig3, header + 'CPDD_snowfall_rain', with_26)
    
    # Store RCP means in CSV file
    store_RCP_mean(path_glacier_CPDDs, 'CPDD', RCP_means)
    store_RCP_mean(path_glacier_snowfall, 'snowfall', RCP_means)
    
    
    ###############     Plot the glacier meltwater discharge   ####################################
    fig5, (ax51) = plot.subplots(ncols=1, nrows=1, axwidth=5, aspect=2)
    ax51.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    
    #if(filter_glacier):
    #    fig5.suptitle("Glacier " + glacier_name_filter + " meltwater discharge evolution under climate change")
    #else:
    #    fig5.suptitle("Average meltwater discharge projections of French alpine glacier under climate change")
        
    ax51.set_ylabel('Meltwater discharge (m$^3$ 10$^{6}$)')
    ax51.set_xlabel('Year')
    
    # Glacier meltwater discharge
    plot_individual_members(ax51, RCP_member_means, RCP_means, 'discharge', filtered_member)
    h = plot_RCP_means(ax51, RCP_means, 'discharge', with_26)
    
    # Save as PDF
    save_plot_as_pdf(fig5, header + 'meltwater_discharge', with_26)
    
    # Store RCP means in CSV file
    store_RCP_mean(path_glacier_discharge, 'discharge', RCP_means)
    
    
    
    #######################################################################################################
    #############   Full glacier evolution vs static glacier geometry projections  #########################
    
    ###############     Plot the evolution of temperature and snowfall    ####################################
    ###############     Winter and summer snowfall, rain and temperature  ####################################
    
    if(static_geometry):
    
        fig6, axs6 = plot.subplots(ncols=3, nrows=3, aspect=2, axwidth=2, spany=0)
    #    if(filter_glacier):
    #        fig6.suptitle("Glacier " + glacier_name_filter + " climate projections")
    #    else:
    #        fig6.suptitle("Glacier retreat topographical feedback on climate projections")
        
        axs6.format(
                abc=True, abcloc='ul', abcstyle='A',
                ygridminor=True,
                ytickloc='both', yticklabelloc='left',
                xlabel='Year'
        )
        
        titles = ['Annual CPDD', 'Summer CPDD', 'Winter CPDD', 'Annual snowfall', 'Summer snowfall', 'Winter snowfall', 'Annual rainfall', 'Summer rainfall', 'Winter rainfall']
        ylabels = ['ΔPDD', 'ΔPDD', 'ΔPDD', 'Δmm', 'Δmm', 'Δmm', 'Δmm', 'Δmm', 'Δmm']
        for i, s_title, s_ylabel in zip(range(0, 9), titles, ylabels):
            if(s_title[0:6] == 'Winter'):
                season_color = 'midnightblue'
            elif(s_title[0:6] == 'Summer'):
               season_color = 'darkred'
            else:
                season_color = 'k'
            
            axs6[i].set_title(s_title, color=season_color)  
            axs6[i].format(ylabel=s_ylabel)
        
        # CPDD
        plot_individual_members_diff(axs6[0], RCP_member_means, RCP_means, 'CPDD', 'static_CPDD', filtered_member)
        h = plot_RCP_means_diff(axs6[0], RCP_means, 'CPDD', 'static_CPDD', with_26, legend=False, linewidth=2)
        
        # Summer CPDD
        plot_individual_members_diff(axs6[1], RCP_member_means, RCP_means, 'summer_CPDD', 'static_summer_CPDD', filtered_member)
        h = plot_RCP_means_diff(axs6[1], RCP_means, 'summer_CPDD', 'static_summer_CPDD', with_26, legend=False, linewidth=2)
        
        # Winter CPDD
        plot_individual_members_diff(axs6[2], RCP_member_means, RCP_means, 'winter_CPDD', 'static_winter_CPDD', filtered_member)
        h = plot_RCP_means_diff(axs6[2], RCP_means, 'winter_CPDD', 'static_winter_CPDD', with_26, legend=False, linewidth=2)
        
        # Snowfall
        plot_individual_members_diff(axs6[3], RCP_member_means, RCP_means, 'snowfall', 'static_snowfall', filtered_member)
        h = plot_RCP_means_diff(axs6[3], RCP_means, 'snowfall', 'static_snowfall', with_26, legend=False, linewidth=2)
        axs6[3].set_ylim([0,400])
        
        # Summer Snowfall
        plot_individual_members_diff(axs6[4], RCP_member_means, RCP_means, 'summer_snowfall', 'static_summer_snowfall', filtered_member)
        h = plot_RCP_means_diff(axs6[4], RCP_means, 'summer_snowfall', 'static_summer_snowfall', with_26, legend=False, linewidth=2)
        
        # Winter Snowfall
        plot_individual_members_diff(axs6[5], RCP_member_means, RCP_means, 'winter_snowfall', 'static_winter_snowfall', filtered_member)
        h = plot_RCP_means_diff(axs6[5], RCP_means, 'winter_snowfall', 'static_winter_snowfall', with_26, legend=False, linewidth=2)
        
        # Rainfall
        plot_individual_members_diff(axs6[6], RCP_member_means, RCP_means, 'rain', 'static_rain', filtered_member)
        h = plot_RCP_means_diff(axs6[6], RCP_means, 'rain', 'static_rain', with_26, legend=False, linewidth=2)
        axs6[6].set_ylim([-250,0])
        
        # Summer rainfall
        plot_individual_members_diff(axs6[7], RCP_member_means, RCP_means, 'summer_rain', 'static_summer_rain', filtered_member)
        h = plot_RCP_means_diff(axs6[7], RCP_means, 'summer_rain', 'static_summer_rain', with_26, legend=False, linewidth=2)
        
        # Winter rainfall
        plot_individual_members_diff(axs6[8], RCP_member_means, RCP_means, 'winter_rain', 'static_winter_rain', filtered_member)
        h = plot_RCP_means_diff(axs6[8], RCP_means, 'winter_rain', 'static_winter_rain', with_26, legend=False, linewidth=2)
        
        fig6.legend(h, loc='r', ncols=1, frame=True)
        
        # Save as PDF
        save_plot_as_pdf(fig6, header + 'static_vs_dynamical_CPDD_snowfall_rain', with_26)
        
        
        ###############     Plot the glacier-wide MB   ####################################
        fig7, (ax7) = plot.subplots(ncols=1, nrows=1, axwidth=5, aspect=4)
        ax7.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    #    if(filter_glacier):
    #        fig7.suptitle("Glacier " + glacier_name_filter + " glacier-wide MB evolution under climate change")
    #    else:
    #        fig7.suptitle("Glacier retreat topographical feedback on glacier-wide MB projections")
        ax7.set_ylabel('Δ Glacier-wide MB (m.w.e. a$^{-1}$)')
        ax7.set_xlabel('Year')
        
        # Glacier-wide MB
        plot_individual_members_diff(ax7, RCP_member_means, RCP_means, 'MB', 'static_MB', filtered_member, alpha=0.3)
        h = plot_RCP_means_diff(ax7, RCP_means, 'MB', 'static_MB', with_26)
        
        # Save as PDF
        save_plot_as_pdf(fig7, header + 'static_MB', with_26)
        
        #plt.show()