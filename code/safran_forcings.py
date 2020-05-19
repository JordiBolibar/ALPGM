# -*- coding: utf-8 -*-

"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

SAFRAN TEMPERATURE (CPDD) AND PRECIPITATION (SNOW) COMPUTATION

"""

## Dependencies: ##
import numpy as np
from numpy import genfromtxt
from numba import jit
import copy
#import matplotlib.pyplot as plt
import os
from difflib import SequenceMatcher
import math
import xarray as xr
#import dask as da
from pathlib import Path
import time

import settings

### FUNCTIONS  ####

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def similar(a, b):
    ratios = []
    for glacier_name in a:
        ratios.append(SequenceMatcher(None, glacier_name, b).ratio())
    ratios = np.asarray(ratios)
    return ratios.max(), ratios.argmax()

@jit
def find_glacier_idx(glacier_massif, massif_number, altitudes, glacier_altitude, aspects, glacier_aspect):
    massif_altitudes_idx = np.where(massif_number == float(glacier_massif))[0]
    glacier_aspect_idx = np.where(aspects == float(glacier_aspect))[0]
    massif_alt_aspect_idx = np.array(list(set(massif_altitudes_idx).intersection(glacier_aspect_idx)))
    index_alt = find_nearest(altitudes[massif_alt_aspect_idx], glacier_altitude)
    final_idx = int(massif_alt_aspect_idx[index_alt])
    
    return final_idx

@jit
def get_SAFRAN_glacier_coordinates(massif_number, zs, aspects, glims_data, glims_rabatel):
    glacier_centroid_altitude = glims_data['MEAN_Pixel']
    GLIMS_IDs = glims_data['GLIMS_ID']
    RGI_IDs = glims_data['ID']
    glacier_massifs = glims_data['Massif_SAFRAN']
    glacier_names = glims_data['Glacier']
    glacier_aspects = glims_data['Aspect_num']
    glacier_SMB_coordinates, all_glacier_coordinates = [], []
    
    # Glaciers with SMB Data (Rabatel et al. 2016)
    for glims_id, glacier_name, glacier_massif, glacier_altitude, glacier_aspect in zip(glims_rabatel['GLIMS_ID'], glims_rabatel['Glacier'], glims_rabatel['Massif_SAFRAN'], glims_rabatel['MEAN_Pixel'], glims_rabatel['Aspect_num']):
        glacier_SMB_coordinates.append([glacier_name, find_glacier_idx(glacier_massif, massif_number, zs, glacier_altitude, aspects, glacier_aspect), float(glacier_altitude), glims_id, int(glacier_massif)])
    
    # All glaciers loop
    for glims_id, glacier_name, glacier_massif, glacier_altitude, glacier_aspect in zip(GLIMS_IDs, glacier_names, glacier_massifs, glacier_centroid_altitude, glacier_aspects):
        all_glacier_coordinates.append([glacier_name, find_glacier_idx(glacier_massif, massif_number, zs, glacier_altitude, aspects, glacier_aspect), float(glacier_altitude), glims_id, int(glacier_massif), RGI_IDs])
        
    return np.asarray(glacier_SMB_coordinates), np.asarray(all_glacier_coordinates)

# Computes seasonal meteo anomalies at glacier scale
def compute_local_anomalies(idx, glacier_CPDDs, glacier_winter_snow, glacier_summer_snow, 
                      local_anomalies, raw_local_anomalies):
    
    # The anomalies are always computed with respect to the 1984-2014 mean
    glacier_CPDDs[idx]['CPDD'] = np.asarray(glacier_CPDDs[idx]['CPDD'])
    glacier_CPDDs_training = glacier_CPDDs[idx]['CPDD'][-49:]
    glacier_CPDDs[idx]['Mean'] = glacier_CPDDs_training.mean()
    glacier_winter_snow[idx]['snow'] = np.asarray(glacier_winter_snow[idx]['snow'])
    glacier_winter_snow_training = glacier_winter_snow[idx]['snow'][-49:]
    glacier_winter_snow[idx]['Mean'] = glacier_winter_snow_training.mean()
    glacier_summer_snow[idx]['snow'] = np.asarray(glacier_summer_snow[idx]['snow'])
    glacier_summer_snow_training = glacier_summer_snow[idx]['snow'][-49:]
    glacier_summer_snow[idx]['Mean'] = glacier_summer_snow_training.mean()
    
    local_anomalies['CPDD'].append(copy.deepcopy(glacier_CPDDs[idx]))
    local_anomalies['years'].append(copy.deepcopy(glacier_CPDDs[idx]['years']))
    local_anomalies['CPDD'][-1]['CPDD'] = glacier_CPDDs[idx]['CPDD'] - glacier_CPDDs[idx]['Mean']
    raw_local_anomalies['CPDD'].append(local_anomalies['CPDD'][-1]['CPDD'])
    
    local_anomalies['w_snow'].append(copy.deepcopy(glacier_winter_snow[idx]))
    local_anomalies['w_snow'][-1]['w_snow'] = glacier_winter_snow[idx]['snow'] - glacier_winter_snow[idx]['Mean']
    raw_local_anomalies['w_snow'].append(local_anomalies['w_snow'][-1]['w_snow']) 
    
    local_anomalies['s_snow'].append(copy.deepcopy(glacier_summer_snow[idx]))
    local_anomalies['s_snow'][-1]['s_snow'] = glacier_summer_snow[idx]['snow'] - glacier_summer_snow[idx]['Mean']
    raw_local_anomalies['s_snow'].append(local_anomalies['s_snow'][-1]['s_snow'])
    
    return glacier_CPDDs, glacier_winter_snow, glacier_summer_snow, local_anomalies, raw_local_anomalies


# Computes monthly meteo anomalies at glacier scale
def compute_monthly_anomalies(idx, glacier_mon_temp, glacier_mon_snow,
                              local_mon_anomalies, raw_local_mon_anomalies):
    
    # The monthly meteo anomalies, as well as the seasonal ones, are always computed with respect to the 1984-2014 period
    mon_range = range(0, 12)
    
    for mon_idx in mon_range:
        mon_avg_temp, mon_avg_snow = [],[]
        for glacier_temp, glacier_snow in zip(glacier_mon_temp[idx]['mon_temp'], glacier_mon_snow[idx]['mon_snow']):
            mon_avg_temp.append(glacier_temp[mon_idx])
            mon_avg_snow.append(glacier_snow[mon_idx])
        mon_avg_temp = np.asarray(mon_avg_temp)
        mon_avg_snow = np.asarray(mon_avg_snow)
        
        glacier_mon_temp[idx]['mon_means'].append(mon_avg_temp[-49:].mean())
        glacier_mon_snow[idx]['mon_means'].append(mon_avg_snow[-49:].mean())
        
    local_mon_anomalies['temp'] = np.append(local_mon_anomalies['temp'], copy.deepcopy(glacier_mon_temp[idx]))
    local_mon_anomalies['snow'] = np.append(local_mon_anomalies['snow'], copy.deepcopy(glacier_mon_snow[idx]))
    
    for mon_idx in mon_range:
        year_idx = 0
        for glacier_temp, glacier_snow in zip(local_mon_anomalies['temp'][-1]['mon_temp'], local_mon_anomalies['snow'][-1]['mon_snow']):
             local_mon_anomalies['temp'][-1]['mon_temp'][year_idx][mon_idx] = local_mon_anomalies['temp'][-1]['mon_temp'][year_idx][mon_idx] - local_mon_anomalies['temp'][-1]['mon_means'][mon_idx]  
             local_mon_anomalies['snow'][-1]['mon_snow'][year_idx][mon_idx] = local_mon_anomalies['snow'][-1]['mon_snow'][year_idx][mon_idx] - local_mon_anomalies['snow'][-1]['mon_means'][mon_idx]
             year_idx = year_idx+1
    
    return glacier_mon_temp, glacier_mon_snow, local_mon_anomalies, raw_local_mon_anomalies


@jit
def get_SMB_glaciers(glacier_SMB_coordinates):
    smb_glaciers = []
    for glacier in glacier_SMB_coordinates:
        smb_glaciers.append(glacier[0])
    return np.asarray(smb_glaciers)

# Sorts the temperature and snow data in the same order as the GLIMS Rabatel 2003 file
#@jit
def sort_SMB_rabatel_data(i_season_meteo_SMB, local_anomalies_SMB, raw_local_anomalies_SMB, i_monthly_meteo_SMB, local_mon_anomalies_SMB, glims_rabatel):
    glacier_order = []
    
    # Seasonal
    season_meteo_SMB = {'CPDD':[], 'winter_snow':[], 'summer_snow':[]}
    season_meteo_anomalies_SMB = {'CPDD':[], 'winter_snow':[], 'summer_snow':[]}
    season_raw_meteo_anomalies_SMB = {'CPDD':[], 'winter_snow':[], 'summer_snow':[]}
    
    # Monthly
    monthly_meteo_SMB = {'temp':[], 'snow':[]}
    monthly_meteo_anomalies_SMB = {'temp':[], 'snow':[]}
    
    # We first capture the glacier order
    for glims_id in glims_rabatel['GLIMS_ID']:
        glacier_idx = 0
        for cpdd_smb_glacier in i_season_meteo_SMB['CPDD']:
            if(glims_id == cpdd_smb_glacier['GLIMS_ID']):
                
                # Seasonal
                season_meteo_SMB['CPDD'].append(i_season_meteo_SMB['CPDD'][glacier_idx])
                season_meteo_SMB['winter_snow'].append(i_season_meteo_SMB['winter_snow'][glacier_idx])  
                season_meteo_SMB['summer_snow'].append(i_season_meteo_SMB['summer_snow'][glacier_idx]) 
                
                season_meteo_anomalies_SMB['CPDD'].append(local_anomalies_SMB['CPDD'][glacier_idx])
                season_meteo_anomalies_SMB['winter_snow'].append(local_anomalies_SMB['w_snow'][glacier_idx])
                season_meteo_anomalies_SMB['summer_snow'].append(local_anomalies_SMB['s_snow'][glacier_idx])
                
                season_raw_meteo_anomalies_SMB['CPDD'].append(raw_local_anomalies_SMB['CPDD'][glacier_idx])
                season_raw_meteo_anomalies_SMB['winter_snow'].append(raw_local_anomalies_SMB['w_snow'][glacier_idx])
                season_raw_meteo_anomalies_SMB['summer_snow'].append(raw_local_anomalies_SMB['s_snow'][glacier_idx])
                
                # Monthly
                monthly_meteo_SMB['temp'].append(i_monthly_meteo_SMB['temp'][glacier_idx])
                monthly_meteo_SMB['snow'].append(i_monthly_meteo_SMB['snow'][glacier_idx])
                
                monthly_meteo_anomalies_SMB['temp'].append(local_mon_anomalies_SMB['temp'][glacier_idx])
                monthly_meteo_anomalies_SMB['snow'].append(local_mon_anomalies_SMB['snow'][glacier_idx])
                
                glacier_order.append(glacier_idx)
            glacier_idx = glacier_idx+1
            
    # And then we apply it to all the datasets    
    glacier_order = np.asarray(glacier_order, dtype=np.int32)
    
    return season_meteo_SMB, season_meteo_anomalies_SMB, season_raw_meteo_anomalies_SMB, monthly_meteo_SMB, monthly_meteo_anomalies_SMB

# Interpolates topo data to build the training matrix
@jit
def interpolate_extended_glims_variable(variable_name, glims_glacier, glims_2003, glims_1985, glims_1967):
    # In case there are multiple results, we choose the one with the most similar area
    idx_2003 = np.where(glims_2003['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
    if(idx_2003.size > 1):
        idx_aux = find_nearest(glims_2003[idx_2003]['Area'], glims_glacier['Area'])
        idx_2003 = idx_2003[idx_aux]
    idx_1985 = np.where(glims_1985['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
    if(idx_1985.size > 1):
        idx_aux = find_nearest(glims_1985[idx_2003]['Area'], glims_glacier['Area'])
        idx_1985 = idx_1985[idx_aux]
    
    idx_1959 = np.where(glims_1967['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
    if(idx_1959.size > 1):
        idx_aux = find_nearest(glims_1967[idx_2003]['Area'], glims_glacier['Area'])
        idx_1959 = idx_1959[idx_aux]
    
    var_1959 = glims_1967[idx_1959][variable_name]
    var_2015 = glims_glacier[variable_name]
    var_2003 = glims_2003[idx_2003][variable_name]
    var_1985 = glims_1985[idx_1985][variable_name]
    
    if(not math.isnan(var_2015)):
        interp_1959_1984 = np.linspace(var_1959, var_1985, num=(1984-1959))
        interp_1984_2003 = np.linspace(var_1985, var_2003, num=(2003-1984))
        interp_2003_2015 = np.linspace(var_2003, var_2015, num=(2015-2003)+1)
        interp_1959_2003 = np.append(interp_1959_1984, interp_1984_2003)
        interp_1959_2015 = np.append(interp_1959_2003, interp_2003_2015)
    else:
        interp_1959_2015 = [var_1985]
        
    return interp_1959_2015

# Interpolates and generates the glacier mean altitudes from the topo inventories
@jit
def generate_glacier_altitudes(glacier, glims_1967, glims_1985, glims_2003, glims_2015, glacier_altitudes):
    glacier_name = glacier[0]
    glacier_alt = float(glacier[2])
    glims_ID = glacier[3]
    
    glims_idx = np.where(glims_2015['GLIMS_ID'] == glims_ID)[0]
    
    # Procedure to verify glacier name and ID
    apply_glims_idx = False
    if(glims_idx.size == 0):
        similar_info = similar(glims_2015['Glacier'], glacier_name)
        if(similar_info[0] > 0.8):
            glims_idx = similar_info[1]
            apply_glims_idx = True
    elif(glims_idx.size > 1):
        is1, oneidx, is2, twoidx = False, 0, False, 0
        for idx in glims_idx:
            if(glims_2015[idx]['Glacier'] == glacier_name):
                glims_idx = idx
                apply_glims_idx = True
            elif(glims_2015[idx]['Glacier'].find('1') != -1):
                is1 = True
                oneidx = idx
            elif(glims_2015[idx]['Glacier'].find('2') != -1):
                is2 = True
                twoidx = idx
        if((glacier_name.find('1') != -1) & is1):
            glims_idx = oneidx
            apply_glims_idx = True
        elif((glacier_name.find('2') != -1) & is2):
            glims_idx = twoidx
            apply_glims_idx = True
        else:
            similar_info = similar(glims_2015[glims_idx]['Glacier'], glacier_name)
            if(similar_info[0] > 0.8):
                glims_idx = glims_idx[similar_info[1]]
                apply_glims_idx = True
    else:
        apply_glims_idx = True
    
    if(apply_glims_idx):
        glims_glacier = glims_2015[glims_idx]
        
        glacier_alt = interpolate_extended_glims_variable('MEAN_Pixel', glims_glacier, glims_2003, glims_1985, glims_1967)
    else:
        glacier_alt = [glacier_alt]
        
    glacier_altitudes.append(glacier_alt)
    
    return glacier_altitudes


def main(compute):

    print("\n-----------------------------------------------")
    print("              SAFRAN FORCING ")
    print("-----------------------------------------------\n")
    
    if(compute):
        ######   FILE PATHS    #######
        # Folders     
#        workspace = 'C:\\Jordi\\PhD\\Python\\'
        workspace = Path(os.getcwd()).parent
        path_safran_forcings = settings.path_safran
        path_smb = os.path.join(workspace, 'glacier_data', 'smb')
        path_smb_function_safran = os.path.join(path_smb, 'smb_function', 'SAFRAN')
        path_glims = os.path.join(workspace, 'glacier_data', 'GLIMS') 
        
        #### Flags  #####
        bypass_glacier_data = False
        t_lim = 0.0
        
#        year_start = 1984
#        year_start = 1967
        year_start = 1959
#        year_start = 2014
        year_end = 2015
#        year_end = 2014
        
        path_temps = os.path.join(path_smb_function_safran, 'daily_temps_years_' + str(year_start) + '-' + str(year_end) + '.txt')
        path_snow = os.path.join(path_smb_function_safran, 'daily_snow_years_' + str(year_start) + '-' + str(year_end) + '.txt')
        path_rain = os.path.join(path_smb_function_safran, 'daily_rain_years_' + str(year_start) + '-' + str(year_end) + '.txt')
        path_dates = os.path.join(path_smb_function_safran, 'daily_dates_years_' + str(year_start) + '-' + str(year_end) + '.txt')
        path_zs = os.path.join(path_smb_function_safran, 'zs_years' + str(year_start) + '-' + str(year_end) + '.txt')
        
        #### GLIMS data for 1985, 2003 and 2015
        glims_2015 = genfromtxt(os.path.join(path_glims, 'GLIMS_2015.csv'), delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), 
                    ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), 
                    ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), 
                    ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50')])
        glims_2003 = genfromtxt(os.path.join(path_glims, 'GLIMS_2003.csv'), delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), 
                     ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), 
                     ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), 
                     ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8'), ('ID', '<f8')])
        glims_1985 = genfromtxt(os.path.join(path_glims, 'GLIMS_1985.csv'), delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), 
                     ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), 
                     ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), 
                     ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50')])
        glims_1967 = genfromtxt(os.path.join(path_glims, 'GLIMS_1967.csv'), delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), 
                     ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), 
                     ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), 
                     ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50')])
        
        
        ####  GLIMS data for the 30 glaciers with remote sensing SMB data (Rabatel et al. 2016)   ####
        glims_rabatel = genfromtxt(os.path.join(path_glims, 'GLIMS_Rabatel_30_2003.csv'), delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), 
                       ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), 
                       ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), 
                       ('x_coord', '<f8'), ('y_coord', '<f8'), ('slope20', '<f8'), ('GLIMS_ID', '<U50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8')])

        # SAFRAN massif indexes
        # We start one year after in order to take into account the previous year's accumulation period
        year_period = range(year_start, year_end+1)
        
        # We read the first year to get some basic information
        dummy_safran = xr.open_dataset(os.path.join(path_safran_forcings, '84', 'FORCING.nc'))
        
        # We get the massif points closer to each glacier's centroid
        glacier_SMB_coordinates, all_glacier_coordinates = get_SAFRAN_glacier_coordinates(dummy_safran['massif_number'], 
                                                                                          dummy_safran['ZS'], dummy_safran['aspect'], 
                                                                                          glims_2003, glims_rabatel)
        smb_glaciers = get_SMB_glaciers(glacier_SMB_coordinates)
        
        # We store the coordinates of all glaciers
        with open(os.path.join(path_smb_function_safran, 'all_glacier_coordinates.txt'), 'wb') as coords_f:
                    np.save(coords_f, all_glacier_coordinates)
        
        # We create the data structures to process all the CPDD and snow accumulation data
        glacier_CPDD_layout = {'Glacier':"", 'GLIMS_ID':"", 'CPDD':[], 'Mean':0.0, 'years':[]}
        glacier_snow_layout = {'Glacier':"", 'GLIMS_ID':"", 'snow':[], 'Mean':0.0, 'years':[]}
        glacier_CPDDs_all, glacier_winter_snow_all, glacier_summer_snow_all = [],[],[]
        glacier_CPDDs_SMB, glacier_winter_snow_SMB, glacier_summer_snow_SMB = [],[],[]
        
        glacier_mon_temp_layout = {'Glacier':"", 'GLIMS_ID':"", 'mon_temp':[], 'mon_means':[], 'years':[]}
        glacier_mon_snow_layout = {'Glacier':"", 'GLIMS_ID':"", 'mon_snow':[], 'mon_means':[], 'years':[]}
        glacier_mon_temp_all, glacier_mon_snow_all = [],[]
        glacier_mon_temp_SMB, glacier_mon_snow_SMB = [],[]
        
        glacier_altitudes = []
        for glacier in all_glacier_coordinates:
#            glacier_massif = int(glacier[4])
            if(np.any(smb_glaciers == glacier[0])):
                current_CPDD_layout_SMB = copy.deepcopy(glacier_CPDD_layout)
                current_CPDD_layout_SMB['Glacier'] = glacier[0]
                current_CPDD_layout_SMB['GLIMS_ID'] = glacier[3]
                current_snow_layout_SMB = copy.deepcopy(glacier_snow_layout)
                current_snow_layout_SMB['Glacier'] = glacier[0]
                current_snow_layout_SMB['GLIMS_ID'] = glacier[3]
                glacier_CPDDs_SMB.append(current_CPDD_layout_SMB)
                glacier_winter_snow_SMB.append(copy.deepcopy(current_snow_layout_SMB))
                glacier_summer_snow_SMB.append(copy.deepcopy(current_snow_layout_SMB))
                
                current_mon_temp_layout_SMB = copy.deepcopy(glacier_mon_temp_layout)
                current_mon_temp_layout_SMB['Glacier'] = glacier[0]
                current_mon_temp_layout_SMB['GLIMS_ID'] = glacier[3]
                current_mon_snow_layout_SMB = copy.deepcopy(glacier_mon_snow_layout)
                current_mon_snow_layout_SMB['Glacier'] = glacier[0]
                current_mon_snow_layout_SMB['GLIMS_ID'] = glacier[3]
                glacier_mon_temp_SMB.append(current_mon_temp_layout_SMB)
                glacier_mon_snow_SMB.append(copy.deepcopy(current_mon_snow_layout_SMB))
                
            current_CPDD_layout = copy.deepcopy(glacier_CPDD_layout)
            current_CPDD_layout['Glacier'] = glacier[0]
            current_CPDD_layout['GLIMS_ID'] = glacier[3]
            current_snow_layout = copy.deepcopy(glacier_snow_layout)
            current_snow_layout['Glacier'] = glacier[0]
            current_snow_layout['GLIMS_ID'] = glacier[3]
            glacier_CPDDs_all.append(current_CPDD_layout)
            glacier_winter_snow_all.append(copy.deepcopy(current_snow_layout))
            glacier_summer_snow_all.append(copy.deepcopy(current_snow_layout))
            
            current_mon_temp_layout_all = copy.deepcopy(glacier_mon_temp_layout)
            current_mon_temp_layout_all['Glacier'] = glacier[0]
            current_mon_temp_layout_all['GLIMS_ID'] = glacier[3]
            current_mon_snow_layout_all = copy.deepcopy(glacier_mon_snow_layout)
            current_mon_snow_layout_all['Glacier'] = glacier[0]
            current_mon_snow_layout_all['GLIMS_ID'] = glacier[3]
            glacier_mon_temp_all.append(current_mon_temp_layout_all)
            glacier_mon_snow_all.append(copy.deepcopy(current_mon_snow_layout_all))
            
            # We interpolate the glacier altitudes
            glacier_altitudes = generate_glacier_altitudes(glacier, glims_1967, glims_1985, glims_2003, glims_2015, glacier_altitudes)
            
        glacier_altitudes = np.asarray(glacier_altitudes)
        
        glacier_CPDDs_all = np.asarray(glacier_CPDDs_all)
        glacier_mon_temp_all = np.asarray(glacier_mon_temp_all)
        glacier_CPDDs_SMB = np.asarray(glacier_CPDDs_SMB)
        glacier_mon_temp_SMB = np.asarray(glacier_mon_temp_SMB)
        glacier_winter_snow_all = np.asarray(glacier_winter_snow_all)
        glacier_mon_snow_all = np.asarray(glacier_mon_snow_all)
        glacier_winter_snow_SMB = np.asarray(glacier_winter_snow_SMB)
        glacier_mon_snow_SMB = np.asarray(glacier_mon_snow_SMB)
        glacier_summer_snow_all = np.asarray(glacier_summer_snow_all)
        glacier_summer_snow_SMB = np.asarray(glacier_summer_snow_SMB)
        
#        local_anomalies = {'CPDD': np.array([]), 'w_snow': np.array([]), 's_snow':np.array([])}
        local_anomalies = {'CPDD': [], 'w_snow': [], 's_snow':[], 'years':[]}
        raw_local_anomalies = copy.deepcopy(local_anomalies)
        
#        local_mon_anomalies = {'temp': np.array([]), 'snow': np.array([])}
        local_mon_anomalies = {'temp': [], 'snow': [], 'years':[]}
        raw_local_mon_anomalies = copy.deepcopy(local_mon_anomalies)
        
        local_anomalies_SMB = copy.deepcopy(local_anomalies)
        raw_local_anomalies_SMB = copy.deepcopy(local_anomalies)
        
        local_mon_anomalies_SMB = copy.deepcopy(local_mon_anomalies)
        raw_local_mon_anomalies_SMB = copy.deepcopy(local_mon_anomalies)
        
        glacier_SMB_coordinates, all_glacier_coordinates = get_SAFRAN_glacier_coordinates(dummy_safran['massif_number'], 
                                                                                          dummy_safran['ZS'], dummy_safran['aspect'], 
                                                                                          glims_2003, glims_rabatel)
        
        daily_temps_years, daily_snow_years, daily_rain_years, daily_dates_years, zs_years = [], [], [], [], []
            
        if(True):
        #if(not (CPDD_generated and winter_snow_generated and summer_snow_generated)):
            print("\nProcessing SAFRAN data...")
#            nfigure = 1
            
            annual_paths = np.array([])
            annual_paths = np.append(annual_paths, os.path.join(path_safran_forcings, str(year_period[0]-1)[-2:], 'FORCING.nc'))
            for year in year_period:
                annual_paths = np.append(annual_paths, os.path.join(path_safran_forcings, str(year)[-2:], 'FORCING.nc'))
            
            start = time.time()
            # We load all SAFRAN years with xarray and dask
            safran_climate = xr.open_mfdataset(annual_paths, concat_dim="time", combine='by_coords', parallel=True)
            
            end = time.time()
            print("\n-> open SAFRAN dataset processing time: " + str(end - start) + " s")
            
            for year in year_period: 
                print("Hydrological year: " + str(year-1) + "-" + str(year))
                
                start = time.time()
                # We load into memory only the current year to speed things up
                # Only two years are loaded: compute dask arrays in memory so computations are faster
                safran_tmean_d = (safran_climate.sel(time = slice(str(year-1)+'-10-01', str(year)+'-09-30'))['Tair'].resample(time="1D").mean() -273.15).compute()
                safran_snow_d = (safran_climate.sel(time = slice(str(year-1)+'-10-01', str(year)+'-09-30'))['Snowf'].resample(time="1D").sum()*3600).compute()
                safran_rain_d = (safran_climate.sel(time = slice(str(year-1)+'-10-01', str(year)+'-09-30'))['Rainf'].resample(time="1D").sum()*3600).compute()
                
                zs = safran_climate['ZS'].sel(time = slice(str(year-1)+'-10-01', str(year)+'-09-30')).compute()
                
                # Store daily raw data for future re-processing
                daily_temps_years.append(safran_tmean_d.data)
                daily_snow_years.append(safran_snow_d.data)
                daily_rain_years.append(safran_rain_d.data)
                daily_dates_years.append(safran_tmean_d.time.data)
                zs_years.append(zs.data)
                
                if(not bypass_glacier_data):
                
                    # Initialize year and glacier indexes
                    i, j = 0, 0
                    
                    for glacier_coords, glacier_alt in zip(all_glacier_coordinates, glacier_altitudes):
    
                        # Reset the indexes
                        glacier_name = glacier_coords[0]
                        glacier_idx = int(glacier_coords[1])
    #                    glims_ID = glacier_coords[3]
    #                    glacier_massif = int(glacier_coords[4])
                        
                        if(len(glacier_alt) > 1):
                            glacier_alt_y = glacier_alt[np.where(year == np.asarray(year_period))[0][0]]
                        else:
                            glacier_alt_y = glacier_alt[0]
                        
                        # Re-scale temperature at glacier's actual altitude
                        safran_tmean_d_g = copy.deepcopy(safran_tmean_d[:, glacier_idx] + ((zs[0,glacier_idx].data - glacier_alt_y)/1000.0)*6.0)
                        
                        # We adjust the snowfall rate at the glacier's altitude
                        safran_snow_d_g = copy.deepcopy(safran_snow_d[:, glacier_idx])
                        safran_rain_d_g = copy.deepcopy(safran_rain_d[:, glacier_idx])
                        safran_snow_d_g.data = np.where(safran_tmean_d_g.data > t_lim, 0.0, safran_snow_d_g.data)
                        safran_snow_d_g.data = np.where(safran_tmean_d_g.data < t_lim, safran_snow_d_g.data + safran_rain_d_g.data, safran_snow_d_g.data)
                        
                        # Monthly data during the current hydrological year
                        # Compute dask arrays prior to storage
                        safran_tmean_m_g = safran_tmean_d_g.resample(time="1MS").mean().data
                        safran_snow_m_g = safran_snow_d_g.resample(time="1MS").sum().data
                        
                        # Compute CPDD
                        # Compute dask arrays prior to storage
                        glacier_CPDD = np.sum(np.where(safran_tmean_d_g.data < 0, 0, safran_tmean_d_g.data))
                        glacier_CPDDs_all[j]['years'].append(year)
                        glacier_CPDDs_all[j]['CPDD'].append(glacier_CPDD)
                        
                        # Compute snowfall
                        # Compute dask arrays prior to storage
                        glacier_year_accum_snow = np.sum(safran_snow_d_g.sel(time = slice(str(year-1)+'-10-01', str(year)+'-03-31')).data)
                        glacier_year_ablation_snow = np.sum(safran_snow_d_g.sel(time = slice(str(year)+'-04-01', str(year)+'-09-30')).data)
                        
                        glacier_winter_snow_all[j]['years'].append(year) 
                        glacier_winter_snow_all[j]['snow'].append(glacier_year_accum_snow)
                        glacier_summer_snow_all[j]['years'].append(year)
                        glacier_summer_snow_all[j]['snow'].append(glacier_year_ablation_snow)
                        
                        glacier_mon_temp_all[j]['years'].append(year)
                        glacier_mon_temp_all[j]['mon_temp'].append(safran_tmean_m_g)
                        glacier_mon_snow_all[j]['mon_snow'].append(safran_snow_m_g)
                        
                        # Now we store the data for the sub-dataset of glaciers with SMB data
                        if(np.any(smb_glaciers == glacier_name)):
                            glacier_CPDDs_SMB[i]['years'].append(year)
                            glacier_CPDDs_SMB[i]['CPDD'].append(glacier_CPDD)
                            
                            glacier_winter_snow_SMB[i]['years'].append(year)
                            glacier_winter_snow_SMB[i]['snow'].append(glacier_year_accum_snow)
                            
                            glacier_summer_snow_SMB[i]['years'].append(year)
                            glacier_summer_snow_SMB[i]['snow'].append(glacier_year_ablation_snow)
                            
                            glacier_mon_temp_SMB[i]['years'].append(year)
                            glacier_mon_temp_SMB[i]['mon_temp'].append(safran_tmean_m_g)
                            glacier_mon_snow_SMB[i]['mon_snow'].append(safran_snow_m_g)
                            
                        # If we reach the end of the time period, we compute the local anomalies and means
                        if(year == year_period[-1]):
                            
                            glacier_CPDDs_all, glacier_winter_snow_all, glacier_summer_snow_all, local_anomalies, raw_local_anomalies = compute_local_anomalies(j, glacier_CPDDs_all, 
                                                                                                                                                                glacier_winter_snow_all, 
                                                                                                                                                                glacier_summer_snow_all,  
                                                                                                                                                                local_anomalies, 
                                                                                                                                                                raw_local_anomalies)
                            
                            glacier_mon_temp_all, glacier_mon_snow_all, local_mon_anomalies, raw_local_mon_anomalies = compute_monthly_anomalies(j, glacier_mon_temp_all, 
                                                                                                                                                 glacier_mon_snow_all, 
                                                                                                                                                 local_mon_anomalies, 
                                                                                                                                                 raw_local_mon_anomalies)
                            
                            
                            
                            # Glaciers with SMB data
                            if(np.any(smb_glaciers == glacier_name)):
                                glacier_CPDDs_SMB, glacier_winter_snow_SMB, glacier_summer_snow_SMB, local_anomalies_SMB, raw_local_anomalies_SMB = compute_local_anomalies(i, glacier_CPDDs_SMB, 
                                                                                                                                                                            glacier_winter_snow_SMB, 
                                                                                                                                                                            glacier_summer_snow_SMB,  
                                                                                                                                                                            local_anomalies_SMB, 
                                                                                                                                                                            raw_local_anomalies_SMB)
                                
                                glacier_mon_temp_SMB, glacier_mon_snow_SMB, local_mon_anomalies_SMB, raw_local_mon_anomalies_SMB = compute_monthly_anomalies(i, glacier_mon_temp_SMB, 
                                                                                                                                                             glacier_mon_snow_SMB, 
                                                                                                                                                             local_mon_anomalies_SMB, 
                                                                                                                                                             raw_local_mon_anomalies_SMB)
                            
                            
                        ### End of glacier loop  ###
                        
                        
                        # We iterate the independent indexes
                        j = j+1
                        if(np.any(smb_glaciers == glacier_name)):
                            i = i+1
                   
                    end = time.time()
                    print("-> processing time: " + str(end - start) + " s")
                
            ### End of years loop ###
            
            if(not bypass_glacier_data):
                # We combine the meteo SMB dataset forcings
                # Seasonal
                season_meteo_SMB = {'CPDD':glacier_CPDDs_SMB, 'winter_snow':glacier_winter_snow_SMB, 'summer_snow':glacier_summer_snow_SMB}
                
                # Monthly
                monthly_meteo_SMB = {'temp':glacier_mon_temp_SMB, 'snow':glacier_mon_snow_SMB}
                
                # We sort the SMB meteo forcings data to fit the order to the GLIMS Rabatel file                                                                                                                                                                                                      
                season_meteo_SMB, season_meteo_anomalies_SMB, season_raw_meteo_anomalies_SMB, monthly_meteo_SMB, monthly_meteo_anomalies_SMB = sort_SMB_rabatel_data(season_meteo_SMB, 
                                                                                                                                                                     local_anomalies_SMB, 
                                                                                                                                                                     raw_local_anomalies_SMB, 
                                                                                                                                                                     monthly_meteo_SMB, 
                                                                                                                                                                     local_mon_anomalies_SMB,
                                                                                                                                                                     glims_rabatel)
                # We combine the full meteo dataset forcings for all glaciers
                # Seasonal
                season_meteo = {'CPDD':glacier_CPDDs_all, 'winter_snow':glacier_winter_snow_all, 'summer_snow':glacier_summer_snow_all}
                season_meteo_anomalies = {'CPDD':local_anomalies['CPDD'], 'winter_snow':local_anomalies['w_snow'], 'summer_snow':local_anomalies['s_snow']}
                season_raw_meteo_anomalies = {'CPDD':raw_local_anomalies['CPDD'], 'winter_snow':raw_local_anomalies['w_snow'], 'summer_snow':raw_local_anomalies['s_snow']}
                # Monthly
                monthly_meteo = {'temp':glacier_mon_temp_all, 'snow':glacier_mon_snow_all}
                monthly_meteo_anomalies = {'temp':local_mon_anomalies['temp'], 'snow':local_mon_anomalies['snow']}
                                                                                                                                                                   
                # If the forcing folder is not created we create it
                if not os.path.exists(path_smb_function_safran):
                    # We create a new folder in order to store the raster plots
                    os.makedirs(path_smb_function_safran)
                    
                # We store the compacted seasonal and monthly meteo forcings
                # Glaciers with SMB data (machine learning model training)
                with open(os.path.join(path_smb_function_safran, 'season_meteo_SMB.txt'), 'wb') as season_f:
                            np.save(season_f, season_meteo_SMB)
                with open(os.path.join(path_smb_function_safran, 'season_meteo_anomalies_SMB.txt'), 'wb') as season_a_f:
                            np.save(season_a_f, season_meteo_anomalies_SMB)
                with open(os.path.join(path_smb_function_safran, 'season_raw_meteo_anomalies_SMB.txt'), 'wb') as season_raw_a_f:
                            np.save(season_raw_a_f, season_raw_meteo_anomalies_SMB)
                with open(os.path.join(path_smb_function_safran, 'monthly_meteo_SMB.txt'), 'wb') as mon_f:
                            np.save(mon_f, monthly_meteo_SMB)
                with open(os.path.join(path_smb_function_safran, 'monthly_meteo_anomalies_SMB.txt'), 'wb') as mon_a_f:
                            np.save(mon_a_f, monthly_meteo_anomalies_SMB)
                
                # All glaciers
                with open(os.path.join(path_smb_function_safran, 'season_meteo.txt'), 'wb') as season_f:
                            np.save(season_f, season_meteo)
                with open(os.path.join(path_smb_function_safran, 'season_meteo_anomalies.txt'), 'wb') as season_a_f:
                            np.save(season_a_f, season_meteo_anomalies)
                with open(os.path.join(path_smb_function_safran, 'season_raw_meteo_anomalies.txt'), 'wb') as season_raw_a_f:
                            np.save(season_raw_a_f, season_raw_meteo_anomalies)
                with open(os.path.join(path_smb_function_safran, 'monthly_meteo.txt'), 'wb') as mon_f:
                            np.save(mon_f, monthly_meteo)
                with open(os.path.join(path_smb_function_safran, 'monthly_meteo_anomalies.txt'), 'wb') as mon_a_f:
                            np.save(mon_a_f, monthly_meteo_anomalies)
                        
            
            # We store the base SAFRAN data for future runs
            with open(path_temps, 'wb') as dtemp_f:
                    np.save(dtemp_f, daily_temps_years)
            with open(path_snow, 'wb') as dsnow_f:
                    np.save(dsnow_f, daily_snow_years)
            with open(path_rain, 'wb') as drain_f:
                    np.save(drain_f, daily_rain_years)
            with open(path_dates, 'wb') as ddates_f:
                    np.save(ddates_f, daily_dates_years)
            with open(path_zs, 'wb') as dzs_f:
                    np.save(dzs_f, zs_years)
            
    else:
         print("Skipping...")           
    ###  End of main function   ###