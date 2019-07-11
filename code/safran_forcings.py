# -*- coding: utf-8 -*-

"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

SAFRAN TEMPERATURE (CPDD) AND PRECIPITATION (SNOW) COMPUTATION

"""

## Dependencies: ##
from netCDF4 import Dataset
import numpy as np
from numpy import genfromtxt
from numba import jit
import copy
#import matplotlib.pyplot as plt
import os
from difflib import SequenceMatcher
import math
import pandas as pd
from pathlib import Path

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

def find_glacier_idx(glacier_massif, massif_number, altitudes, glacier_altitude, aspects, glacier_aspect):
    massif_altitudes_idx = np.where(massif_number == float(glacier_massif))[0]
    glacier_aspect_idx = np.where(aspects == float(glacier_aspect))[0]
    massif_alt_aspect_idx = np.array(list(set(massif_altitudes_idx).intersection(glacier_aspect_idx)))
    index_alt = find_nearest(altitudes[massif_alt_aspect_idx], glacier_altitude)
    final_idx = int(massif_alt_aspect_idx[index_alt])
    
    return final_idx

@jit
def get_SAFRAN_glacier_coordinates(massif_number, zs, aspects, glims_data, glims_rabatel):
    glacier_centroid_altitude = glims_data['MEDIAN_Pixel']
    GLIMS_IDs = glims_data['GLIMS_ID']
    glacier_massifs = glims_data['Massif_SAFRAN']
    glacier_names = glims_data['Glacier']
    glacier_aspects = glims_data['Aspect_num']
    glacier_SMB_coordinates, all_glacier_coordinates = [], []
    
    # Glaciers with SMB Data (Rabatel et al. 2016)
    for glims_id, glacier_name, glacier_massif, glacier_altitude, glacier_aspect in zip(glims_rabatel['GLIMS_ID'], glims_rabatel['Glacier'], glims_rabatel['Massif_SAFRAN'], glims_rabatel['MEDIAN_Pixel'], glims_rabatel['Aspect_num']):
        glacier_SMB_coordinates.append([glacier_name, find_glacier_idx(glacier_massif, massif_number, zs, glacier_altitude, aspects, glacier_aspect), float(glacier_altitude), glims_id, int(glacier_massif)])
    
    # All glaciers loop
    for glims_id, glacier_name, glacier_massif, glacier_altitude, glacier_aspect in zip(GLIMS_IDs, glacier_names, glacier_massifs, glacier_centroid_altitude, glacier_aspects):
        all_glacier_coordinates.append([glacier_name, find_glacier_idx(glacier_massif, massif_number, zs, glacier_altitude, aspects, glacier_aspect), float(glacier_altitude), glims_id, int(glacier_massif)])
        
    return np.asarray(glacier_SMB_coordinates), np.asarray(all_glacier_coordinates)

@jit
# Computes daily temps from hourly data
def get_mean_temps(datetimes, hourly_data):
    ref_day = -9
    daily_data = []
    idx, day_idx = 0, 0
    first = True
    for time_hour in datetimes:
        current_day = time_hour.astype(object).timetuple().tm_yday
        if(current_day == ref_day):
            daily_data[day_idx] = daily_data[day_idx] + hourly_data[idx]/24.0
        else:
            ref_day = current_day
            daily_data.append(hourly_data[idx]/24.0)
            if(not first):
                day_idx = day_idx + 1
            else:
                first = False
            
        idx = idx + 1 
        
    return np.asarray(daily_data)

@jit
# Computes daily precipitations from hourly data
def get_precips(datetimes, hourly_data):
    ref_day = -9
    daily_data = []
    idx, day_idx = 0, 0
    isFirst = True
    for time_hour in datetimes:
        current_day = time_hour.astype(object).timetuple().tm_yday
        if(current_day == ref_day):
            daily_data[day_idx] = daily_data[day_idx] + hourly_data[idx]
        else:
            ref_day = current_day
            daily_data.append(hourly_data[idx])
            if(not isFirst):                 
                day_idx = day_idx + 1
            else:
                isFirst = False
        idx = idx + 1 
    return np.asarray(daily_data)

@jit
# Computes monthly temperature from daily data
def get_monthly_temps(daily_data, daily_datetimes):
    d = {'Dates': daily_datetimes, 'Temps': daily_data}
    df_datetimes = pd.DataFrame(data=d)
    df_datetimes.set_index('Dates', inplace=True)
    df_datetimes.index = pd.to_datetime(df_datetimes.index)
    df_datetimes = df_datetimes.resample('M').mean()
    
    monthly_avg_data = df_datetimes.Temps.to_numpy()
    
    return monthly_avg_data[:12]
    

@jit
# Computes monthly snowfall from daily data
def get_monthly_snow(daily_data, daily_datetimes):
    d = {'Dates': daily_datetimes, 'Temps': daily_data}
    df_datetimes = pd.DataFrame(data=d)
    df_datetimes.set_index('Dates', inplace=True)
    df_datetimes.index = pd.to_datetime(df_datetimes.index)
    df_datetimes = df_datetimes.resample('M').sum()
    
    monthly_avg_data = df_datetimes.Temps.to_numpy()
    
    return monthly_avg_data[:12]

# Computes seasonal meteo anomalies at glacier scale
def compute_local_anomalies(idx, glacier_CPDDs, glacier_winter_snow, glacier_summer_snow, 
                      local_CPDD_anomalies, raw_local_CPDD_anomalies,
                      local_w_snow_anomalies, raw_local_w_snow_anomalies,
                      local_s_snow_anomalies, raw_local_s_snow_anomalies):
    
    # The anomalies are always computed with respect to the 1984-2014 mean
    glacier_CPDDs[idx]['CPDD'] = np.asarray(glacier_CPDDs[idx]['CPDD'])
    glacier_CPDDs_training = glacier_CPDDs[idx]['CPDD'][1::2][-32:]
    glacier_CPDDs[idx]['Mean'] = glacier_CPDDs_training.mean()
    glacier_winter_snow[idx]['snow'] = np.asarray(glacier_winter_snow[idx]['snow'])
    glacier_winter_snow_training = glacier_winter_snow[idx]['snow'][1::2][-32:]
    glacier_winter_snow[idx]['Mean'] = glacier_winter_snow_training.mean()
    glacier_summer_snow[idx]['snow'] = np.asarray(glacier_summer_snow[idx]['snow'])
    glacier_summer_snow_training = glacier_summer_snow[idx]['snow'][1::2][-32:]
    glacier_summer_snow[idx]['Mean'] = glacier_summer_snow_training.mean()
    
    local_CPDD_anomalies.append(copy.deepcopy(glacier_CPDDs[idx]))
    local_CPDD_anomalies[-1]['CPDD'][1::2] = glacier_CPDDs[idx]['CPDD'][1::2] - glacier_CPDDs[idx]['Mean']
    raw_local_CPDD_anomalies.append(local_CPDD_anomalies[-1]['CPDD'][1::2])
    
    local_w_snow_anomalies.append(copy.deepcopy(glacier_winter_snow[idx]))
    local_w_snow_anomalies[-1]['snow'][1::2] = glacier_winter_snow[idx]['snow'][1::2] - glacier_winter_snow[idx]['Mean']
    raw_local_w_snow_anomalies.append(local_w_snow_anomalies[-1]['snow'][1::2]) 
    
    local_s_snow_anomalies.append(copy.deepcopy(glacier_summer_snow[idx]))
    local_s_snow_anomalies[-1]['snow'][1::2] = glacier_summer_snow[idx]['snow'][1::2] - glacier_summer_snow[idx]['Mean']
    raw_local_s_snow_anomalies.append(local_s_snow_anomalies[-1]['snow'][1::2])
    
    return glacier_CPDDs, glacier_winter_snow, glacier_summer_snow, local_CPDD_anomalies, raw_local_CPDD_anomalies, local_w_snow_anomalies, raw_local_w_snow_anomalies, local_s_snow_anomalies, raw_local_s_snow_anomalies


# Computes monthly meteo anomalies at glacier scale
def compute_monthly_anomalies(idx, glacier_mon_temp, glacier_mon_snow,
                              local_mon_temp_anomalies, local_mon_snow_anomalies,
                              raw_local_mon_temp_anomalies, raw_local_mon_snow_anomalies):
    
    # The monthly meteo anomalies, as well as the seasonal ones, are always computed with respect to the 1984-2014 period
    mon_range = range(0, 12)
    
    for mon_idx in mon_range:
        mon_avg_temp, mon_avg_snow = [],[]
        for glacier_temp, glacier_snow in zip(glacier_mon_temp[idx]['mon_temp'], glacier_mon_snow[idx]['mon_snow']):
            mon_avg_temp.append(glacier_temp[mon_idx])
            mon_avg_snow.append(glacier_snow[mon_idx])
        mon_avg_temp = np.asarray(mon_avg_temp)
        mon_avg_snow = np.asarray(mon_avg_snow)
        
        glacier_mon_temp[idx]['mon_means'].append(mon_avg_temp[-32:].mean())
        glacier_mon_snow[idx]['mon_means'].append(mon_avg_snow[-32:].mean())
        
    local_mon_temp_anomalies.append(copy.deepcopy(glacier_mon_temp[idx]))
    local_mon_snow_anomalies.append(copy.deepcopy(glacier_mon_snow[idx]))
    
    for mon_idx in mon_range:
        year_idx = 0
        for glacier_temp, glacier_snow in zip(local_mon_temp_anomalies[-1]['mon_temp'], local_mon_snow_anomalies[-1]['mon_snow']):
             local_mon_temp_anomalies[-1]['mon_temp'][year_idx][mon_idx] = local_mon_temp_anomalies[-1]['mon_temp'][year_idx][mon_idx] - local_mon_temp_anomalies[-1]['mon_means'][mon_idx]  
             local_mon_snow_anomalies[-1]['mon_snow'][year_idx][mon_idx] = local_mon_snow_anomalies[-1]['mon_snow'][year_idx][mon_idx] - local_mon_snow_anomalies[-1]['mon_means'][mon_idx]
             year_idx = year_idx+1
    
    return glacier_mon_temp, glacier_mon_snow, local_mon_temp_anomalies, local_mon_snow_anomalies


@jit
def get_SMB_glaciers(glacier_SMB_coordinates):
    smb_glaciers = []
    for glacier in glacier_SMB_coordinates:
        smb_glaciers.append(glacier[0])
    return np.asarray(smb_glaciers)

# Sorts the temperature and snow data in the same order as the GLIMS Rabatel 2003 file
def sort_SMB_rabatel_data(season_meteo_SMB, season_meteo_anomalies_SMB, season_raw_meteo_anomalies_SMB, monthly_meteo_SMB, monthly_meteo_anomalies_SMB, glims_rabatel):
    glacier_order = []
    
    # We first capture the glacier order
    for glims_id in glims_rabatel['GLIMS_ID']:
        glacier_idx = 0
        for cpdd_smb_glacier in season_meteo_SMB['CPDD']:
            if(glims_id == cpdd_smb_glacier['GLIMS_ID']):
                glacier_order.append(glacier_idx)
            glacier_idx = glacier_idx+1
            
    # And then we apply it to all the datasets    
    glacier_order = np.asarray(glacier_order, dtype=np.int32)
    
    season_meteo_SMB['CPDD'] = season_meteo_SMB['CPDD'][glacier_order] 
    season_meteo_SMB['winter_snow'] = season_meteo_SMB['winter_snow'][glacier_order]  
    season_meteo_SMB['summer_snow'] = season_meteo_SMB['summer_snow'][glacier_order]  
    
    season_meteo_anomalies_SMB['CPDD'] = season_meteo_anomalies_SMB['CPDD'][glacier_order]  
    season_meteo_anomalies_SMB['winter_snow'] = season_meteo_anomalies_SMB['winter_snow'][glacier_order] 
    season_meteo_anomalies_SMB['summer_snow'] = season_meteo_anomalies_SMB['summer_snow'][glacier_order] 
    
    season_raw_meteo_anomalies_SMB['CPDD'] = season_raw_meteo_anomalies_SMB['CPDD'][glacier_order]  
    season_raw_meteo_anomalies_SMB['winter_snow'] = season_raw_meteo_anomalies_SMB['winter_snow'][glacier_order] 
    season_raw_meteo_anomalies_SMB['summer_snow'] = season_raw_meteo_anomalies_SMB['summer_snow'][glacier_order] 
    
    monthly_meteo_SMB['temp'] = monthly_meteo_SMB['temp'][glacier_order]
    monthly_meteo_SMB['snow'] = monthly_meteo_SMB['snow'][glacier_order]
    
    monthly_meteo_anomalies_SMB['temp'] = monthly_meteo_anomalies_SMB['temp'][glacier_order]
    monthly_meteo_anomalies_SMB['snow'] = monthly_meteo_anomalies_SMB['snow'][glacier_order]
    
    return season_meteo_SMB, season_meteo_anomalies_SMB, season_raw_meteo_anomalies_SMB, monthly_meteo_SMB, monthly_meteo_anomalies_SMB

# Interpolates topo data to build the training matrix
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
        workspace = str(Path(os.getcwd()).parent) + '\\'
        path_safran_forcings = 'C:\\Jordi\\PhD\\Data\\SAFRAN-Nivo-2017\\'
        path_smb = workspace + 'glacier_data\\smb\\'
        path_smb_function_safran = path_smb + 'smb_function\\SAFRAN\\'
        path_glims = workspace + 'glacier_data\\GLIMS\\' 
        
        year_start = 1984
#        year_start = 1959
        year_end = 2015
        
        path_temps = path_smb_function_safran +'daily_temps_years_' + str(year_start) + '-' + str(year_end) + '.txt'
        path_snow = path_smb_function_safran +'daily_snow_years_' + str(year_start) + '-' + str(year_end) + '.txt'
        path_rain = path_smb_function_safran +'daily_rain_years_' + str(year_start) + '-' + str(year_end) + '.txt'
        path_dates = path_smb_function_safran +'daily_dates_years_' + str(year_start) + '-' + str(year_end) + '.txt'
        path_zs = path_smb_function_safran +'zs_years' + str(year_start) + '-' + str(year_end) + '.txt'
        
        #### GLIMS data for 1985, 2003 and 2015
        glims_2015 = genfromtxt(path_glims + 'GLIMS_2015.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50')])
        glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8')])
        glims_1985 = genfromtxt(path_glims + 'GLIMS_1985.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50')])
        glims_1967 = genfromtxt(path_glims + 'GLIMS_1967.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<U50')])

        
        ####  GLIMS data for the 30 glaciers with remote sensing SMB data (Rabatel et al. 2016)   ####
        glims_rabatel = genfromtxt(path_glims + 'GLIMS_Rabatel_30_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<U50'), ('Annee', '<i8'), ('Massif', '<U50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<U50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('slope20', '<f8'), ('GLIMS_ID', '<U50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8')])

        # SAFRAN massif indexes
        # We start one year after in order to take into account the previous year's accumulation period
        year_period = range(year_start, year_end+1)
        
        # We read the first year to get some basic information
        dummy_SAFRAN_forcing = Dataset(path_safran_forcings + '84\\FORCING.nc')
        
        aspects = dummy_SAFRAN_forcing.variables['aspect'][:]
        zs = dummy_SAFRAN_forcing.variables['ZS'][:]
        massif_number = dummy_SAFRAN_forcing.variables['massif_number'][:]
        
        # We get the massif points closer to each glacier's centroid
        glacier_SMB_coordinates, all_glacier_coordinates = get_SAFRAN_glacier_coordinates(massif_number, zs, aspects, glims_2003, glims_rabatel)
        smb_glaciers = get_SMB_glaciers(glacier_SMB_coordinates)
        
        # We store the coordinates of all glaciers
        with open(path_smb_function_safran+'all_glacier_coordinates.txt', 'wb') as coords_f:
                    np.save(coords_f, all_glacier_coordinates)
        
        # We create the data structures to process all the CPDD and snow accumulation data
        glacier_CPDD_layout = {'Glacier':"", 'GLIMS_ID':"", 'CPDD':[], 'Mean':0.0}
        glacier_snow_layout = {'Glacier':"", 'GLIMS_ID':"", 'snow':[], 'Mean':0.0}
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
        
        local_CPDD_anomalies, local_w_snow_anomalies, local_s_snow_anomalies = [],[],[]
        raw_local_CPDD_anomalies, raw_local_w_snow_anomalies, raw_local_s_snow_anomalies = [],[],[]
        raw_yearly_mean_CPDD, raw_yearly_mean_winter_snow, raw_yearly_mean_summer_snow, raw_yearly_mean_ablation_season = [],[],[],[]
        
        local_mon_temp_anomalies, local_mon_snow_anomalies = [],[]
        raw_local_mon_temp_anomalies, raw_local_mon_snow_anomalies = [],[]
        
        local_CPDD_anomalies_SMB, local_w_snow_anomalies_SMB, local_s_snow_anomalies_SMB = [],[],[]
        raw_local_CPDD_anomalies_SMB, raw_local_w_snow_anomalies_SMB, raw_local_s_snow_anomalies_SMB = [],[],[]
        raw_yearly_mean_CPDD_SMB, raw_yearly_mean_winter_snow_SMB, raw_yearly_mean_summer_snow_SMB, raw_yearly_mean_ablation_season_SMB = [],[],[],[]
        
        local_mon_temp_anomalies_SMB, local_mon_snow_anomalies_SMB = [],[]
        raw_local_mon_temp_anomalies_SMB, raw_local_mon_snow_anomalies_SMB = [],[]
        
        glacier_SMB_coordinates, all_glacier_coordinates = get_SAFRAN_glacier_coordinates(massif_number, zs, aspects, glims_2003, glims_rabatel)
            
        if(True):
        #if(not (CPDD_generated and winter_snow_generated and summer_snow_generated)):
            print("\nProcessing SAFRAN data...")
            daily_temps_years, daily_snow_years, daily_rain_years, daily_dates_years, zs_years = [],[],[],[],[]
#            nfigure = 1
            for year in year_period:
                print("Hydrological year: " + str(year-1) + "-" + str(year))
                
                # We also need to fetch the previous year since data goes from 1st of August to 31st of July
                SAFRAN_forcing_1 = Dataset(path_safran_forcings + str(year-1)[-2:] + '\\FORCING.nc')
                current_SAFRAN_forcing = Dataset(path_safran_forcings + str(year)[-2:] + '\\FORCING.nc')
                
                zs = current_SAFRAN_forcing.variables['ZS'][:]
                zs_years.append(zs)
                
                massif_number = current_SAFRAN_forcing.variables['massif_number'][:]
                
                # Temperatures (from K to C)
                temps_mean_1 = SAFRAN_forcing_1.variables['Tair'][:] -273.15
                temps_mean = current_SAFRAN_forcing.variables['Tair'][:] -273.15
                
                snow_h_1 = SAFRAN_forcing_1.variables['Snowf'][:]*3600
                snow_h = current_SAFRAN_forcing.variables['Snowf'][:]*3600
                
                rain_h_1 = SAFRAN_forcing_1.variables['Rainf'][:]*3600
                rain_h = current_SAFRAN_forcing.variables['Rainf'][:]*3600
                
                times_1 = current_SAFRAN_forcing.variables['time'][:]
                start_1 = np.datetime64(str(year-1) + '-08-01 06:00:00')
                datetimes_1 = np.array([start_1 + np.timedelta64(np.int32(time), 'h') for time in times_1])
                times = current_SAFRAN_forcing.variables['time'][:]
                start = np.datetime64(str(year) + '-08-01 06:00:00')
                datetimes = np.array([start + np.timedelta64(np.int32(time), 'h') for time in times])
                
                # We convert the temperature to daily accumulated snowfall
                daily_temps_mean_1 = get_mean_temps(datetimes_1, temps_mean_1)
                daily_temps_mean = get_mean_temps(datetimes, temps_mean)
                
                # We convert the snow to daily accumulated snowfall
                snow_sum_1 = get_precips(datetimes_1, snow_h_1)
                snow_sum = get_precips(datetimes, snow_h)
                rain_sum_1 = get_precips(datetimes_1, rain_h_1)
                rain_sum = get_precips(datetimes, rain_h)
                
                # We get the daily datetimes
                daily_datetimes_1 = datetimes_1[::24]
                daily_datetimes = datetimes[::24]
                
                if(year == year_start):
                    daily_temps_years.append(daily_temps_mean_1)
                    daily_snow_years.append(snow_sum_1)
                    daily_rain_years.append(rain_sum_1)
                    daily_dates_years.append(daily_datetimes_1)
                    
                daily_temps_years.append(daily_temps_mean)
                daily_snow_years.append(snow_sum)
                daily_rain_years.append(rain_sum)
                daily_dates_years.append(daily_datetimes)
                
                # We get the indexes for the precipitation ablation and accumulation periods
                # Classic 120-270 ablation period
                ablation_idx_1 = range(daily_datetimes_1.size-95, daily_datetimes_1.size)
                ablation_idx = range(0, 55)
                accum_idx_1 = range(56, daily_datetimes_1.size-96)
                
                year_CPDD, year_w_snow, year_s_snow, year_ablation_season = 0,0,0,0
                year_CPDD_SMB, year_w_snow_SMB, year_s_snow_SMB, year_ablation_season_SMB = 0,0,0,0
                
                i, j = 0, 0
                start_div, end_div = 30, 10
                
                for glacier_coords, glacier_alt in zip(all_glacier_coordinates, glacier_altitudes):

                    # Reset the indexes
                    glacier_name = glacier_coords[0]
                    glacier_idx = int(glacier_coords[1])
#                    glims_ID = glacier_coords[3]
#                    glacier_massif = int(glacier_coords[4])
                    if(len(glacier_alt) > 1):
                        glacier_alt_y = glacier_alt[np.where(year == np.asarray(year_period))[0][0]]
                    else:
                        glacier_alt_y = glacier_alt
                        
                    #### We compute the monthly average data
                    
                    # Monthly average temperature
                    glacier_temps_1 = daily_temps_mean_1[:, glacier_idx] + ((zs[glacier_idx] - glacier_alt_y)/1000.0)*6.0
                    glacier_temps = daily_temps_mean[:, glacier_idx] + ((zs[glacier_idx] - glacier_alt_y)/1000.0)*6.0
                    mon_temps_1 = get_monthly_temps(glacier_temps_1, daily_datetimes_1) 
                    mon_temps = get_monthly_temps(glacier_temps, daily_datetimes)
                    mon_temp_year = np.append(mon_temps_1[2:], mon_temps[:2])
                    
                    # Monthly average snowfall
                    # We adjust the snowfall rate at the glacier's altitude
                    glacier_snow_1 = snow_sum_1[:, glacier_idx] 
                    glacier_snow = snow_sum[:, glacier_idx] 
                    glacier_rain_1 = rain_sum_1[:, glacier_idx] 
                    glacier_rain = rain_sum[:, glacier_idx] 
                    glacier_snow_1 = np.where(glacier_temps_1 > 2.0, 0.0, glacier_snow_1)
                    glacier_snow_1 = np.where(((glacier_temps_1 < 2.0) & (glacier_snow_1 == 0.0)), glacier_rain_1, glacier_snow_1)
                    glacier_snow = np.where(glacier_temps > 2.0, 0.0, glacier_snow)
                    glacier_snow = np.where(((glacier_temps < 2.0) & (glacier_snow == 0.0)), glacier_rain, glacier_snow)
                    
                    mon_snow_1 = get_monthly_snow(glacier_snow_1, daily_datetimes_1)
                    mon_snow = get_monthly_snow(glacier_snow, daily_datetimes)
                    mon_snow_year = np.append(mon_snow_1[2:], mon_snow[:2])
                    
                    year_1_offset = 213
                    year_offset = 152
                    
                    temp_year = np.append(daily_temps_mean_1[-year_1_offset:, glacier_idx], daily_temps_mean[:year_offset, glacier_idx]) + ((zs[glacier_idx] - glacier_alt_y)/1000.0)*6.0
                    
                    pos_temp_year = np.where(temp_year < 0, 0, temp_year)
                    integ_temp = np.cumsum(pos_temp_year)
                    
                    ### Dynamic temperature ablation period
                    start_y_ablation = np.where(integ_temp > integ_temp.max()/start_div)[0]
        #            start_y_ablation = np.where(integ_temp > 30)[0]
                    end_y_ablation = np.where(integ_temp > (integ_temp.max() - integ_temp.max()/end_div))[0]
                    
                    start_ablation = start_y_ablation[0] + (daily_temps_mean_1[:,glacier_idx].size - year_1_offset)
                    end_ablation = end_y_ablation[0] - year_1_offset
                    
                    # Plot ablation season evolution
#                    if(j == 432):
#                        print("Glacier: " + str(glacier_name))
#                        print("Year: " + str(year))
#                        
#                        plt.figure(nfigure)
#                        plt.title("Ablation season length", fontsize=20)
#                        plt.ylabel('CPDD', fontsize=15)
#                        plt.xlabel('Day of year', fontsize=15)
#                        plt.tick_params(labelsize=15)
#                        plt.legend()
#                        plt.plot(integ_temp, color='maroon',  linewidth=4)
#                        plt.axvspan(start_y_ablation[0], end_y_ablation[0], color='lightcoral')
#                        nfigure = nfigure+1
#                        plt.show()

                    if(start_ablation > 366):
                        start_ablation = 366
        #            
                    # We get the indexes for the ablation and accumulation periods
                    ablation_temp_idx_1 = range(start_ablation, daily_datetimes_1.size)
                    ablation_temp_idx = range(0, end_ablation)
                    
                    # We correct the glacier's temperature depending on the glacier's altitude
                    # We use deepcopy in order to update the temperature values
                    glacier_ablation_temps = copy.deepcopy(np.append(daily_temps_mean_1[ablation_temp_idx_1, glacier_idx], daily_temps_mean[ablation_temp_idx, glacier_idx])) + ((zs[glacier_idx] - glacier_alt_y)/1000.0)*6.0
                    
                    dummy_glacier_ablation_temps = copy.deepcopy(np.append(daily_temps_mean_1[ablation_idx_1, glacier_idx], daily_temps_mean[ablation_idx, glacier_idx]) + ((zs[glacier_idx] - glacier_alt_y)/1000.0)*6.0)
                    dummy_glacier_accumulation_temps = copy.deepcopy(daily_temps_mean_1[accum_idx_1, glacier_idx] + ((zs[glacier_idx] - glacier_alt_y)/1000.0)*6.0)
                    
                    glacier_year_pos_temps = np.where(glacier_ablation_temps < 0, 0, glacier_ablation_temps)
                    dummy_glacier_ablation_pos_temps = np.where(dummy_glacier_ablation_temps < 0, 0, dummy_glacier_ablation_temps)
                    dummy_glacier_accum_pos_temps = np.where(dummy_glacier_accumulation_temps < 0, 0, dummy_glacier_accumulation_temps)
                    
                    glacier_accum_snow = snow_sum_1[accum_idx_1, glacier_idx]
                    glacier_accum_rain = rain_sum_1[accum_idx_1, glacier_idx]
                    
                    glacier_ablation_snow = np.append(snow_sum_1[ablation_idx_1, glacier_idx], snow_sum[ablation_idx, glacier_idx])
                    glacier_ablation_rain = np.append(rain_sum_1[ablation_idx_1, glacier_idx], rain_sum[ablation_idx, glacier_idx])
                    
                    # We recompute the rain/snow limit with the new adjusted temperatures
                    glacier_accum_snow = np.where(dummy_glacier_accum_pos_temps > 2.0, 0.0, glacier_accum_snow)
                    glacier_accum_snow = np.where(((dummy_glacier_accumulation_temps < 2.0) & (glacier_accum_snow == 0.0)), glacier_accum_rain, glacier_accum_snow)
                    glacier_ablation_snow = np.where(dummy_glacier_ablation_pos_temps > 2.0, 0.0, glacier_ablation_snow)
                    glacier_ablation_snow = np.where(((dummy_glacier_ablation_temps < 2.0) & (glacier_ablation_snow == 0.0)), glacier_ablation_rain, glacier_ablation_snow)
                    
                    glacier_ablation_season = len(ablation_idx_1) + len(ablation_idx)
                    year_ablation_season = year_ablation_season + glacier_ablation_season
                    
                    glacier_CPDD = np.sum(glacier_year_pos_temps)
                    glacier_CPDDs_all[j]['CPDD'].append(year)
                    glacier_CPDDs_all[j]['CPDD'].append(glacier_CPDD)
                    year_CPDD = year_CPDD+glacier_CPDD
                    
                    glacier_year_accum_snow = np.sum(glacier_accum_snow) 
                    glacier_year_ablation_snow = np.sum(glacier_ablation_snow)
                    glacier_winter_snow_all[j]['snow'].append(year) 
                    glacier_winter_snow_all[j]['snow'].append(glacier_year_accum_snow)
                    year_w_snow = year_w_snow + glacier_year_accum_snow
                    glacier_summer_snow_all[j]['snow'].append(year)
                    glacier_summer_snow_all[j]['snow'].append(glacier_year_ablation_snow)
                    year_s_snow = year_s_snow + glacier_year_ablation_snow
                    
                    glacier_mon_temp_all[j]['years'].append(year)
                    glacier_mon_temp_all[j]['mon_temp'].append(mon_temp_year)
                    glacier_mon_snow_all[j]['mon_snow'].append(mon_snow_year)
                    
                    # Now we store the data for the sub-dataset of glaciers with SMB data
                    if(np.any(smb_glaciers == glacier_name)):
                        glacier_CPDDs_SMB[i]['CPDD'].append(year)
                        glacier_CPDDs_SMB[i]['CPDD'].append(glacier_CPDD)
                        year_CPDD_SMB = year_CPDD_SMB+glacier_CPDD
                        glacier_winter_snow_SMB[i]['snow'].append(year)
                        glacier_winter_snow_SMB[i]['snow'].append(glacier_year_accum_snow)
                        year_w_snow_SMB = year_w_snow_SMB + glacier_year_accum_snow
                        glacier_summer_snow_SMB[i]['snow'].append(year)
                        glacier_summer_snow_SMB[i]['snow'].append(glacier_year_ablation_snow)
                        year_s_snow_SMB = year_s_snow_SMB + glacier_year_ablation_snow
                        year_ablation_season_SMB = year_ablation_season_SMB + glacier_ablation_season
                        
                        glacier_mon_temp_SMB[i]['years'].append(year)
                        glacier_mon_temp_SMB[i]['mon_temp'].append(mon_temp_year)
                        glacier_mon_snow_SMB[i]['mon_snow'].append(mon_snow_year)
                        
                        
                    # If we reach the end of the time period, we compute the local anomalies and means
                    if(year == year_period[-1]):
                        glacier_CPDDs_all, glacier_winter_snow_all, glacier_summer_snow_all, CPDD_LocalAnomaly_all, raw_CPDD_LocalAnomaly_all, winter_snow_LocalAnomaly_all, winter_raw_snow_LocalAnomaly_all, summer_snow_LocalAnomaly_all, summer_raw_snow_LocalAnomaly_all = compute_local_anomalies(j, glacier_CPDDs_all, glacier_winter_snow_all, glacier_summer_snow_all,  
                                                                                                                                                                                                                                                                                                        local_CPDD_anomalies, raw_local_CPDD_anomalies,
                                                                                                                                                                                                                                                                                                        local_w_snow_anomalies, raw_local_w_snow_anomalies,
                                                                                                                                                                                                                                                                                                        local_s_snow_anomalies, raw_local_s_snow_anomalies)
                        
                        glacier_mon_temp_all, glacier_mon_snow_all, local_mon_temp_anomalies, local_mon_snow_anomalies = compute_monthly_anomalies(j, glacier_mon_temp_all, glacier_mon_snow_all, 
                                                                                                                                                   local_mon_temp_anomalies, local_mon_snow_anomalies, 
                                                                                                                                                   raw_local_mon_temp_anomalies, raw_local_mon_snow_anomalies)
                        
                        
                        
                        # Glaciers with SMB data
                        if(np.any(smb_glaciers == glacier_name)):
                            glacier_CPDDs_SMB, glacier_winter_snow_SMB, glacier_summer_snow_SMB, CPDD_SMB_LocalAnomaly, raw_CPDD_SMB_LocalAnomaly, winter_snow_SMB_LocalAnomaly, raw_winter_snow_SMB_LocalAnomaly, summer_snow_SMB_LocalAnomaly, raw_summer_snow_SMB_LocalAnomaly = compute_local_anomalies(i, glacier_CPDDs_SMB, glacier_winter_snow_SMB, glacier_summer_snow_SMB,  
                                                                                                                                                                                                                                                                                local_CPDD_anomalies_SMB, raw_local_CPDD_anomalies_SMB,
                                                                                                                                                                                                                                                                                local_w_snow_anomalies_SMB, raw_local_w_snow_anomalies_SMB,
                                                                                                                                                                                                                                                                                local_s_snow_anomalies_SMB, raw_local_s_snow_anomalies_SMB)
                            glacier_mon_temp_SMB, glacier_mon_snow_SMB, local_mon_temp_anomalies_SMB, local_mon_snow_anomalies_SMB = compute_monthly_anomalies(i, glacier_mon_temp_SMB, glacier_mon_snow_SMB, 
                                                                                                                                                               local_mon_temp_anomalies_SMB, local_mon_snow_anomalies_SMB, 
                                                                                                                                                               raw_local_mon_temp_anomalies_SMB, raw_local_mon_snow_anomalies_SMB)
                        
                        
                        ### End of glacier loop  ###
                    
                    
                    # We iterate the independent indexes
                    j = j+1
                    if(np.any(smb_glaciers == glacier_name)):
                        i = i+1
                        
                    ### End of years loop ###
                    
                raw_yearly_mean_CPDD.append(year_CPDD/all_glacier_coordinates.shape[0])
                raw_yearly_mean_CPDD_SMB.append(year_CPDD_SMB/glacier_SMB_coordinates.shape[0])
                raw_yearly_mean_winter_snow.append(year_w_snow/all_glacier_coordinates.shape[0])
                raw_yearly_mean_winter_snow_SMB.append(year_w_snow_SMB/glacier_SMB_coordinates.shape[0])
                raw_yearly_mean_summer_snow.append(year_s_snow/all_glacier_coordinates.shape[0])
                raw_yearly_mean_summer_snow_SMB.append(year_s_snow_SMB/glacier_SMB_coordinates.shape[0])
                raw_yearly_mean_ablation_season.append(year_ablation_season/all_glacier_coordinates.shape[0])
                raw_yearly_mean_ablation_season_SMB.append(year_ablation_season_SMB/glacier_SMB_coordinates.shape[0])
                
            # We combine the meteo SMB dataset forcings
            # Seasonal
            season_meteo_SMB = {'CPDD':np.asarray(glacier_CPDDs_SMB), 'winter_snow':np.asarray(glacier_winter_snow_SMB), 'summer_snow':np.asarray(glacier_summer_snow_SMB)}
            season_meteo_anomalies_SMB = {'CPDD':np.asarray(CPDD_SMB_LocalAnomaly), 'winter_snow':np.asarray(winter_snow_SMB_LocalAnomaly), 'summer_snow':np.asarray(summer_snow_SMB_LocalAnomaly)}
            season_raw_meteo_anomalies_SMB = {'CPDD':np.asarray(raw_CPDD_SMB_LocalAnomaly), 'winter_snow':np.asarray(raw_winter_snow_SMB_LocalAnomaly), 'summer_snow':np.asarray(raw_summer_snow_SMB_LocalAnomaly)}
            # Monthly
            monthly_meteo_SMB = {'temp':np.asarray(glacier_mon_temp_SMB), 'snow':np.asarray(glacier_mon_snow_SMB)}
            monthly_meteo_anomalies_SMB = {'temp':np.asarray(local_mon_temp_anomalies_SMB), 'snow':np.asarray(local_mon_snow_anomalies_SMB)}
            
            # We sort the SMB meteo forcings data to fit the order to the GLIMS Rabatel file                                                                                                                                                                                                      
            season_meteo_SMB, season_meteo_anomalies_SMB, season_raw_meteo_anomalies_SMB, monthly_meteo_SMB, monthly_meteo_anomalies_SMB = sort_SMB_rabatel_data(season_meteo_SMB, 
                                                                                                                                                                 season_meteo_anomalies_SMB, 
                                                                                                                                                                 season_raw_meteo_anomalies_SMB, 
                                                                                                                                                                 monthly_meteo_SMB, 
                                                                                                                                                                 monthly_meteo_anomalies_SMB,
                                                                                                                                                                 glims_rabatel)
            # We combine the full meteo dataset forcings for all glaciers
            # Seasonal
            season_meteo = {'CPDD':np.asarray(glacier_CPDDs_all), 'winter_snow':np.asarray(glacier_winter_snow_all), 'summer_snow':np.asarray(glacier_summer_snow_all)}
            season_meteo_anomalies = {'CPDD':np.asarray(CPDD_LocalAnomaly_all), 'winter_snow':np.asarray(winter_snow_LocalAnomaly_all), 'summer_snow':np.asarray(summer_snow_LocalAnomaly_all)}
            season_raw_meteo_anomalies = {'CPDD':np.asarray(raw_CPDD_LocalAnomaly_all), 'winter_snow':np.asarray(winter_raw_snow_LocalAnomaly_all), 'summer_snow':np.asarray(summer_raw_snow_LocalAnomaly_all)}
            # Monthly
            monthly_meteo = {'temp':np.asarray(glacier_mon_temp_all), 'snow':np.asarray(glacier_mon_snow_all)}
            monthly_meteo_anomalies = {'temp':np.asarray(local_mon_temp_anomalies), 'snow':np.asarray(local_mon_snow_anomalies)}
                                                                                                                                                               
            
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
                    
            # If the forcing folder is not created we create it
            if not os.path.exists(path_smb_function_safran):
                # We create a new folder in order to store the raster plots
                os.makedirs(path_smb_function_safran)
                    
            # We store the compacted seasonal and monthly meteo forcings
            # Glaciers with SMB data (machine learning model training)
            with open(path_smb_function_safran+'season_meteo_SMB.txt', 'wb') as season_f:
                        np.save(season_f, season_meteo_SMB)
            with open(path_smb_function_safran+'season_meteo_anomalies_SMB.txt', 'wb') as season_a_f:
                        np.save(season_a_f, season_meteo_anomalies_SMB)
            with open(path_smb_function_safran+'season_raw_meteo_anomalies_SMB.txt', 'wb') as season_raw_a_f:
                        np.save(season_raw_a_f, season_raw_meteo_anomalies_SMB)
            with open(path_smb_function_safran+'monthly_meteo_SMB.txt', 'wb') as mon_f:
                        np.save(mon_f, monthly_meteo_SMB)
            with open(path_smb_function_safran+'monthly_meteo_anomalies_SMB.txt', 'wb') as mon_a_f:
                        np.save(mon_a_f, monthly_meteo_anomalies_SMB)
            
            # All glaciers
            with open(path_smb_function_safran+'season_meteo.txt', 'wb') as season_f:
                np.save(season_f, season_meteo)
            with open(path_smb_function_safran+'season_meteo_anomalies.txt', 'wb') as season_a_f:
                        np.save(season_a_f, season_meteo_anomalies)
            with open(path_smb_function_safran+'season_raw_meteo_anomalies.txt', 'wb') as season_raw_a_f:
                        np.save(season_raw_a_f, season_raw_meteo_anomalies)
            with open(path_smb_function_safran+'monthly_meteo.txt', 'wb') as mon_f:
                        np.save(mon_f, monthly_meteo)
            with open(path_smb_function_safran+'monthly_meteo_anomalies.txt', 'wb') as mon_a_f:
                        np.save(mon_a_f, monthly_meteo_anomalies)
            
            # Ablation season length
            with open(path_smb_function_safran+'raw_yearly_mean_ablation_season.txt', 'wb') as rym_as_f:
                        np.save(rym_as_f, raw_yearly_mean_ablation_season_SMB)
    
    else:
         print("Skipping...")           
    ###  End of main function   ###