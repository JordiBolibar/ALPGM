#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

GLACIER SMB MODELLING AND PROJECTION BASED ON MULTIPLE REGRESSION FUNCTION

"""

## Dependencies: ##
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import settings
import os
import matplotlib
import subprocess
import shutil
import copy
import math
import random
from numba import jit
from netCDF4 import Dataset
import sys
import pandas as pd
from pathlib import Path
from glacier_evolution import store_file, get_aspect_deg, empty_folder, make_ensemble_simulation, preload_ensemble_SMB_models

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from keras import backend as K
#from keras import optimizers
from keras.models import load_model
#import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


######   FILE PATHS    #######

cv = 'LOGO\\'
    
# Folders     
#workspace = 'C:\\Jordi\\PhD\\Python\\'
workspace = Path(os.getcwd()).parent 
root = str(workspace.parent) + '\\'
workspace = str(workspace) + '\\'
path_smb = workspace + 'glacier_data\\smb\\'
path_smb_validations = path_smb + 'smb_simulations\\validation\\'
path_smb_function = path_smb + 'smb_function\\' + cv
path_smb_simulations = path_smb + 'smb_simulations\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 
path_glacier_coordinates = workspace + 'glacier_data\\glacier_coordinates\\' 
path_safran_forcings = 'C:\\Jordi\\PhD\\Data\\SAFRAN-Nivo-2017\\'
path_smb_function_safran = path_smb + 'smb_function\\SAFRAN\\'
path_smb_all_glaciers = path_smb + 'smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\'
path_smb_all_glaciers_smb = path_smb + 'smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\smb\\'
path_smb_all_glaciers_area = path_smb + 'smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\area\\'
path_smb_all_glaciers_slope = path_smb + 'smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\slope\\'
path_training_data = workspace + 'glacier_data\\glacier_evolution\\training\\'
path_training_glacier_info = path_training_data + 'glacier_info\\'
global path_slope20
path_slope20 = workspace + 'glacier_data\\glacier_evolution\\glacier_slope20\\SAFRAN\\1\\'

######     FUNCTIONS     ######

def label_line(ax, line, label, color='0.5', fs=14, halign='left'):
    """Add an annotation to the given line with appropriate placement and rotation.
    Based on code from:
        [How to rotate matplotlib annotation to match a line?]
        (http://stackoverflow.com/a/18800233/230468)
        User: [Adam](http://stackoverflow.com/users/321772/adam)
    Arguments
    ---------
    ax : `matplotlib.axes.Axes` object
        Axes on which the label should be added.
    line : `matplotlib.lines.Line2D` object
        Line which is being labeled.
    label : str
        Text which should be drawn as the label.
    ...
    Returns
    -------
    text : `matplotlib.text.Text` object
    """
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
#    y1 = ydata[0]
#    y2 = ydata[-1]

    if halign.startswith('l'):
        xx = x1
        halign = 'left'
    elif halign.startswith('r'):
        xx = x2
        halign = 'right'
    elif halign.startswith('c'):
        xx = 0.5*(x1 + x2)
        halign = 'center'
    else:
        raise ValueError("Unrecogznied `halign` = '{}'.".format(halign))

    yy = np.interp(xx, xdata, ydata)

    ylim = ax.get_ylim()
    # xytext = (10, 10)
    xytext = (25, 0)
    text = ax.annotate(label, xy=(xx, yy), xytext=xytext, textcoords='offset pixels',
                       size=fs, color=color, zorder=1,
                       horizontalalignment=halign, verticalalignment='center_baseline')

    ax.set_ylim(ylim)
    return text

def store_smb_data(file_name, data):
    try:
        np.savetxt(file_name, data, delimiter=";", fmt="%.7f")
    except IOError:
        print("File currently opened. Please close it to proceed.")
        os.system('pause')
        # We try again
        try:
            print("\nRetrying storing " + str(file_name))
            np.savetxt(file_name, data, delimiter=";", fmt="%.7f")
        except IOError:
            print("File still not available. Aborting simulations.")

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def similar(a, b):
    ratios = []
    for glacier_name in a:
        ratios.append(SequenceMatcher(None, glacier_name, b).ratio())
    ratios = np.asarray(ratios)
    return ratios.max(), ratios.argmax()

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def store_glacier_info(glacier_info, path):
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + 'glacier_info_' + str(glacier_info['glimsID'])
    with open(path, 'wb') as glacier_info_f:
        np.save(glacier_info_f, glacier_info)
#        print("File saved: " + str(path))

def interpolate_glims_variable(variable_name, glims_glacier, glims_2003, glims_1985):
    var_2015 = glims_glacier[variable_name]
    # In case there are multiple results, we choose the one with the most similar area
    idx_2003 = np.where(glims_2003['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
    if(idx_2003.size > 1):
        idx_aux = find_nearest(glims_2003[idx_2003]['Area'], glims_glacier['Area'])
        idx_2003 = idx_2003[idx_aux]
    idx_1985 = np.where(glims_1985['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
    if(idx_1985.size > 1):
        idx_aux = find_nearest(glims_1985[idx_2003]['Area'], glims_glacier['Area'])
        idx_1985 = idx_1985[idx_aux]
        
    var_2003 = glims_2003[idx_2003][variable_name]
    var_1985 = glims_1985[idx_1985][variable_name]
    interp_1984_2003 = np.linspace(var_1985, var_2003, num=(2003-1984))
    interp_2003_2014 = np.linspace(var_2003, var_2015, num=(2014-2003)+1)
    interp_1984_2014 = np.append(interp_1984_2003, interp_2003_2014)
    
    return interp_1984_2014

def interpolate_extended_glims_variable(variable_name, glims_glacier, glims_2015, glims_1985, glims_1967):
    # In case there are multiple results, we choose the one with the most similar area
    
    # 2015 glacier inventory
    idx_2015 = np.where(glims_2015['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
    # Decision making in case of glacier split with the same GLIMS ID
    if(idx_2015.size > 1):
        # Special case for Rouies / Chardon glacier
        if(glims_2015['GLIMS_ID'] == 'G006275E44878N'):
            if(glims_glacier['Glacier'] == 'des Rouies'):
                idx_2015 = np.where(glims_2015['Glacier'] == 'des Rouies')
            else:
                idx_2015 = np.where(glims_2015['Glacier'] == 'du Chardon_1') 
        else:
            idx_area_aux = find_nearest(glims_2015[idx_2015]['Area'], glims_glacier['Area'])
            idx_zmean_aux = find_nearest(glims_2015[idx_2015]['MEAN_Pixel'], glims_glacier['MEAN_Pixel'])
            # Choose most similar glacier based on surface area or mean altitude
            if(idx_area_aux == idx_zmean_aux):
                idx_2015 = idx_2015[idx_zmean_aux]
            else:
                if((glims_glacier['Area'] - glims_2015['Area'][idx_2015[idx_zmean_aux]]) > glims_glacier['Area']*0.5):
                    idx_2015 = idx_2015[idx_zmean_aux]
                else:
                    idx_2015 = idx_2015[idx_area_aux]
    # 1985 glacier inventory
    idx_1985 = np.where(glims_1985['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
    if(idx_1985.size > 1):
        idx_aux = find_nearest(glims_1985[idx_1985]['Area'], glims_glacier['Area'])
        idx_1985 = idx_1985[idx_aux]
        
    var_2003 = glims_glacier[variable_name]
    var_2015 = glims_2015[idx_2015][variable_name]
    var_1985 = glims_1985[idx_1985][variable_name]
    var_1967 = glims_1985[idx_1985][variable_name]
    
#    if(variable_name == 'Area'):
#        var_1967 = var_1967/1000000  # from m2 to km2 (1967)
#    import pdb; pdb.set_trace()
    
    if(var_2015.size == 0): # If there's no data in 2015, we fill with NaNs the 2003-2015 period
        interp_1967_1984 = np.linspace(var_1967, var_1985, num=(1984-1967))
        interp_1984_2003 = np.linspace(var_1985, var_2003, num=(2003-1984)+1)
#        interp_2003_2015 = np.empty((2015-2003)+1)
#        interp_2003_2015[:] = np.nan
        interp_1967_2003 = np.append(interp_1967_1984, interp_1984_2003)
#        interp_1967_2015 = np.append(interp_1967_2003, interp_2003_2015)
        interp_data = interp_1967_2003
    elif(not math.isnan(var_2015)):
        interp_1967_1984 = np.linspace(var_1967, var_1985, num=(1984-1967))
        interp_1984_2003 = np.linspace(var_1985, var_2003, num=(2003-1984))
        interp_2003_2015 = np.linspace(var_2003, var_2015, num=(2015-2003)+1)
        interp_1967_2003 = np.append(interp_1967_1984, interp_1984_2003)
        interp_1967_2015 = np.append(interp_1967_2003, interp_2003_2015)
        interp_data = interp_1967_2015
    else:
        interp_data = [var_1985]
        
    return np.asarray(interp_data)

# We retrieve the slope from a glacier from the middle year of the simulation (2003)
def get_slope20(glims_glacier):
    glacier_name = glims_glacier['Glacier'].decode('ascii')
    glimsID = glims_glacier['GLIMS_ID'].decode('ascii')
    alt_max = glims_glacier['MAX_Pixel']
    alt_min = glims_glacier['MIN_Pixel']
    length = glims_glacier['Length']
    # Load the file
    if(str(glacier_name[-1]).isdigit() and str(glacier_name[-1]) != '1'):
        appendix = str(glacier_name[-1])
        if(os.path.isfile(path_slope20 + glimsID + '_' + appendix + '_slope20.csv')):
            slope20_array = genfromtxt(path_slope20 + glimsID + '_' + appendix + '_slope20.csv', delimiter=';', dtype=np.dtype('float32'))
        elif(os.path.isfile(path_slope20 + glimsID + '_slope20.csv')):
            slope20_array = genfromtxt(path_slope20 + glimsID + '_slope20.csv', delimiter=';', dtype=np.dtype('float32'))
        else:
            slope20_array = np.array([])
    elif(os.path.isfile(path_slope20 + glimsID + '_slope20.csv')):
        slope20_array = genfromtxt(path_slope20 + glimsID + '_slope20.csv', delimiter=';', dtype=np.dtype('float32'))
    else:
        slope20_array = np.array([])
    ### Retrieve the 20% slope
    if(slope20_array.size == 0 or slope20_array.size == 2):
        slope20 = np.rad2deg(np.arcsin((alt_max - alt_min)/length))
#        print("Manual slope: " + str(slope20))
    elif(slope20_array[:,1].mean() > 55):
        slope20 = np.rad2deg(np.arcsin((alt_max - alt_min)/length))
#        print("Manual slope: " + str(np.rad2deg(np.arctan((alt_max - alt_min)/length))))
    else:
        slope20 = slope20_array[:,1].mean()
#        print("Retrieved slope: " + str(slope20_array[:,1].mean()))
#    
    if(slope20 > 50 or np.isnan(slope20)):
        # Limit slope with random variance to avoid unrealistic slopes
        slope20 = random.uniform(40.0, 55.0)
    
    print("Glacier tongue slope: " + str(slope20))
    
    return slope20

def create_spatiotemporal_matrix(season_anomalies, mon_anomalies, glims_glacier, glacier_mean_altitude, glacier_area, best_models):
    x_reg_array = []
    
    max_alt = glims_glacier['MAX_Pixel']
    slope20 = get_slope20(glims_glacier)
#    slope20 = glims_glacier['slope20']
#    slope20 = glims_glacier['slope20_evo']
    lon = glims_glacier['x_coord']
    lat = glims_glacier['y_coord']
    aspect = np.cos(get_aspect_deg(glims_glacier['Aspect'].decode('ascii')))
    
#    print('max_alt: ' + str(max_alt))
#    print('slope20: ' + str(slope20))
#    print("glacier_area: " + str(glacier_area))
#    print("glacier_mean_altitude: " + str(glacier_mean_altitude))
##    
#    print("season_anomalies: " + str(season_anomalies))
#    print("mon_anomalies: " + str(mon_anomalies))
    
    count, n_model = 0,0
    for model in best_models:
        x_reg, x_reg_nn, x_reg_full = [],[],[]
        x_reg_idxs = eval(model['f1'])
        for cpdd_y, w_snow_y, s_snow_y,  mon_temp_y, mon_snow_y, mean_alt_y, area_y in zip(season_anomalies['CPDD'], season_anomalies['w_snow'], season_anomalies['s_snow'], mon_anomalies['temp'], mon_anomalies['snow'], glacier_mean_altitude, glacier_area):
            
            # We get the current iteration combination
            input_variables_array = np.array([cpdd_y, w_snow_y, s_snow_y, mean_alt_y, max_alt, slope20, area_y, lon, lat, aspect, mean_alt_y*cpdd_y, slope20*cpdd_y, max_alt*cpdd_y, area_y*cpdd_y, lat*cpdd_y, lon*cpdd_y, aspect*cpdd_y, mean_alt_y*w_snow_y, slope20*w_snow_y, max_alt*w_snow_y, area_y*w_snow_y, lat*w_snow_y, lon*w_snow_y, aspect*w_snow_y, mean_alt_y*s_snow_y, slope20*s_snow_y, max_alt*s_snow_y, area_y*s_snow_y, lat*s_snow_y, lon*s_snow_y, aspect*s_snow_y])
            x_reg.append(input_variables_array[x_reg_idxs])
            input_full_array = np.append(input_variables_array, mon_temp_y)
            input_full_array = np.append(input_full_array, mon_snow_y)
            x_reg_full.append(input_full_array)
            
            # We create a separate smaller matrix for the Artificial Neural Network algorithm
            input_features_nn_array = np.array([cpdd_y, w_snow_y, s_snow_y, mean_alt_y, max_alt, slope20, area_y, lon, lat, aspect])
            input_features_nn_array = np.append(input_features_nn_array, mon_temp_y)
            input_features_nn_array = np.append(input_features_nn_array, mon_snow_y)
            x_reg_nn.append(input_features_nn_array)
            
        count = count+1
            
        # Add constant
        x_reg = np.asarray(x_reg)
        x_reg = np.expand_dims(x_reg, axis=0)
        if(n_model == 0):
            x_reg_array = x_reg
        else:
            x_reg_array = np.append(x_reg_array, x_reg, axis=0)
        x_reg_full = np.asarray(x_reg_full)
        n_model = n_model+1
    
#    x_reg_array = np.asarray(x_reg_array)
    x_reg_nn = np.asarray(x_reg_nn)
    
    return x_reg_array, x_reg_full, x_reg_nn

########################   SAFRAN CLIMATIC FORCINGS    ####################################################

def find_glacier_idx(glacier_massif, massif_number, altitudes, glacier_altitude, aspects, glacier_aspect):
    # We extract the glacier index combining different SAFRAN data matrix
    massif_altitudes_idx = np.where(massif_number == float(glacier_massif))[0]
    glacier_aspect_idx = np.where(aspects == float(glacier_aspect))[0]
    massif_alt_aspect_idx = np.array(list(set(massif_altitudes_idx).intersection(glacier_aspect_idx)))
    index_alt = find_nearest(altitudes[massif_alt_aspect_idx], glacier_altitude)
    final_idx = int(massif_alt_aspect_idx[index_alt])
    
    return final_idx

@jit
def find_glacier_coordinates(massif_number, zs, aspects, glims_data):
    glacier_centroid_altitude = glims_data['MEAN_Pixel']
    GLIMS_IDs = glims_data['GLIMS_ID']
    glacier_massifs = glims_data['Massif_SAFRAN']
    glacier_names = glims_data['Glacier']
    glacier_aspects = glims_data['Aspect_num']
    all_glacier_coordinates = []
    
    # All glaciers loop
    for glims_id, glacier_name, glacier_massif, glacier_altitude, glacier_aspect in zip(GLIMS_IDs, glacier_names, glacier_massifs, glacier_centroid_altitude, glacier_aspects):
        all_glacier_coordinates.append([glacier_name, find_glacier_idx(glacier_massif, massif_number, zs, glacier_altitude, aspects, glacier_aspect), float(glacier_altitude), glims_id, int(glacier_massif)])
        
    return np.asarray(all_glacier_coordinates)

def get_SAFRAN_glacier_coordinates(glims_dataset):
     # We read the first year to get some basic information
    dummy_SAFRAN_forcing = Dataset(path_safran_forcings + '84\\FORCING.nc')
    
    aspects = dummy_SAFRAN_forcing.variables['aspect'][:]
    zs = dummy_SAFRAN_forcing.variables['ZS'][:]
    massif_number = dummy_SAFRAN_forcing.variables['massif_number'][:]
    
    all_glacier_coordinates = find_glacier_coordinates(massif_number, zs, aspects, glims_dataset)
    
    return all_glacier_coordinates

@jit
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

def compute_local_anomalies(glacier_CPDD, glacier_winter_snow, glacier_summer_snow,
                            CPDD_ref, w_snow_ref, s_snow_ref):
    
    local_CPDD_anomaly = glacier_CPDD - CPDD_ref
    local_w_snow_anomaly = glacier_winter_snow - w_snow_ref
    local_s_snow_anomaly = glacier_summer_snow - s_snow_ref
    
    return local_CPDD_anomaly, local_w_snow_anomaly, local_s_snow_anomaly

def compute_monthly_anomalies(mon_temps, mon_snow, mon_temp_ref, mon_snow_ref):
    
    mon_temp_anomaly = mon_temps - mon_temp_ref
    mon_snow_anomaly = mon_snow - mon_snow_ref
    
    return mon_temp_anomaly, mon_snow_anomaly

def get_default_SAFRAN_forcings(year_start, year_end):
    
    path_temps = path_smb_function_safran +'daily_temps_years_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_snow = path_smb_function_safran +'daily_snow_years_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_rain = path_smb_function_safran +'daily_rain_years_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_dates = path_smb_function_safran +'daily_dates_years_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_zs = path_smb_function_safran +'zs_years' + str(year_start) + '-' + str(year_end) + '.txt'
    if(os.path.exists(path_temps) & os.path.exists(path_snow) & os.path.exists(path_rain) & os.path.exists(path_dates) & os.path.exists(path_zs)):
        print("Fetching SAFRAN forcings...")
        print("\nTaking forcings from " + str(path_temps))
        
        with open(path_temps, 'rb') as temps_f:
            daily_temps_years = np.load(temps_f, encoding='latin1')
        with open(path_snow, 'rb') as snow_f:
            daily_snow_years = np.load(snow_f, encoding='latin1')
        with open(path_rain, 'rb') as rain_f:
            daily_rain_years = np.load(rain_f, encoding='latin1')
        with open(path_dates, 'rb') as dates_f:
            daily_dates_years = np.load(dates_f, encoding='latin1')
        with open(path_zs, 'rb') as zs_f:
            zs_years = np.load(zs_f, encoding='latin1')
            
    else:
        sys.exit("\n[ ERROR ] SAFRAN base forcing files not found. Please execute SAFRAN forcing module before")
        
        
    return daily_temps_years, daily_snow_years, daily_rain_years, daily_dates_years, zs_years

def get_adjusted_glacier_SAFRAN_forcings(year, year_start, glacier_mean_altitude, SAFRAN_idx, daily_temps_years, daily_snow_years, daily_rain_years, daily_dates_years, zs_years, CPDD_ref, w_snow_ref, s_snow_ref, mon_temp_ref, mon_snow_ref):
    # We also need to fetch the previous year since data goes from 1st of August to 31st of July
    idx = year - (year_start-1)
    glacier_idx = int(SAFRAN_idx)
    
    zs = zs_years[idx-1]
    
    daily_temps_mean_1 = copy.deepcopy(daily_temps_years[idx-1]) + ((zs[glacier_idx] - glacier_mean_altitude)/1000.0)*6.0
    daily_temps_mean = copy.deepcopy(daily_temps_years[idx]) + ((zs[glacier_idx] - glacier_mean_altitude)/1000.0)*6.0
    
    snow_sum_1 = copy.deepcopy(daily_snow_years[idx-1])
    snow_sum = copy.deepcopy(daily_snow_years[idx])
    
    rain_sum_1 = copy.deepcopy(daily_rain_years[idx-1])
    rain_sum = copy.deepcopy(daily_rain_years[idx])
    
    # We get the daily datetimes
    daily_datetimes_1 = copy.deepcopy(daily_dates_years[idx-1])
    daily_datetimes = copy.deepcopy(daily_dates_years[idx])
    
    #### We compute the monthly average data
    # Monthly average temperature
    glacier_temps_1 = daily_temps_mean_1[:, glacier_idx] 
    glacier_temps = daily_temps_mean[:, glacier_idx]
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
    temp_year = np.append(daily_temps_mean_1[-year_1_offset:, glacier_idx], daily_temps_mean[:year_offset, glacier_idx]) 
    pos_temp_year = np.where(temp_year < 0, 0, temp_year)
    integ_temp = np.cumsum(pos_temp_year)
    
    start_div, end_div = 30, 10
    
    ### Dynamic temperature ablation period
    start_y_ablation = np.where(integ_temp > integ_temp.max()/start_div)[0]
#            start_y_ablation = np.where(integ_temp > 30)[0]
    end_y_ablation = np.where(integ_temp > (integ_temp.max() - integ_temp.max()/end_div))[0]
    
    start_ablation = start_y_ablation[0] + (daily_temps_mean_1[:,glacier_idx].size - year_1_offset)
    end_ablation = end_y_ablation[0] - year_1_offset
    if(start_ablation > 366):
        start_ablation = 366
        
    # We get the dynamic indexes for the ablation and accumulation periods for temperature
    ablation_temp_idx_1 = range(start_ablation, daily_datetimes_1.size)
    ablation_temp_idx = range(0, end_ablation)
#    accum_temp_idx_1 = range(end_ablation+1, start_ablation-1)
    
    # We get the indexes for the ablation and accumulation periods for snowfall
    # Classic 120-270 ablation period
    ablation_idx_1 = range(daily_datetimes_1.size-95, daily_datetimes_1.size)
    ablation_idx = range(0, 55)
    accum_idx_1 = range(56, daily_datetimes_1.size-96)
    
    glacier_ablation_temps = np.append(daily_temps_mean_1[ablation_temp_idx_1, glacier_idx], daily_temps_mean[ablation_temp_idx, glacier_idx]) 
#    glacier_accumulation_temps = daily_temps_mean_1[accum_temp_idx_1, glacier_idx] 
    dummy_glacier_ablation_temps = np.append(daily_temps_mean_1[ablation_idx_1, glacier_idx], daily_temps_mean[ablation_idx, glacier_idx]) 
    dummy_glacier_accumulation_temps = daily_temps_mean_1[accum_idx_1, glacier_idx] 
    
#    glacier_ablation_temps = glacier_ablation_temps + ((zs[glacier_idx] - glacier_mean_altitude)/1000.0)*6.0
#    glacier_accumulation_temps = glacier_accumulation_temps + ((zs[glacier_idx] - glacier_mean_altitude)/1000.0)*6.0
                    
    glacier_year_pos_temps = np.where(glacier_ablation_temps < 0, 0, glacier_ablation_temps)
#    glacier_accum_pos_temps = np.where(glacier_accumulation_temps < 0, 0, glacier_accumulation_temps)
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
    
    # We compute the cumulative yearly CPDD and snowfall
    glacier_CPDD = np.sum(glacier_year_pos_temps)
    glacier_winter_snow = np.sum(glacier_accum_snow) 
    glacier_summer_snow = np.sum(glacier_ablation_snow)
    
    CPDD_LocalAnomaly, winter_snow_LocalAnomaly, summer_snow_LocalAnomaly = compute_local_anomalies(glacier_CPDD, glacier_winter_snow, glacier_summer_snow,
                                                                                                    CPDD_ref, w_snow_ref, s_snow_ref)

    mon_temp_anomaly, mon_snow_anomaly = compute_monthly_anomalies(mon_temp_year, mon_snow_year, mon_temp_ref, mon_snow_ref) 
    
    
    return CPDD_LocalAnomaly, winter_snow_LocalAnomaly, summer_snow_LocalAnomaly,  mon_temp_anomaly, mon_snow_anomaly

def get_meteo_references(season_meteo_SMB, monthly_meteo_SMB, glimsID, glacierName):
    found = False
    glacier_CPDDs = season_meteo_SMB['CPDD']
    glacier_winter_snow = season_meteo_SMB['winter_snow']
    glacier_summer_snow = season_meteo_SMB['summer_snow']
    glacier_mon_temps = monthly_meteo_SMB['temp']
    glacier_mon_snow = monthly_meteo_SMB['snow']
    
    for cpdd, w_snow, s_snow, mon_temps, mon_snow in zip(glacier_CPDDs, glacier_winter_snow, glacier_summer_snow, glacier_mon_temps, glacier_mon_snow):
        if(cpdd['GLIMS_ID'] == glimsID):
            if((found and cpdd['Glacier'] == glacierName) or not found):
                
                CPDD_ref = cpdd['Mean']
                w_snow_ref = w_snow['Mean']
                s_snow_ref = s_snow['Mean'] 
                
                mon_temp_ref = mon_temps['mon_means']
                mon_snow_ref = mon_snow['mon_means']
                
                found = True
    return CPDD_ref, w_snow_ref, s_snow_ref, mon_temp_ref, mon_snow_ref

def get_monthly_temps(daily_data, daily_datetimes):
    
    d = {'Dates': daily_datetimes, 'Temps': daily_data}
    df_datetimes = pd.DataFrame(data=d)
    df_datetimes.set_index('Dates', inplace=True)
    df_datetimes.index = pd.to_datetime(df_datetimes.index)
    df_datetimes = df_datetimes.resample('M').mean()
    
    monthly_avg_data = df_datetimes.Temps.to_numpy()
    
    return monthly_avg_data[:12]
    
def get_monthly_snow(daily_data, daily_datetimes):
    d = {'Dates': daily_datetimes, 'Temps': daily_data}
    df_datetimes = pd.DataFrame(data=d)
    df_datetimes.set_index('Dates', inplace=True)
    df_datetimes.index = pd.to_datetime(df_datetimes.index)
    df_datetimes = df_datetimes.resample('M').sum()
    
    monthly_avg_data = df_datetimes.Temps.to_numpy()
    
    return monthly_avg_data[:12]

def compute_SAFRAN_anomalies(glacier_info, year_range, year_start, all_glacier_coordinates, season_meteo, monthly_meteo, daily_meteo_data):
    
    # We get the glacier indexes
    SAFRAN_idx = all_glacier_coordinates[np.where(all_glacier_coordinates[:,3] == glacier_info['glimsID'])[0]][0][1]
    # We extract the meteo references for the simulation period
    CPDD_ref, w_snow_ref, s_snow_ref, mon_temp_ref, mon_snow_ref = get_meteo_references(season_meteo[()], monthly_meteo[()], glacier_info['glimsID'], glacier_info['name'])
    
    CPDD_LocalAnomaly, winter_snow_LocalAnomaly, summer_snow_LocalAnomaly, mon_temp_anomaly, mon_snow_anomaly = [],[],[],[],[]
    # We compute the meteo anomalies year by year to reproduce the main flow of the model
    for year, z_mean in zip(year_range, glacier_info['mean_altitude']):
        CPDD_LocalAnomaly_y, winter_snow_LocalAnomaly_y, summer_snow_LocalAnomaly_y,  mon_temp_anomaly_y, mon_snow_anomaly_y = get_adjusted_glacier_SAFRAN_forcings(year, year_start, z_mean, SAFRAN_idx, 
                                                                                                                             daily_meteo_data['temps'], daily_meteo_data['snow'], daily_meteo_data['rain'], daily_meteo_data['dates'], 
                                                                                                                             daily_meteo_data['zs'], CPDD_ref, w_snow_ref, s_snow_ref, mon_temp_ref, mon_snow_ref)
        CPDD_LocalAnomaly.append(CPDD_LocalAnomaly_y)
        winter_snow_LocalAnomaly.append(winter_snow_LocalAnomaly_y)
        summer_snow_LocalAnomaly.append(summer_snow_LocalAnomaly_y)
        mon_temp_anomaly.append(mon_temp_anomaly_y)
        mon_snow_anomaly.append(mon_snow_anomaly_y)
        
    CPDD_LocalAnomaly = np.asarray(CPDD_LocalAnomaly)
    winter_snow_LocalAnomaly = np.asarray(winter_snow_LocalAnomaly)
    summer_snow_LocalAnomaly = np.asarray(summer_snow_LocalAnomaly)
    mon_temp_anomaly = np.asarray(mon_temp_anomaly)
    mon_snow_anomaly = np.asarray(mon_snow_anomaly)
    
    season_anomalies = {'CPDD': CPDD_LocalAnomaly, 'w_snow':winter_snow_LocalAnomaly, 's_snow':summer_snow_LocalAnomaly}
    mon_anomalies = {'temp':mon_temp_anomaly, 'snow':mon_snow_anomaly}
    
    return season_anomalies, mon_anomalies


def main(compute, reconstruct):

    ################################################################################
    ##################		                MAIN               	#####################
    ################################################################################  
     
    print("\n-----------------------------------------------")
    print("               SMB VALIDATION ")
    print("-----------------------------------------------\n")
    
    if(compute):
        
        if(reconstruct):
            path_ann = path_smb + 'ANN\\LOGO\\no_weights\\'
            path_cv_ann = path_ann + 'CV\\'
        else:
            # Set LOGO for model validation
            if(settings.smb_model_type == 'ann_no_weights'):
                path_ann = path_smb + 'ANN\\LOGO\\no_weights\\'
                path_cv_ann = path_ann + 'CV\\'
            elif(settings.smb_model_type == 'ann_weights'):
                path_ann = path_smb + 'ANN\\LOGO\\weights\\'
                path_cv_ann = path_ann + 'CV\\'
        
        # Close all previous open plots in the workflow
        plt.close('all')
            
        # We read the SMB from the csv file
        SMB_raw = genfromtxt(path_smb + 'SMB_raw_extended.csv', delimiter=';', dtype=float)
        SMB_uncertainties = genfromtxt(path_glacier_coordinates + 'glaciers_rabatel_uncertainties.csv', delimiter=';', dtype=float)
        
        ### We detect the forcing between SPAZM, SAFRAN or ADAMONT
        if(settings.simulation_type == "future"):
            forcing = settings.projection_forcing
        else:
            forcing = settings.historical_forcing
            
        # We determine the path depending on the forcing
        path_smb_function_forcing = path_smb + 'smb_function\\' + forcing + "\\"
        
        #### GLIMS data for 1985, 2003 and 2015
        glims_2015 = genfromtxt(path_glims + 'GLIMS_2015.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50')])
        glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8'), ('ID', '<i8')])
        glims_1985 = genfromtxt(path_glims + 'GLIMS_1985.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50')])
        glims_1967 = genfromtxt(path_glims + 'GLIMS_1967-71.csv', delimiter=';', skip_header=1,  dtype=[('Glacier', '<a50'), ('Latitude', '<f8'), ('Longitude', '<f8'), ('Massif', '<a50'),  ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('Year', '<f8'), ('Perimeter', '<f8'), ('Area', '<f8'),  ('Code_WGI', '<a50'), ('Length', '<f8'), ('MEAN_Pixel', '<f8'), ('Slope', '<f8'), ('Aspect', '<a50'), ('Code', '<a50'), ('BV', '<a50'), ('GLIMS_ID', '<a50')])

        
        ####  GLIMS data for the 30 glaciers with remote sensing SMB data (Rabatel et al. 2016)   ####
        glims_rabatel = genfromtxt(path_glims + 'GLIMS_Rabatel_30_2015.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('slope20', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8'), ('slope20_evo', '<f8')])        
        
        best_models = genfromtxt(path_smb + 'chosen_models_3_5.csv', delimiter=';', skip_header=1, dtype=None) 
        
        # We open all the files with the data to be modelled
        
        # Use latin1 encoding to avoid incompatibities between Python 2 and 3
        with open(path_smb+'smb_function\\full_scaler_spatial.txt', 'rb') as full_scaler_f:
            full_scaler = np.load(full_scaler_f)[()]                 
        with open(path_smb_function+'lasso_logo_models.txt', 'rb') as lasso_logos_f:
            lasso_logo_models = np.load(lasso_logos_f)
            
        # We load the climate forcings
        # SMB glaciers (Rabatel et al. 2016)
        # We load the compacted seasonal and monthly meteo forcings
        with open(path_smb_function_forcing+'season_meteo_SMB.txt', 'rb') as season_f:
            season_meteo_SMB = np.load(season_f)
        with open(path_smb_function_forcing+'season_meteo_anomalies_SMB.txt', 'rb') as season_a_f:
            season_meteo_anomalies_SMB = np.load(season_a_f)
#        with open(path_smb_function_forcing+'season_raw_meteo_anomalies_SMB.txt', 'rb') as season_raw_a_f:
#            season_raw_meteo_anomalies_SMB = np.load(season_raw_a_f)
        with open(path_smb_function_forcing+'monthly_meteo_SMB.txt', 'rb') as mon_f:
            monthly_meteo_SMB = np.load(mon_f)
#        with open(path_smb_function_forcing+'monthly_meteo_anomalies_SMB.txt', 'rb') as mon_a_f:
#            monthly_meteo_anomalies_SMB = np.load(mon_a_f)
            
        # Meteo data for all the glaciers
        with open(path_smb_function_forcing+'season_meteo.txt', 'rb') as season_f:
            season_meteo = np.load(season_f)
        with open(path_smb_function_forcing+'monthly_meteo.txt', 'rb') as mon_f:
            monthly_meteo = np.load(mon_f)
            
        # We retrieve all the SAFRAN glacier coordinates
        with open(path_smb_function_forcing+'all_glacier_coordinates.txt', 'rb') as coords_f:
            all_glacier_coordinates = np.load(coords_f)
            
        # CV model RMSE to compute the dynamic ensemble weights
        with open(settings.path_ann +'RMSE_per_fold.txt', 'rb') as rmse_f:
            CV_RMSE = np.load(rmse_f)
            
        # We extract the simulation interval in years
        # [()] needed to access the dictionary
        year_start = int(season_meteo_anomalies_SMB[()]['CPDD'][0]['CPDD'][0])
        year_end = int(season_meteo_anomalies_SMB[()]['CPDD'][0]['CPDD'][-2])
        
        if(forcing == "SAFRAN"):
            if(reconstruct):
                start_ref = 1967
                end_ref = 2015
            else:
                start_ref = 1984
                end_ref = 2014
        
#        # Reference period for climate forcings
        year_range = range(start_ref, end_ref +1)
        
        # We use only the 19 best models
        best_models = best_models[:19]
        
        if(forcing == "SAFRAN"):
            print("Getting all the SAFRAN forcing data....")
#            all_glacier_coordinates = get_SAFRAN_glacier_coordinates(glims_rabatel)
            daily_temps_years, daily_snow_years, daily_rain_years, daily_dates_years, zs_years = get_default_SAFRAN_forcings(year_start, year_end)
            daily_meteo_data = {'temps': daily_temps_years, 'snow':daily_snow_years, 'rain':daily_rain_years, 'dates':daily_dates_years, 'zs':zs_years}
            
            if(reconstruct):
                ########################################################
                ####  SMB simulation for all the French alpine glaciers
                print("\nNow we simulate the glacier-wide SMB for all the French alpine glaciers")
                glacier_idx = 0
#                # We retrie the ANN model
#                ann_model = load_model(path_ann + 'ann_glacier_model.h5', custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error})
                
                # We remove all previous simulations from the folder
                if(os.path.exists(path_smb_all_glaciers)):
                    shutil.rmtree(path_smb_all_glaciers)
                    
                cumulative_smb_glaciers = []
                
                # We preload the ensemble SMB models to speed up the simulations
                ensemble_SMB_models = preload_ensemble_SMB_models()
                
                for glims_glacier in glims_2003:
    #                if(glacier_name == "d'Argentiere"):
#                    if(glacier_idx == 421):
                    if(True):
                        glimsID = glims_glacier['GLIMS_ID'].decode('ascii')
#                    if(glimsID == 'G006934E45883N' or glimsID == 'G006985E45951N'):
                        glacier_name = glims_glacier['Glacier'].decode('ascii')
                        massif_safran = glims_glacier['Massif_SAFRAN']
#                        print("\nSimulating Glacier: " + str(glacier_name))
                        print("\n# " + str(glacier_idx))
    #                    print("Glacier GLIMS ID: " + str(glimsID))
                        
                        glacier_mean_altitude = interpolate_extended_glims_variable('MEAN_Pixel', glims_glacier, glims_2015, glims_1985, glims_1967)
                        glacier_max_altitude = interpolate_extended_glims_variable('MAX_Pixel', glims_glacier, glims_2015, glims_1985, glims_1967)
                        glacier_area = interpolate_extended_glims_variable('Area', glims_glacier, glims_2015, glims_1985, glims_1967)
                        
                        glacier_info = {'name':glacier_name, 'glimsID':glimsID, 'mean_altitude':glacier_mean_altitude, 'max_altitude':glacier_max_altitude, 'area': glacier_area, 'massif_SAFRAN': int(massif_safran), 'ID': glims_glacier['ID']}
                        
                        ####  SAFRAN FORCINGS  ####
                        # We recompute the SAFRAN forcings based on the current topographical data
                        season_anomalies, mon_anomalies = compute_SAFRAN_anomalies(glacier_info, year_range, year_start, all_glacier_coordinates, season_meteo, monthly_meteo, daily_meteo_data)
                        
                        # We create the spatiotemporal matrix to train the machine learning algorithm
                        x_reg_array, x_reg_full, x_reg_nn = create_spatiotemporal_matrix(season_anomalies, mon_anomalies, glims_glacier, glacier_mean_altitude, glacier_area, best_models)
                        
                        #####  Machine learning SMB simulations   ###################
                        SMB_nn, SMB_ensemble = make_ensemble_simulation(ensemble_SMB_models, CV_RMSE, x_reg_nn, 34, glimsID, glims_rabatel, evolution=False)
#                        SMB_nn = ann_model.predict(x_reg_nn, batch_size = 34)
#                        SMB_nn = np.asarray(SMB_nn)[:,0].flatten()
                        
#                        print("# years: " + str(SMB_nn.size))
                        print("\nGlacier: " + str(glacier_name))
                        print("Cumulative SMB: " + str(np.sum(SMB_nn)))
                        print("Area: " + str(glacier_area[-1]))
                        print("SMB: " + str(SMB_nn))
                        
                        # TODO: remove after test
#                        if(glimsID == 'G006934E45883N' or glimsID == 'G006985E45951N'):
#                            import pdb; pdb.set_trace()
                        
                        # We reduce the year range if the glacier disappeared before 2015
                        if(glacier_area.size < 49):
                            print("Glacier not in 2015: " + str(glacier_name))
                            year_end_file = 2003
                        else:
                            year_end_file = end_ref
                        
                        # We store the cumulative SMB for all the glaciers with their RGI ID
#                        cumulative_smb_glaciers = np.concatenate((cumulative_smb_glaciers, np.array([glims_glacier['ID'], np.sum(SMB_nn)])))
                        cumulative_smb_glaciers.append([glims_glacier['ID'], np.sum(SMB_nn)])
                        
                        # We store the simulated SMB 
                        store_file(SMB_nn, path_smb_all_glaciers_smb, "", "SMB", glimsID, start_ref, year_end_file+1)
                        store_file(glacier_area, path_smb_all_glaciers_area, "", "Area", glimsID, start_ref, year_end_file+1)
                        store_glacier_info(glacier_info, path_smb_all_glaciers_area)
                        
                        glacier_slope = np.repeat(x_reg_nn[0][5], year_end_file+1-start_ref)
                        store_file(glacier_slope, path_smb_all_glaciers_slope, "", "Slope_20", glimsID, start_ref, year_end_file+1)
                        
                        # We store all the ensemble predictions of SMB
                        ensemble_path = path_smb_all_glaciers_smb + glimsID + '_ensemble_SMB.txt'
                        if(os.path.exists(ensemble_path)):
                            ensemble_path = path_smb_all_glaciers_smb + glimsID + '_ensemble_SMB_2.txt'
                        with open(ensemble_path, 'wb') as ensemble_smb_f:
                            np.save(ensemble_smb_f, SMB_ensemble)
                        
                        
                    glacier_idx = glacier_idx+1
                    
                # End glacier loop
                
                # We store the cumulative glacier-wide SMB of all the glaciers in the French Alps
                cumulative_smb_glaciers = np.asarray(cumulative_smb_glaciers)
                store_smb_data(path_smb_all_glaciers + 'cumulative_smb_all_glaciers.csv', cumulative_smb_glaciers)
                
            else:
                
                # We make a deep copy to avoid issues when flattening for processing
                SMB_all = copy.deepcopy(SMB_raw)
                
                # We remove all previous simulations
                empty_folder(path_training_data + 'glacier_info\\')
                empty_folder(path_training_data + 'slope20\\')
                empty_folder(path_training_data + 'SMB\\')
                 
                # We split the sample weights in order to iterate them in the main loop
                SMB_flat = SMB_raw.flatten()
                smb_matrix_shp = SMB_raw.shape
                finite_mask = np.isfinite(SMB_flat)
#                temp_smb = np.where(np.isnan(SMB_raw), -99, SMB_raw)
#                zero_idx = np.argwhere(temp_smb == -99)
                
                sample_weights_flat = np.sqrt(compute_sample_weight(class_weight="balanced", y=SMB_flat[finite_mask]))
                
                # We refill the nan holes
                gidx = 0
                for glacier_y in SMB_flat:
                    if(math.isnan(glacier_y)):
                        sample_weights_flat = np.insert(sample_weights_flat, gidx, 'nan')
                    gidx = gidx+1
                
                sample_weights = np.reshape(sample_weights_flat, smb_matrix_shp)
                
                # We filter the NaN values
                finite_mask = np.isfinite(SMB_raw)
                SMB_raw = SMB_raw[finite_mask]
                
                ########################################################
                # SMB validation of training glacier dataset
                
                nfigure = 0
                glacier_idx = 0
                SMB_lasso_glaciers, SMB_ols_glaciers, SMB_ann_glaciers = [],[],[]
                for glims_glacier, SMB_glacier_nan, glacier_weights, lasso_logo_model  in zip(glims_rabatel, SMB_all, sample_weights, lasso_logo_models):
                    if(True):
                        glacier_name = glims_glacier['Glacier'].decode('ascii')
                    
#                    if(glacier_name == "d'Argentiere"):
                        glimsID = glims_glacier['GLIMS_ID'].decode('ascii')
                        print("\nSimulating Glacier: " + str(glacier_name))
    #                    print("Glacier GLIMS ID: " + str(glimsID))
                        
                        glacier_mask = np.isfinite(SMB_glacier_nan)
                        SMB_glacier = SMB_glacier_nan[glacier_mask]
        
                        # We retrie de CV ANN model
                        glacier_idx = np.where(glimsID.encode('ascii') == glims_rabatel['GLIMS_ID'])[0][0]
                        cv_ann_model = load_model(path_cv_ann + 'glacier_' + str(glacier_idx+1) + '_model.h5', custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error})
                        
                        glacier_mean_altitude = interpolate_glims_variable('MEAN_Pixel', glims_glacier, glims_2003, glims_1985)
                        glacier_area = interpolate_glims_variable('Area', glims_glacier, glims_2003, glims_1985)
                        
                        glacier_info = {'name':glacier_name, 'glimsID':glimsID, 'mean_altitude':glacier_mean_altitude, 'area': glacier_area}
                        
                        ####  SAFRAN FORCINGS  ####
                        # We recompute the SAFRAN forcings based on the current topographical data
                        season_anomalies, mon_anomalies = compute_SAFRAN_anomalies(glacier_info, year_range, year_start, all_glacier_coordinates, season_meteo_SMB, monthly_meteo_SMB, daily_meteo_data)
                        
                        # We create the spatiotemporal matrix to train the machine learning algorithm
                        x_reg_array, x_reg_full, x_reg_nn = create_spatiotemporal_matrix(season_anomalies, mon_anomalies, glims_glacier, glacier_mean_altitude, glacier_area, best_models)
                        
                        
                        # We filter the nan values
                        x_reg_array = x_reg_array[:, glacier_mask, :]
                        x_reg_full = x_reg_full[glacier_mask, :]
                        x_reg_nn = x_reg_nn[glacier_mask,:]
                        glacier_weights = glacier_weights[glacier_mask]
                        
                        #####  Machine learning SMB simulations   ###################
                        # ANN CV model
                        SMB_nn = cv_ann_model.predict(x_reg_nn, batch_size = 34)
                        SMB_nn = np.asarray(SMB_nn)[:,0].flatten()
                        
                        if(settings.smb_model_type == 'ann_weights'):
                            rmse_ann = math.sqrt(mean_squared_error(SMB_glacier, SMB_nn, glacier_weights))
                            r2_ann = r2_score(SMB_glacier, SMB_nn, glacier_weights)
                        elif(settings.smb_model_type == 'ann_no_weights'):
                            rmse_ann = math.sqrt(mean_squared_error(SMB_glacier, SMB_nn))
                            r2_ann = r2_score(SMB_glacier, SMB_nn)
                        SMB_ann_glaciers = np.concatenate((SMB_ann_glaciers, SMB_nn), axis=None)
                        
                        # Lasso
                        scaled_x_reg = full_scaler.transform(x_reg_full)
                        SMB_lasso = lasso_logo_model.predict(scaled_x_reg)
                        
                        rmse_lasso = math.sqrt(mean_squared_error(SMB_glacier, SMB_lasso))
                        r2_lasso = lasso_logo_model.score(scaled_x_reg, SMB_glacier)
                        SMB_lasso_glaciers = np.concatenate((SMB_lasso_glaciers, SMB_lasso), axis=None)
                        
                        
                        print("\nGlacier " + str(glacier_name))
                        print("Lasso r2 score: " + str(r2_lasso))
                        print("ANN r2 score: " + str(r2_ann))
                        
                        print("Lasso RMSE : " + str(rmse_lasso))
                        print("ANN RMSE : " + str(rmse_ann))
                        
                        # We refill the nan holes for the plot
                        gidx = 0
                        for glacier_y in SMB_glacier_nan:
                            if(math.isnan(glacier_y)):
                                SMB_glacier = np.insert(SMB_glacier, gidx, 'nan')
                                SMB_lasso = np.insert(SMB_lasso, gidx, 'nan')
                                SMB_nn = np.insert(SMB_nn, gidx, 'nan')
                            gidx = gidx+1
                            
                        #########  SMB SIMULATION PLOTS   ##############
                        plt.figure(nfigure)
                        fig, axes = plt.subplots()
                        plt.title("Glacier " + str(glacier_name), fontsize=20)
                        plt.ylabel('Glacier-wide SMB (m.w.e.)', fontsize=16)
                        plt.xlabel('Year', fontsize=16)
                        
                        plt.plot(range(start_ref,end_ref+1), np.cumsum(SMB_glacier), linewidth=3, color = 'C0', label='Ground truth SMB')
                        axes.fill_between(range(start_ref,end_ref+1), np.cumsum(SMB_glacier)-SMB_uncertainties[:,1], np.cumsum(SMB_glacier)+SMB_uncertainties[:,1], facecolor = "red", alpha=0.3)
                        plt.plot(range(start_ref,end_ref+1), np.cumsum(SMB_lasso), linewidth=3, color = 'C3', label='Lasso SMB')
                        plt.plot(range(start_ref,end_ref+1), np.cumsum(SMB_nn), linewidth=3, color = 'C8', label='ANN SMB')
                        
                        plt.tick_params(labelsize=14)
                        plt.legend()
                        plt.draw()
                        nfigure = nfigure+1
                        
                        # We store all the glacier's data to be compared later
                        print("\nStoring data...")
                        # We store the simulated SMB 
                         
                        store_file(SMB_nn, path_training_data, "SMB\\", "SMB", glimsID, start_ref, end_ref+1)
                        store_glacier_info(glacier_info, path_training_data + 'glacier_info\\')
                        
                        glacier_slope = np.repeat(x_reg_nn[0][5], end_ref+1-start_ref)
#                        import pdb; pdb.set_trace()
                        store_file(glacier_slope, path_training_data, "slope20\\", "Slope_20", glimsID, start_ref, end_ref+1)
                        
                        glacier_idx = glacier_idx+1
            
            if(not reconstruct):
                SMB_ann_glaciers = np.asarray(SMB_ann_glaciers)
                SMB_lasso_glaciers = np.asarray(SMB_lasso_glaciers)
                SMB_ols_glaciers = np.asarray(SMB_ols_glaciers)
                
                print("\nGathering all plots in a pdf file... ")
                try:
                    pdf_plots = path_smb_function_forcing + "SMB_lasso_ANN_SMB_simulations.pdf"
                    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_plots)
                    for fig in range(1, nfigure+1): 
                        pdf.savefig( fig )
                    pdf.close()
                    plt.close('all')
                    subprocess.Popen(pdf_plots, shell=True)
                    
                except IOError:
                    print("Could not open pdf. File already opened. Please close the pdf file.")
                    os.system('pause')
                    # We try again
                    try:
                        subprocess.Popen(pdf_plots, shell=True)
                    except IOError:
                        print("File still not available")
                        pass
                    
                plt.show()
                
                sample_weights = compute_sample_weight(class_weight='balanced', y=SMB_raw)
                
                if(settings.smb_model_type == 'ann_weights'):
                    rmse_ann = math.sqrt(mean_squared_error(SMB_raw, SMB_ann_glaciers, sample_weights))
                    r2_ann = r2_score(SMB_raw, SMB_ann_glaciers, sample_weights)
            
                elif(settings.smb_model_type == 'ann_no_weights'):
                    rmse_ann = math.sqrt(mean_squared_error(SMB_raw, SMB_ann_glaciers))
                    r2_ann = r2_score(SMB_raw, SMB_ann_glaciers)
                
                rmse_lasso = math.sqrt(mean_squared_error(SMB_raw, SMB_lasso_glaciers))
                r2_lasso = r2_score(SMB_raw, SMB_lasso_glaciers)
                
                print("\n Mean Lasso r2 for all glaciers: " + str(r2_lasso))
                print("\n Mean Lasso RMSE for all glaciers: " + str(rmse_lasso))
                print("\n Mean ANN r2 for all glaciers: " + str(r2_ann))
                print("\n Mean ANN RMSE for all glaciers: " + str(rmse_ann))
            
    else:
        print("\nSkipping...") 
            
    
    ###  End of main function   ###          
                