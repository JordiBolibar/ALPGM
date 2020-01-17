# -*- coding: utf-8 -*-

"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

ADAMONT TEMPERATURE (CPDD) AND PRECIPITATION (SNOW) COMPUTATION 

"""

## Dependencies: ##
from netCDF4 import Dataset
import numpy as np
from numpy import genfromtxt
from numba import jit
import copy
import os
import settings
from pathlib import Path
import pandas as pd

### FUNCTIONS  ####
def find_nearest_altitude(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def find_glacier_idx(glacier_massif, massif_number, altitudes, glacier_altitude, aspects, glacier_aspect):
    #### Aspects not used for ADAMONT
    massif_altitudes_idx = np.where(massif_number == float(glacier_massif))[0]
    if(len(massif_altitudes_idx) > 0):
        index_alt = find_nearest_altitude(altitudes[massif_altitudes_idx], glacier_altitude)
        final_idx = int(massif_altitudes_idx[index_alt])
    else:
        # ADAMONT data for this massif not available
        final_idx = -1
    return final_idx

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

@jit
def get_ADAMONT_glacier_coordinates(massif_number, zs, aspects, glims_data, glims_rabatel):
    glacier_ADAMONT_coordinates, all_glacier_coordinates = [],[]
    
    # Glaciers with SMB Data (Rabatel et al. 2016) + GLACIOCLIM
    for glims_id, glacier_name, glacier_massif, glacier_altitude, glacier_aspect in zip(glims_rabatel['GLIMS_ID'], glims_rabatel['Glacier'], glims_rabatel['Massif_SAFRAN'], glims_rabatel['MEDIAN_Pixel'], glims_rabatel['Aspect_num']):
        final_idx = find_glacier_idx(glacier_massif, massif_number, zs, glacier_altitude, aspects, glacier_aspect)
        if(final_idx != -1):
            glacier_ADAMONT_coordinates.append([glacier_name, final_idx, float(glacier_altitude), glims_id, int(glacier_massif)])
    
    # All glaciers loop
    for glims_id, glacier_name, glacier_massif, glacier_altitude, glacier_aspect in zip(glims_data['GLIMS_ID'], glims_data['Glacier'], glims_data['Massif_SAFRAN'], glims_data['MEDIAN_Pixel'], glims_data['Aspect_num']):
        final_idx = find_glacier_idx(glacier_massif, massif_number, zs, glacier_altitude, aspects, glacier_aspect)
        if(final_idx != -1):
            all_glacier_coordinates.append([glacier_name, final_idx, float(glacier_altitude), glims_id, int(glacier_massif)])
        
    return np.asarray(glacier_ADAMONT_coordinates), np.asarray(all_glacier_coordinates)

@jit
def get_SMB_glaciers(glacier_ADAMONT_coordinates):
    smb_glaciers = []
    for glacier in glacier_ADAMONT_coordinates:
        smb_glaciers.append(glacier[0])
    return np.asarray(smb_glaciers)


def compute_mean_values(idx, glacier_CPDDs, glacier_winter_snow, glacier_summer_snow):
    glacier_CPDDs[idx]['CPDD'] = np.asarray(glacier_CPDDs[idx]['CPDD'])
    glacier_CPDDs[idx]['Mean'] = glacier_CPDDs[idx]['CPDD'][1::2].mean()
    glacier_winter_snow[idx]['snow'] = np.asarray(glacier_winter_snow[idx]['snow'])
    glacier_winter_snow[idx]['Mean'] = glacier_winter_snow[idx]['snow'][1::2].mean()
    glacier_summer_snow[idx]['snow'] = np.asarray(glacier_summer_snow[idx]['snow'])
    glacier_summer_snow[idx]['Mean'] = glacier_summer_snow[idx]['snow'][1::2].mean()

    return glacier_CPDDs, glacier_winter_snow, glacier_summer_snow

def compute_mean_monthly_values(idx, glacier_mon_temp, glacier_mon_snow):
    
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
        
    return glacier_mon_temp, glacier_mon_snow


def main(compute):
    
    print("\n-----------------------------------------------")
    print("              ADAMONT FORCING ")
    print("-----------------------------------------------\n")
    
    if(compute):
        ######   FILE PATHS    #######
            
        # Folders     
        workspace = str(Path(os.getcwd()).parent) + '\\'
        # Path to be updated with location of the ADAMONT forcings
        path_adamont_forcings = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\treated\\'
#        path_adamont_forcings = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\FORCING_ADAMONT_IGE_BERGER\\'
#        path_adamont_forcings = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\FORCING_ADAMONT_IGE_BERGER\\subset_AGU\\'
        path_smb = workspace + 'glacier_data\\smb\\'
        path_smb_function_adamont = path_smb + 'smb_function\\ADAMONT\\'
#        path_glacier_coordinates = workspace + 'glacier_data\\glacier_coordinates\\' 
        path_glims = workspace + 'glacier_data\\GLIMS\\' 
        
        # Files
        forcing_daymean = settings.current_ADAMONT_model_daymean
        forcing_daysum = settings.current_ADAMONT_model_daysum
        print("Current ADAMONT combination: " + str(forcing_daymean))
        
        # We read the glacier classification by SAFRAN massifs
        glims_2015 = genfromtxt(path_glims + 'GLIMS_2015_massif.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8')])
#        glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8')])
        glims_rabatel = genfromtxt(path_glims + 'GLIMS_Rabatel_30_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('slope20', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8')])        
    
        file_forcing_daymean = Dataset(path_adamont_forcings + forcing_daymean)
        file_forcing_daysum = Dataset(path_adamont_forcings + forcing_daysum)
        
        lat = file_forcing_daymean.variables['LAT'][:]
        lon = file_forcing_daymean.variables['LON'][:]
        
        ADAMONT_coordinates = []
        for x,y in zip(lon,lat):
            ADAMONT_coordinates.append(np.hstack((x, y)))
        ADAMONT_coordinates = np.asarray(ADAMONT_coordinates)
        
#        import pdb; pdb.set_trace()
        
#        aspects = file_forcing_daymean.variables['aspect'][:]
        aspects = []
    
#        import pdb; pdb.set_trace()
        
        zs = file_forcing_daymean.variables['ZS'][:]
        massif_number = file_forcing_daymean.variables['MASSIF_NUMBER'][:]
        
        # Temperatures (from K to C)
        temps_mean = file_forcing_daymean.variables['Tair'][:] -273.15
        rain_sum = file_forcing_daysum.variables['RAIN'][:] * 3600
        snow_sum = file_forcing_daysum.variables['SNOW'][:] * 3600
        times = file_forcing_daysum.variables['TIME'][:]
        
        start = np.datetime64('2005-08-01 06:00:00')
        year_start = 2006
        year_end = 2099
        
         # We compute the yearly CPDD and snow accumulation only if it not yet available
        season_meteo_generated = os.path.exists(path_smb_function_adamont+'season_meteo.txt')
        monthly_meteo_generated = os.path.exists(path_smb_function_adamont+'glacier_winter_snow.txt')
        
        datetimes = np.array([start + np.timedelta64(np.int32(time), 'h') for time in times])
        yeartimes = []
        for h in datetimes:
            yeartimes.append(h.astype(object).timetuple().tm_year)
        yeartimes = np.asarray(yeartimes)
        
        # We start one year after in order to take into account the previous year's accumulation period
        year_period = range(year_start+1, year_end+1)
        
        # We get the massif points closer to each glacier's centroid
        ##  SEE IF TO COMPUTE WITH GLIMS 2015 IN ORDER TO HAVE 2015 ALTITUDE, OR WITH 2003 IF IN THE END WE STORE THE CPDD AT ALL THE ALTITUDINAL BANDS  ###
        glacier_ADAMONT_coordinates, all_glacier_coordinates = get_ADAMONT_glacier_coordinates(massif_number, zs, aspects, glims_2015, glims_rabatel)
        smb_glaciers = get_SMB_glaciers(glacier_ADAMONT_coordinates)
        
        # We create the data structures to process all the CPDD and snow accumulation data
        glacier_CPDD_layout = {'Glacier':"", 'CPDD':[], 'Mean':0.0, 'GLIMS_ID': ""}
        glacier_snow_layout = {'Glacier':"", 'snow':[], 'Mean':0.0, 'GLIMS_ID': ""}
        glacier_CPDDs_all, glacier_winter_snow_all, glacier_summer_snow_all = [],[],[]
        glacier_CPDDs_SMB, glacier_winter_snow_SMB, glacier_summer_snow_SMB = [],[],[]
        
        # And the data structures for the monthly meteo data
        glacier_mon_temp_layout = {'Glacier':"", 'GLIMS_ID':"", 'mon_temp':[], 'mon_means':[], 'years':[]}
        glacier_mon_snow_layout = {'Glacier':"", 'GLIMS_ID':"", 'mon_snow':[], 'mon_means':[], 'years':[]}
        glacier_mon_temp_all, glacier_mon_snow_all = [],[]
        glacier_mon_temp_SMB, glacier_mon_snow_SMB = [],[]
        
        for glacier in all_glacier_coordinates:
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
        
        raw_yearly_mean_CPDD, raw_yearly_mean_winter_snow, raw_yearly_mean_summer_snow, raw_yearly_mean_ablation_season = [],[],[],[]
        raw_yearly_mean_CPDD_SMB, raw_yearly_mean_winter_snow_SMB, raw_yearly_mean_summer_snow_SMB, raw_yearly_mean_ablation_season_SMB = [],[],[],[]
        
        if(not (season_meteo_generated and monthly_meteo_generated)):
            print("\nProcessing ADAMONT data...")
            for year in year_period:
                print("Hydrological year: " + str(year-1) + "-" + str(year))
                
                current_year_idx_1 = np.where(yeartimes == (year-1))
                current_year_idx = np.where(yeartimes == year)
                
                year_CPDD, year_w_snow, year_s_snow, year_ablation_season = 0,0,0,0
                year_CPDD_SMB, year_w_snow_SMB, year_s_snow_SMB, year_ablation_season_SMB = 0,0,0,0
                
                i, j = 0, 0
                start_div, end_div = 30, 10
                for glacier_coords in all_glacier_coordinates:
                    # Reset the indexes
                    glacier_idx = int(glacier_coords[1])
                    glacier_name = glacier_coords[0]
                    glacier_alt = float(glacier_coords[2])
                    
                    daily_temps_mean_1 = temps_mean[current_year_idx_1, glacier_idx].flatten() + ((zs[glacier_idx] - glacier_alt)/1000.0)*6.0
                    daily_temps_mean = temps_mean[current_year_idx, glacier_idx].flatten() + ((zs[glacier_idx] - glacier_alt)/1000.0)*6.0
                    
                    daily_snow_sum_1 = snow_sum[current_year_idx_1, glacier_idx].flatten()
                    daily_snow_sum = snow_sum[current_year_idx, glacier_idx].flatten()
                    
                    daily_rain_sum_1 = rain_sum[current_year_idx_1, glacier_idx].flatten()
                    daily_rain_sum = rain_sum[current_year_idx, glacier_idx].flatten()
                    
                    daily_datetimes_1 = datetimes[current_year_idx_1].flatten()
                    daily_datetimes = datetimes[current_year_idx].flatten()
                    
                    #### We compute the monthly average data
                    # Monthly average temperature
                    mon_temps_1 = get_monthly_temps(daily_temps_mean_1, daily_datetimes_1) 
                    mon_temps = get_monthly_temps(daily_temps_mean, daily_datetimes)
                    mon_temp_year = np.append(mon_temps_1[9:], mon_temps[:9])
                    
                    # We adjust the snowfall rate at the glacier's altitude
                    glacier_snow_1 = np.where(daily_temps_mean_1 > 2.0, 0.0, daily_snow_sum_1)
                    glacier_snow_1 = np.where(((daily_temps_mean_1 < 2.0) & (glacier_snow_1 == 0.0)), daily_rain_sum_1, glacier_snow_1)
                    glacier_snow = np.where(daily_temps_mean > 2.0, 0.0, daily_snow_sum)
                    glacier_snow = np.where(((daily_temps_mean < 2.0) & (glacier_snow == 0.0)), daily_rain_sum, glacier_snow)
                    
                    mon_snow_1 = get_monthly_snow(glacier_snow_1, daily_datetimes_1)
                    mon_snow = get_monthly_snow(glacier_snow, daily_datetimes)
                    mon_snow_year = np.append(mon_snow_1[9:], mon_snow[:9])
                    
                    # We get the indexes for the ablation and accumulation periods
                    pos_temp_year = np.where(daily_temps_mean < 0, 0, daily_temps_mean)
#                    pos_temp_year_1 = np.where(daily_temps_mean_1 < 0, 0, daily_temps_mean_1)
                    integ_temp = np.cumsum(pos_temp_year)
#                    integ_temp_1 = np.cumsum(pos_temp_year_1)
                    start_y_ablation = np.where(integ_temp > integ_temp.max()/start_div)[0]
#                    start_y_ablation_1 = np.where(integ_temp_1 > integ_temp_1.max()/start_div)[0]
                    end_y_ablation = np.where(integ_temp > (integ_temp.max() - integ_temp.max()/end_div))[0]
#                    end_y_ablation_1 = np.where(integ_temp_1 > (integ_temp_1.max() - integ_temp_1.max()/end_div))[0]
                    
#                    print(glacier_name)
#                    print(glacier_idx)
                    
#                    if(glacier_idx == 165 or glacier_idx == 166):
#                        import pdb; pdb.set_trace()
                    
                    start_ablation = start_y_ablation[0] 
                    end_ablation = end_y_ablation[0] 
#                    start_ablation_1 = start_y_ablation_1[0] 
#                    end_ablation_1 = end_y_ablation_1[0]
                    
                    # Classic 120-270 ablation period for the snow
                    ablation_idx = range(120, 271)
                    accum_idx_1 = range(271, daily_temps_mean_1.size)
                    accum_idx = range(0, 120)
                    
                    # We get the indexes for the ablation and accumulation periods
                    #Dynamic ablation period
                    ablation_temp_idx = range(start_ablation, end_ablation+1)
#                    accum_temp_idx_1 = range(end_ablation_1+1, daily_temps_mean_1.size)
#                    accum_temp_idx = range(0, start_ablation)
                    
                    glacier_ablation_temps =  daily_temps_mean[ablation_temp_idx]
#                    glacier_accumulation_temps = np.append(daily_temps_mean_1[accum_temp_idx_1], daily_temps_mean[accum_temp_idx])
                    
                    dummy_glacier_ablation_temps = daily_temps_mean[ablation_idx]
                    dummy_glacier_accumulation_temps = np.append(daily_temps_mean_1[accum_idx_1], daily_temps_mean[accum_idx])
        
                    glacier_year_pos_temps = np.where(glacier_ablation_temps < 0, 0, glacier_ablation_temps)
#                glacier_accum_pos_temps = np.where(glacier_accumulation_temps < 0, 0, glacier_accumulation_temps)
                    dummy_glacier_ablation_pos_temps = np.where(dummy_glacier_ablation_temps < 0, 0, dummy_glacier_ablation_temps)
                    dummy_glacier_accum_pos_temps = np.where(dummy_glacier_accumulation_temps < 0, 0, dummy_glacier_accumulation_temps)
        
                    glacier_accum_snow = np.append(daily_snow_sum_1[accum_idx_1], daily_snow_sum[accum_idx])
                    glacier_accum_rain = np.append(daily_rain_sum_1[accum_idx_1], daily_rain_sum[accum_idx])
    
                    glacier_ablation_snow = daily_snow_sum[ablation_idx]
                    glacier_ablation_rain = daily_rain_sum[ablation_idx]
                    
                    # We recompute the rain/snow limit with the new adjusted temperatures
                    glacier_accum_snow = np.where(dummy_glacier_accum_pos_temps > 2.0, 0.0, glacier_accum_snow)
                    glacier_accum_snow = np.where(((dummy_glacier_accumulation_temps < 2.0) & (glacier_accum_snow == 0.0)), glacier_accum_rain, glacier_accum_snow)
                    glacier_ablation_snow = np.where(dummy_glacier_ablation_pos_temps > 2.0, 0.0, glacier_ablation_snow)
                    glacier_ablation_snow = np.where(((dummy_glacier_ablation_temps < 2.0) & (glacier_ablation_snow == 0.0)), glacier_ablation_rain, glacier_ablation_snow)
                    
        #            glacier_ablation_season = end_y_ablation[0] - start_y_ablation[0]
                    glacier_ablation_season = len(ablation_temp_idx) + len(ablation_temp_idx)
        #            print("Ablation season length: " + str(glacier_ablation_season)
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
                    glacier_mon_snow_all[j]['years'].append(year)
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
                        # Glaciers with SMB data
                        glacier_CPDDs_all, glacier_winter_snow_all, glacier_summer_snow_all = compute_mean_values(j, glacier_CPDDs_all, glacier_winter_snow_all, glacier_summer_snow_all)
                        
                        glacier_mon_temp_all, glacier_mon_snow_all = compute_mean_monthly_values(j, glacier_mon_temp_all, glacier_mon_snow_all)
                        
                        if(np.any(smb_glaciers == glacier_name)):
                            glacier_CPDDs_SMB, glacier_winter_snow_SMB, glacier_summer_snow_SMB = compute_mean_values(i, glacier_CPDDs_SMB, glacier_winter_snow_SMB, glacier_summer_snow_SMB)
                        
                            glacier_mon_temp_SMB, glacier_mon_snow_SMB = compute_mean_monthly_values(i, glacier_mon_temp_SMB, glacier_mon_snow_SMB)
                            
                        ### End of glacier loop  ###
                        
                    # We iterate the independent indexes
                    j = j+1
                    if(np.any(smb_glaciers == glacier_name)):
                        i = i+1
                
                raw_yearly_mean_CPDD.append(year_CPDD/all_glacier_coordinates.shape[0])
                raw_yearly_mean_CPDD_SMB.append(year_CPDD_SMB/glacier_ADAMONT_coordinates.shape[0])
                raw_yearly_mean_winter_snow.append(year_w_snow/all_glacier_coordinates.shape[0])
                raw_yearly_mean_winter_snow_SMB.append(year_w_snow_SMB/glacier_ADAMONT_coordinates.shape[0])
                raw_yearly_mean_summer_snow.append(year_s_snow/all_glacier_coordinates.shape[0])
                raw_yearly_mean_summer_snow_SMB.append(year_s_snow_SMB/glacier_ADAMONT_coordinates.shape[0])
                raw_yearly_mean_ablation_season.append(year_ablation_season/all_glacier_coordinates.shape[0])
                raw_yearly_mean_ablation_season_SMB.append(year_ablation_season_SMB/glacier_ADAMONT_coordinates.shape[0])
                
            
            # We combine the full meteo dataset forcings for all glaciers
            # Seasonal
            season_meteo = {'CPDD':np.asarray(glacier_CPDDs_all), 'winter_snow':np.asarray(glacier_winter_snow_all), 'summer_snow':np.asarray(glacier_summer_snow_all)}
            # Monthly
            monthly_meteo = {'temp':np.asarray(glacier_mon_temp_all), 'snow':np.asarray(glacier_mon_snow_all)}
            
            print("\nStoring glacier CPDD and snowfall data...")
            
            if not os.path.exists(path_smb_function_adamont):
                os.makedirs(path_smb_function_adamont)
            with open(path_smb_function_adamont+'season_meteo.txt', 'wb') as season_meteo_f:
                        np.save(season_meteo_f, season_meteo)
            with open(path_smb_function_adamont+'monthly_meteo.txt', 'wb') as monthly_meteo_f:
                        np.save(monthly_meteo_f, monthly_meteo)
                        
            # End if CPDD and snow files not generated
                            
            # Ablation season length
            with open(path_smb_function_adamont+'raw_yearly_mean_ablation_season.txt', 'wb') as rym_as_f:
                        np.save(rym_as_f, raw_yearly_mean_ablation_season)
    
    else:
        print("Skipping...")
        
    ### End of main function ###

