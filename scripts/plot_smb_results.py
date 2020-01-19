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
import copy
#import math
import csv
from pathlib import Path
from matplotlib.lines import Line2D
#import numpy.polynomial.polynomial as poly
import statsmodels.api as sm
from scipy import stats
#from sklearn.metrics import r2_score
#import seaborn as sns

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
path_slope_safran = path_smb + 'smb_simulations\\SAFRAN\\1\\all_glaciers_1967_2015\\slope\\'

path_smb_glaciers = np.asarray(os.listdir(path_smb_safran))
path_area_glaciers = np.asarray(os.listdir(path_area_safran))
path_slope_glaciers = np.asarray(os.listdir(path_slope_safran))

glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8'), ('ID', '<i8')])
SMB_raw = genfromtxt(path_smb + 'SMB_raw_temporal.csv', delimiter=';', dtype=float)

all_glacier_smb, flat_all_glacier_smb  = [],[]
all_glaciers_avg_smb = {'avg_smb':[], 'GLIMS_ID':[], 'glacier_name':[]}
annual_smb_obs = [] 
annual_avg_smb_, annual_avg_smb_marzeion = [],[]
annual_avg_smb_big_glaciers_ = []
annual_avg_smb_small_glaciers_ = []
annual_avg_smb_very_small_glaciers_ = []
annual_avg_area, annual_avg_area_marzeion, annual_avg_slope = [],[],[]
flat_big_glaciers, flat_medium_glaciers, flat_small_glaciers = [],[],[]
avg_big_glaciers, avg_medium_glaciers, avg_small_glaciers = [],[],[]
avg_glacier_smb, all_glaciers_smb = [], []
area_glaciers = []

for year_idx in range(0, 49):
    annual_smb_obs.append([])
    annual_avg_smb_.append([])
    annual_avg_smb_marzeion.append([])
    annual_avg_smb_big_glaciers_.append([])
    annual_avg_smb_small_glaciers_.append([])
    annual_avg_smb_very_small_glaciers_.append([])
    annual_avg_area.append([])
    annual_avg_area_marzeion.append([])
    annual_avg_slope.append([])

### Process Marzeion et al. SMB data ###
process_marzeion = True

#### Process Marzeion et al. glacier-wide SMB data   #######
if(process_marzeion):
    path_smb_marzeion = 'C:\\Jordi\\PhD\\Data\\Marzeion\\RGI_6.0\\'
    SMB_raw_marzeion = genfromtxt(path_smb_marzeion + 'specific_mass_balance_CRU401_RGIv6.csv', delimiter=';', dtype=float)
    rgi_6 = genfromtxt(path_smb_marzeion + '11_rgi60_CentralEurope_hypso.csv', delimiter=';', dtype=str)
    
    SMB_marzeion_all = []
    
    for glacier in glims_2003:
        rgi_idx = np.where(SMB_raw_marzeion == glacier['ID'])
        
#        import pdb; pdb.set_trace()
        
        if(rgi_idx[0].size > 0):
            # We store the glacier's SMB timeseries between 1967 and 2015 and we convert them to m.w.e.
            SMB_marzeion_all.append({'IDs': glacier['ID'], 'SMB': (SMB_raw_marzeion[66:115, rgi_idx[1]].flatten())/1000}) 
#            rgi_IDs.append(int(rgi_6[rgi_idx, 0][-4:]))
        else:
            print("Glacier NOT found in RGI 6.0: " + str(glacier['Glacier'].decode('ascii') + " - " + str(glacier['GLIMS_ID'].decode('ascii')) + ' - ' + str(glacier['Area'])))
            # No SMB data from Marzeion for this glacier
            SMB_marzeion_all.append({'IDs': glacier['ID'], 'SMB': []})
    
    SMB_marzeion_all = np.asarray(SMB_marzeion_all)
    
# We plot the glacier-wide SMB observations in the French Alps

fig0, ax0 = plt.subplots(figsize=(20,9))
ax0.set_ylabel('Glacier-wide SMB (m.w.e. $a^{-1}$)', fontsize=16)
ax0.set_xlabel('Year', fontsize=16)

# We compute the average glacier-wide SMB from the 32 glaciers with obs
for glacier in SMB_raw:
    line0, = ax0.plot(range(1967, 2016), glacier[-49:], linewidth=1)
    yr_idx = 0
    for year in glacier[-49:]:
        annual_smb_obs[yr_idx].append(year)
        yr_idx = yr_idx+1
    
annual_avg_smb_obs = []
for year in annual_smb_obs:
    annual_avg_smb_obs.append(np.nanmean(year))
annual_avg_smb_obs = np.asarray(annual_avg_smb_obs)

line0, = ax0.plot(range(1967, 2016), annual_avg_smb_obs, linewidth=3, label='Average SMB', c='midnightblue')
ax0.legend()

ax0.axhline(y=0, color='black', linewidth=0.7, linestyle='-')

#ax1.set_title("Annual glacier-wide SMB of all French alpine glaciers")


#import pdb; pdb.set_trace()

##############################################################

massifs_safran = {'1':'Chablais', '6':'Haute-Tarantaise', '3':'Mont-Blanc', '10':'Vanoise',
                  '11':'Haute-Maurienne','8':'Belledonne','12':'Grandes-Rousses','15':'Oisans','16':'Pelvoux',
                  '13':'Thabor', '19':'Champsaur','18':'Devoluy','21':'Ubaye'}

smb_massif_template = {'Chablais':copy.deepcopy(annual_avg_smb_), 'Haute-Tarantaise':copy.deepcopy(annual_avg_smb_), 'Mont-Blanc':copy.deepcopy(annual_avg_smb_),
                       'Vanoise':copy.deepcopy(annual_avg_smb_), 'Haute-Maurienne':copy.deepcopy(annual_avg_smb_), 
                       'Belledonne':copy.deepcopy(annual_avg_smb_), 'Grandes-Rousses':copy.deepcopy(annual_avg_smb_),'Oisans':copy.deepcopy(annual_avg_smb_),
                       'Pelvoux':copy.deepcopy(annual_avg_smb_), 'Thabor':copy.deepcopy(annual_avg_smb_), 'Champsaur':copy.deepcopy(annual_avg_smb_), 
                       'Ubaye':copy.deepcopy(annual_avg_smb_)}

glacier_smb_massif_template = {'Chablais':[], 'Haute-Tarantaise':[], 'Mont-Blanc':[],
                       'Vanoise':[], 'Haute-Maurienne':[], 
                       'Belledonne':[], 'Grandes-Rousses':[],'Oisans':[],
                       'Pelvoux':[], 'Thabor':[], 'Champsaur':[], 
                       'Ubaye':[]}

smb_massif = copy.deepcopy(smb_massif_template)
mean_smb_glaciers, mean_area_glaciers, mean_slope_glaciers, mean_altitude_glaciers = np.zeros(661), np.zeros(661), np.zeros(661), np.zeros(661)
mean_fullperiod_smb_glaciers, mean_fullperiod_area_glaciers, mean_fullperiod_slope_glaciers, mean_fullperiod_mean_altitude_glaciers = [],[],[],[]
glacier_massifs = []
    
#fig1, ax1 = plt.subplots(figsize=(20,9))
fig1, ax1 = plt.subplots(figsize=(14,7))
ax1.set_ylabel('Glacier-wide SMB (m.w.e. $a^{-1}$)', fontsize=16)
#ax1.set_xlabel('Year', fontsize=16)
#ax1.set_title("Annual glacier-wide SMB of all French alpine glaciers")
ax1.tick_params(labelsize=16)

fig2, ax2 = plt.subplots(figsize=(14,7))
ax2.set_ylabel('Cumulative glacier-wide SMB (m.w.e)', fontsize=16)
ax2.set_xlabel('Year', fontsize=16)
#ax2.set_title("Cumulative glacier-wide SMB of all French alpine glaciers")
ax2.tick_params(labelsize=16)

# We draw vertical and horizontal lines
ax1.axhline(y=0, color='black', linewidth=0.9, linestyle='-')
ax1.axvline(x=1976, color='grey', linewidth=0.9, linestyle='--')
ax1.axvline(x=1980, color='grey', linewidth=0.9, linestyle='--')
ax1.axvline(x=1984, color='grey', linewidth=0.9, linestyle='--')
ax1.axvline(x=2014, color='grey', linewidth=0.9, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.9, linestyle='-')
ax2.axvline(x=1976, color='grey', linewidth=0.9, linestyle='--')
ax2.axvline(x=1980, color='grey', linewidth=0.9, linestyle='--')
ax2.axvline(x=1984, color='grey', linewidth=0.9, linestyle='--')
ax2.axvline(x=2014, color='grey', linewidth=0.9, linestyle='--')

# Iterate all glaciers with the full simulated period
glacier_idx, glacier_idx_2015 = 0, 0
glaciers_not_2015 = 0
big_glaciers, small_glaciers, very_small_glaciers = 0, 0, 0
big_glaciers_2015, small_glaciers_2015, very_small_glaciers_2015 = 0, 0, 0

RGI_IDs = glims_2003['ID']

# Process individual glacier files
for path_smb, path_area, path_slope in zip(path_smb_glaciers, path_area_glaciers, path_slope_glaciers):
    # Glacier area
    area_glacier = genfromtxt(path_area_safran + path_area, delimiter=';')
    area_glacier = area_glacier[:,1].flatten()
    # Glacier slope
    slope_glacier = genfromtxt(path_slope_safran + path_slope, delimiter=';')
    slope_glacier = slope_glacier[:,1].flatten()
    # Glacier SMB
    smb_glacier = genfromtxt(path_smb_safran + path_smb, delimiter=';')
    smb_glacier = smb_glacier[:,1].flatten()
    # Glacier info
    # {'name':glacier_name, 'glimsID':glimsID, 'mean_altitude':glacier_mean_altitude, 'area': glacier_area}
    with open(path_area_safran + 'glacier_info_' + path_area[:14], 'rb') as glacier_info_f:
        glacier_info = np.load(glacier_info_f, encoding='latin1').item()
    
    current_massif = massifs_safran[str(glacier_info['massif_SAFRAN'])]
    mean_altitude = glacier_info['mean_altitude']
    
    all_glacier_smb.append(np.asarray(smb_glacier))
    flat_all_glacier_smb = np.concatenate((flat_all_glacier_smb, smb_glacier), axis=0)
    all_glaciers_avg_smb['avg_smb'].append(np.mean(smb_glacier))
    all_glaciers_avg_smb['GLIMS_ID'].append(glacier_info['glimsID'])
    all_glaciers_avg_smb['glacier_name'].append(glacier_info['name']) 
    
    # Retrieve Marzeion et al. SMB data based on the RGI_ID v6
    marzeion_idx = np.where(RGI_IDs == glacier_info['ID'])
    
    SMB_marzeion = SMB_marzeion_all[marzeion_idx][0]
    
    if(smb_glacier.size < 49):
        nan_tail = np.zeros(2015-2003)
        nan_tail[:] = np.nan
        smb_glacier = np.concatenate((smb_glacier, nan_tail))
        area_glacier_i = area_glacier.mean()
        slope_glacier_i = slope_glacier.mean()
        glaciers_not_2015 = glaciers_not_2015+1
    else:
        area_glacier_i = area_glacier[-15]
        
    if(area_glacier_i < 0.1):
        linewidth = 0.08
#        linewidth = 0.01
    elif(area_glacier_i < 0.5):
        linewidth = 0.15
#        linewidth = 0.01
    elif(area_glacier_i < 5):
        linewidth = 0.5
#        linewidth = 0.01
    else:
        linewidth = 0.6
    
    mean_smb_glaciers[glacier_idx] = np.nanmean(smb_glacier)
    mean_area_glaciers[glacier_idx] = np.nanmean(area_glacier)
    mean_slope_glaciers[glacier_idx] = slope_glacier[0]
    mean_altitude_glaciers[glacier_idx] = np.median(mean_altitude)
    glacier_massifs.append(current_massif)
    
    if(smb_glacier.size == 49):
        mean_fullperiod_smb_glaciers.append(np.nanmean(smb_glacier))
        mean_fullperiod_mean_altitude_glaciers.append(np.median(mean_altitude))
        mean_fullperiod_slope_glaciers.append(slope_glacier[0])
        mean_fullperiod_area_glaciers.append(np.nanmean(area_glacier))
    
    # TODO: see what to 2 with 2 glaciers with strange behaviour
    # So far we filter them from the graphs
#    if(np.sum(smb_glacier[:-15]) < 0):
#        line1, = ax1.plot(range(1967, 2016), smb_glacier, linewidth=linewidth)
#        line2, = ax2.plot(range(1967, 2016), np.cumsum(smb_glacier), linewidth=linewidth)
    
    big_glacier, small_glacier, very_small_glacier = False, False, False
    
    # We compute the overall average SMB glacier by glacier
    avg_glacier_smb.append(np.nanmean(smb_glacier))
    area_glaciers.append(area_glacier_i)
    
    # We store all the glacier's SMB series together
    all_glaciers_smb.append(smb_glacier)
    
    for year_idx in range(0, 49):
        if(len(SMB_marzeion['SMB']) > 0):
            annual_avg_smb_marzeion[year_idx].append(SMB_marzeion['SMB'][year_idx])
            annual_avg_area_marzeion[year_idx].append(area_glacier_i)
#        else:
#            annual_avg_smb_marzeion[year_idx].append(np.nan)
        if(not np.isnan(smb_glacier[year_idx])):
            annual_avg_smb_[year_idx].append(smb_glacier[year_idx])
            annual_avg_area[year_idx].append(area_glacier_i)
            smb_massif[current_massif][year_idx].append(smb_glacier[year_idx])
            
            if(area_glacier_i >= 2):
                annual_avg_smb_big_glaciers_[year_idx].append(smb_glacier[year_idx])
                big_glacier = True
            elif(area_glacier_i > 0.5):
                annual_avg_smb_small_glaciers_[year_idx].append(smb_glacier[year_idx])
                small_glacier = True
            else:
                annual_avg_smb_very_small_glaciers_[year_idx].append(smb_glacier[year_idx])
                very_small_glacier = True
        
    # TODO: erase after investigation of behaviour of last year
#    if(np.sum(smb_glacier[:-15]) > 0):
#    if(smb_glacier[:-15].min() < -3):
#        print("\nStrange glacier")
#        print("Glacier: " + str(glacier_info['name']))
#        print("Massif: " + str(current_massif))
#        print("Zmean: " + str(glacier_info['mean_altitude'][-1]))
#        print("Area: " + str(area_glacier))
#        print("Slope: " + str(slope_glacier))
#        print("SMB: " + str(smb_glacier))
#        print("glacier_info: " + str(glacier_info))
    
    # All glaciers indexes
    glacier_idx = glacier_idx+1
    if(big_glacier):
        flat_big_glaciers = np.concatenate((flat_big_glaciers, smb_glacier), axis=0)
        big_glaciers = big_glaciers+1
        avg_big_glaciers.append(np.average(smb_glacier))
    elif(small_glacier):
        flat_medium_glaciers = np.concatenate((flat_medium_glaciers, smb_glacier), axis=0)
        small_glaciers = small_glaciers+1
        avg_medium_glaciers.append(np.average(smb_glacier))
    elif(very_small_glacier):
        flat_small_glaciers = np.concatenate((flat_small_glaciers, smb_glacier), axis=0)
        very_small_glaciers = very_small_glaciers+1
        avg_small_glaciers.append(np.average(smb_glacier))
    
    # 2003-2015 glacier indexes
    if(not np.isnan(smb_glacier[-1])):
        glacier_idx_2015 = glacier_idx_2015+1
        
        if(big_glacier):
            big_glaciers_2015 = big_glaciers_2015+1
        elif(small_glacier):
            small_glaciers_2015 = small_glaciers_2015+1
        elif(very_small_glacier):
            very_small_glaciers_2015 = very_small_glaciers_2015+1
    
    print("Glacier #" + str(glacier_idx))
    

print("big glaciers: " + str(big_glaciers))
print("medium glaciers: " + str(small_glaciers))
print("small glaciers: " + str(very_small_glaciers))

avg_glacier_smb = np.asarray(avg_glacier_smb)
area_glaciers = np.asarray(area_glaciers)
all_glaciers_smb = np.asarray(all_glaciers_smb)

mean_fullperiod_smb_glaciers = np.asarray(mean_fullperiod_smb_glaciers)
mean_fullperiod_area_glaciers = np.asarray(mean_fullperiod_area_glaciers)
mean_fullperiod_mean_altitude_glaciers = np.asarray(mean_fullperiod_mean_altitude_glaciers)
mean_fullperiod_slope_glaciers = np.asarray(mean_fullperiod_slope_glaciers)

# All glaciers
all_glacier_smb = np.asarray(all_glacier_smb) 
flat_all_glacier_smb = np.asarray(flat_all_glacier_smb)
a_avg_smb, a_avg_smb_marzeion, a_avg_smb_big, a_avg_smb_small, a_avg_smb_v_small = [],[],[],[],[]

for avg_smb, avg_smb_marzeion, avg_smb_big, avg_smb_medium, avg_smb_small, avg_area, avg_area_marzeion in zip(annual_avg_smb_, annual_avg_smb_marzeion, annual_avg_smb_big_glaciers_, annual_avg_smb_small_glaciers_, annual_avg_smb_very_small_glaciers_, annual_avg_area, annual_avg_area_marzeion):
    # Area weighted mean
#    a_avg_smb.append(np.average(avg_smb, weights=avg_area))
#    a_avg_smb_marzeion.append(np.average(avg_smb_marzeion, weights=avg_area_marzeion))
#    a_avg_smb_big.append(np.asarray(avg_smb_big).mean())
#    a_avg_smb_small.append(np.asarray(avg_smb_medium).mean())
#    a_avg_smb_v_small.append(np.asarray(avg_smb_small).mean())
    a_avg_smb.append(np.average(avg_smb))
    a_avg_smb_marzeion.append(np.average(avg_smb_marzeion))
    a_avg_smb_big.append(np.asarray(avg_smb_big).mean())
    a_avg_smb_small.append(np.asarray(avg_smb_medium).mean())
    a_avg_smb_v_small.append(np.asarray(avg_smb_small).mean())

avg_smb_massif = copy.deepcopy(smb_massif_template)  
for massif, avg_massif in zip(smb_massif, avg_smb_massif):
    year_idx = 0
    for annual_smb, annual_avg_smb in zip(smb_massif[massif], avg_smb_massif[avg_massif]):
#        import pdb; pdb.set_trace()
        avg_smb_massif[avg_massif][year_idx] = np.average(annual_smb)
        year_idx = year_idx +1
        
a_avg_smb = np.asarray(a_avg_smb)
a_avg_smb_marzeion = np.asarray(a_avg_smb_marzeion)
a_avg_smb_big = np.asarray(a_avg_smb_big)
a_avg_smb_small = np.asarray(a_avg_smb_small)
a_avg_smb_v_small = np.asarray(a_avg_smb_v_small)

flat_big_glaciers = np.asarray(flat_big_glaciers)
flat_medium_glaciers = np.asarray(flat_medium_glaciers)
flat_small_glaciers = np.asarray(flat_small_glaciers)

avg_big_glaciers = np.asarray(avg_big_glaciers)
avg_medium_glaciers = np.asarray(avg_medium_glaciers)
avg_small_glaciers = np.asarray(avg_small_glaciers)

# We save the average glacier-wide SMB for all the glaciers in a single file
csv_columns = ['glacier_name','GLIMS_ID','avg_smb']
try:
    with open(path_smb_safran + 'single_file\\french_alps_SMB_1967_2015.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=csv_columns)
        writer.writeheader()
        for name, glims_id, avg_smb in zip(all_glaciers_avg_smb['glacier_name'], all_glaciers_avg_smb['GLIMS_ID'], all_glaciers_avg_smb['avg_smb']):
            writer.writerow({'glacier_name':name, 'GLIMS_ID':glims_id, 'avg_smb':avg_smb})
except IOError:
    print("I/O error") 

#import pdb; pdb.set_trace()

# We compute the correlation of each glacier with respect the weighted mean
glacier_correlation = []
for glacier_smb in all_glacier_smb:
    if(glacier_smb.size > 37):
        glacier_correlation.append(np.corrcoef(a_avg_smb, glacier_smb)[0,1]**2)
glacier_correlation = np.asarray(glacier_correlation)

print("\nMax SMB common variance: " + str(glacier_correlation.max()))
print("\nMin SMB common variance: " + str(glacier_correlation.min()))
print("\nAverage SMB common variance: " + str(glacier_correlation.mean()))


# Area weighted mean
line13, = ax1.plot(range(1967, 2016), annual_avg_smb_obs, linewidth=3, c='olivedrab', linestyle='--', label='Mean of glaciers with observations')
line11, = ax1.plot(range(1967, 2016), a_avg_smb, linewidth=3, c='steelblue', label='Area weighted mean (this study)')
line12, = ax1.plot(range(1967, 2016), a_avg_smb_marzeion, linewidth=3, c='darkgoldenrod', label='Area weighted mean (update of Marzeion et al., 2015)')

line23, = ax2.plot(range(1967, 2016), np.cumsum(annual_avg_smb_obs), linewidth=3, c='olivedrab', linestyle='--', label='Mean of glaciers with observations')
line21, = ax2.plot(range(1967, 2016), np.cumsum(a_avg_smb), linewidth=3, c='steelblue', label='Area weighted mean (this study)')
line22, = ax2.plot(range(1967, 2016), np.cumsum(a_avg_smb_marzeion), linewidth=3, c='darkgoldenrod', label='Area weighted mean (update of Marzeion et al., 2015)')

ax1.legend(fontsize='x-large')
ax2.legend(fontsize='x-large')

print("\nNumber of glaciers disappeared between 2003 and 2015: " + str(glaciers_not_2015))


### Average plots for glacier size
fig3, (ax3, ax4) = plt.subplots(1,2)
fig3.suptitle("Annual glacier-wide SMB of French alpine glaciers depending on size")
ax3.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax3.set_ylabel('Glacier-wide SMB (m.w.e. $a^{-1}$)')
ax3.set_xlabel('Year')
line14, = ax3.plot(range(1967, 2016), a_avg_smb_v_small, linewidth=1, label='Glaciers < 0.5 km$^2$', c='darkred')
line13, = ax3.plot(range(1967, 2016), a_avg_smb_small, linewidth=1, label='Glaciers 0.5 - 2 km$^2$', c='C1')
line12, = ax3.plot(range(1967, 2016), a_avg_smb_big, linewidth=1, label='Glaciers > 2 km$^2$', c='C0')
ax3.legend()

ax4.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax4.set_ylabel('Cumulative glacier-wide SMB (m.w.e.)')
ax4.set_xlabel('Year')
line24, = ax4.plot(range(1967, 2016), np.cumsum(a_avg_smb_v_small), linewidth=1, label='Glaciers < 0.5 km$^2$ (n = 534)', c='darkred')
line23, = ax4.plot(range(1967, 2016), np.cumsum(a_avg_smb_small), linewidth=1, label='Glaciers 0.5 - 2 km$^2$ (n = 93)', c='C1')
line22, = ax4.plot(range(1967, 2016), np.cumsum(a_avg_smb_big), linewidth=1, label='Glaciers > 2 km$^2$ (n = 34)', c='C0')
ax4.legend()

annual_smb_big_glaciers_flat = np.asarray(annual_avg_smb_big_glaciers_).flatten()

annual_avg_smb_small_glaciers_flat_s = np.asarray(annual_avg_smb_small_glaciers_).flatten()
annual_smb_small_glaciers_flat = []
for value in annual_avg_smb_small_glaciers_flat_s:
    annual_smb_small_glaciers_flat = np.concatenate((annual_smb_small_glaciers_flat, value), axis=0)
    
annual_avg_smb_very_small_glaciers_flat_s = np.asarray(annual_avg_smb_very_small_glaciers_).flatten()
annual_smb_very_small_glaciers_flat = []
for value in annual_avg_smb_very_small_glaciers_flat_s:
    annual_smb_very_small_glaciers_flat = np.concatenate((annual_smb_very_small_glaciers_flat, value), axis=0)

big_avg_smbs, small_avg_smbs, very_small_avg_smbs = [],[],[] 
big_avg_areas, small_avg_areas, very_small_avg_areas = [],[],[] 
glacier_smb_per_massif = copy.deepcopy(glacier_smb_massif_template)
glacier_area_per_massif = copy.deepcopy(glacier_smb_massif_template)
for area, smb, massif in zip(mean_area_glaciers, mean_smb_glaciers, glacier_massifs):
    # We compute the average SMB per glacier per size
    if(area >= 2):
        big_avg_smbs.append(smb)
        big_avg_areas.append(area)
    elif(area >= 0.5):
        small_avg_smbs.append(smb)
        small_avg_areas.append(area)
    else:
        very_small_avg_smbs.append(smb)
        very_small_avg_areas.append(area)
    # We compute the average SMB per glacier per massif
    glacier_smb_per_massif[massif].append(smb)
    glacier_area_per_massif[massif].append(area)
        
big_avg_smbs = np.asarray(big_avg_smbs)
small_avg_smbs = np.asarray(small_avg_smbs)
very_small_avg_smbs = np.asarray(very_small_avg_smbs)

print("\nOverall mean annual glacier-wide SMB per glacier size: ")
print("Overall: " + str(np.nanmean(flat_all_glacier_smb)))
print("Big glaciers: " + str(np.nanmean(flat_big_glaciers)))
print("Medium glaciers: " + str(np.nanmean(flat_medium_glaciers)))
print("Small glaciers: " + str(np.nanmean(flat_small_glaciers)))

print("\nMean (per year) annual glacier-wide SMB per glacier size: ")
print("Big glaciers: " + str(np.average(a_avg_smb_big)))
print("Medium glaciers: " + str(np.average(a_avg_smb_small)))
print("Small glaciers: " + str(np.average(a_avg_smb_v_small)))

print("\nMean (per glacier) annual glacier-wide SMB per glacier size: ")
print("Big glaciers: " + str(np.average(big_avg_smbs, weights=big_avg_areas)))
print("Medium glaciers: " + str(np.average(small_avg_smbs, weights=small_avg_areas)))
print("Small glaciers: " + str(np.average(very_small_avg_smbs, weights=very_small_avg_areas)))

print("\n ----------------------- ")

#import pdb; pdb.set_trace()

print("\nStandard deviation per glacier size (overall): ")
print("Overall: " + str(np.std(flat_all_glacier_smb, axis=0, dtype=np.float64)))
print("Big glaciers: " + str(np.std(flat_big_glaciers, axis=0, dtype=np.float64, ddof = 1)))
print("Medium glaciers: " + str(np.std(flat_medium_glaciers, axis=0, dtype=np.float64, ddof = 1)))
print("Small glaciers: " + str(np.std(flat_small_glaciers, axis=0, dtype=np.float64, ddof = 1)))

print("\nStandard deviation per glacier size (glacier means): ")
print("Overall: " + str(np.std(mean_smb_glaciers, axis=0, dtype=np.float64)))
print("Big glaciers: " + str(np.std(big_avg_smbs, axis=0, dtype=np.float64, ddof = 1)))
print("Medium glaciers: " + str(np.std(small_avg_smbs, axis=0, dtype=np.float64, ddof = 1)))
print("Small glaciers: " + str(np.std(very_small_avg_smbs, axis=0, dtype=np.float64, ddof = 1)))

print("\nArea weighted mean annual glacier-wide SMB (year by year): " + str(np.average(a_avg_smb)))
print("\nArea weighted mean annual glacier-wide SMB (glacier by glacier)\n: " + str(np.average(avg_glacier_smb, weights=area_glaciers)))

### Average SMB with uncertainties
avg_a_uncertainty = 0.32 # m.w.e (MAE)

fig5, ax5 = plt.subplots()
ax5.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax5.set_ylabel('Glacier-wide SMB (m.w.e. $a^{-1}$)')
ax5.set_xlabel('Year')
ax5.set_title("Mean annual glacier-wide SMB of all French alpine glaciers")
ax5.fill_between(range(1967, 2016), a_avg_smb-avg_a_uncertainty, a_avg_smb+avg_a_uncertainty, facecolor = "red", alpha=0.3)
line52, = ax5.plot(range(1967, 2016), a_avg_smb, linewidth=2, label='Area weighted average')
ax5.legend()

fig6, ax6 = plt.subplots()
ax6.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax6.set_ylabel('Cumulative glacier-wide SMB (m.w.e.)')
ax6.set_xlabel('Year')
ax6.set_title("Cumulative weighted mean annual glacier-wide SMB of all French alpine glaciers")
ax6.fill_between(range(1967, 2016), np.cumsum(a_avg_smb-avg_a_uncertainty), np.cumsum(a_avg_smb+avg_a_uncertainty), facecolor = "red", alpha=0.3)
line62, = ax6.plot(range(1967, 2016), np.cumsum(a_avg_smb), linewidth=2, label='Area weighted average')
ax6.legend()

# Scatter plots
# Remove outliers to see trend
mean_fullperiod_area_glaciers = mean_fullperiod_area_glaciers[(mean_fullperiod_smb_glaciers > -1.5) & (mean_fullperiod_smb_glaciers < 0.0)]
mean_fullperiod_slope_glaciers = mean_slope_glaciers[(mean_fullperiod_smb_glaciers > -1.5) & (mean_fullperiod_smb_glaciers < 0.0)]
mean_fullperiod_mean_altitude_glaciers = mean_fullperiod_mean_altitude_glaciers[(mean_fullperiod_smb_glaciers > -1.5) & (mean_fullperiod_smb_glaciers < 0.0)]
mean_fullperiod_smb_glaciers = mean_fullperiod_smb_glaciers[(mean_fullperiod_smb_glaciers > -1.5) & (mean_fullperiod_smb_glaciers < 0.0)]

# We compute stats for the linear regression fits
x_area_const = sm.add_constant(mean_fullperiod_area_glaciers)
area_model = sm.OLS(mean_fullperiod_smb_glaciers, x_area_const)
area_fit = area_model.fit()
print("Area stats")
#print(area_fit.summary())
print(stats.pearsonr(mean_fullperiod_area_glaciers, mean_fullperiod_smb_glaciers))

x_slope_const = sm.add_constant(mean_fullperiod_slope_glaciers)
slope_model = sm.OLS(mean_fullperiod_smb_glaciers, x_slope_const)
slope_fit = slope_model.fit()
print("Slope stats")
#print(slope_fit.summary())
print(stats.pearsonr(mean_fullperiod_slope_glaciers, mean_fullperiod_smb_glaciers))

fig7, (ax7, ax8, ax81) = plt.subplots(1,3, figsize=(9, 5))
plt.subplots_adjust(top=0.88, bottom=0.11, left=0.11, right=0.9, hspace=0.2, wspace=0.1)
#fig7.suptitle("Average annual glacier-wide SMB vs Surface area and Slope")
ax7.set_xlabel('Surface area (km$^2$)')
ax7.set_ylabel('Annual glacier-wide SMB (m.w.e. $a^{-1}$)')
#ax7.set_title("Average annual glacier-wide SMB vs Glacier surface area", y=1.03)
ax7.scatter(mean_fullperiod_area_glaciers, mean_fullperiod_smb_glaciers, s=4, alpha=0.7)
log_area = np.log10(mean_fullperiod_area_glaciers)
ax7.plot(np.unique(mean_fullperiod_area_glaciers), np.poly1d(np.polyfit(log_area, mean_fullperiod_smb_glaciers, 1))(np.unique(log_area)), c='purple')
ax7.set_xscale('log')

ax8.set_xlabel('Lowermost 20% altitudinal range slope (°)')
ax8.set_yticklabels([])
ax8.scatter(mean_fullperiod_slope_glaciers, mean_fullperiod_smb_glaciers, s=4, alpha=0.7)
ax8.plot(np.unique(mean_fullperiod_slope_glaciers), np.poly1d(np.polyfit(mean_fullperiod_slope_glaciers, mean_fullperiod_smb_glaciers, 1))(np.unique(mean_fullperiod_slope_glaciers)), c='purple')

mean_fullperiod_smb_glaciers = mean_fullperiod_smb_glaciers[(mean_fullperiod_mean_altitude_glaciers > 2300) & (mean_fullperiod_mean_altitude_glaciers < 3400)]
mean_fullperiod_mean_altitude_glaciers = mean_fullperiod_mean_altitude_glaciers[(mean_fullperiod_mean_altitude_glaciers > 2300) & (mean_fullperiod_mean_altitude_glaciers < 3400)]
ax81.set_xlabel('Mean altitude (m.a.s.l.)')
ax81.set_yticklabels([])
ax81.scatter(mean_fullperiod_mean_altitude_glaciers, mean_fullperiod_smb_glaciers, s=4, alpha=0.7)
ax81.plot(np.unique(mean_fullperiod_mean_altitude_glaciers), np.poly1d(np.polyfit(mean_fullperiod_mean_altitude_glaciers, mean_fullperiod_smb_glaciers, 1))(np.unique(mean_fullperiod_mean_altitude_glaciers)), c='purple')
#import pdb; pdb.set_trace()

# We compute stats for the linear regression fits
x_altitude_const = sm.add_constant(mean_fullperiod_mean_altitude_glaciers)
altitude_model = sm.OLS(mean_fullperiod_smb_glaciers, x_altitude_const)
altitude_fit = altitude_model.fit()
print("Altitude stats")
#print(altitude_fit.summary())
print(stats.pearsonr(mean_fullperiod_mean_altitude_glaciers, mean_fullperiod_smb_glaciers))

standard_deviation = np.std(mean_smb_glaciers, axis=0, dtype=np.float64)

### Decade average SMB (computed year by year)
avg_smb_70s = a_avg_smb[3:13].mean()
avg_smb_80s = a_avg_smb[13:23].mean()
avg_smb_90s = a_avg_smb[23:33].mean()
avg_smb_00s = a_avg_smb[33:43].mean()
avg_smb_10s = a_avg_smb[43:].mean()

avg_smb_marzeion_70s = a_avg_smb_marzeion[3:13].mean()
avg_smb_marzeion_80s = a_avg_smb_marzeion[13:23].mean()
avg_smb_marzeion_90s = a_avg_smb_marzeion[23:33].mean()
avg_smb_marzeion_00s = a_avg_smb_marzeion[33:43].mean()
avg_smb_marzeion_10s = a_avg_smb_marzeion[43:].mean()

avg_decadal_smb = np.array([avg_smb_70s, avg_smb_80s, avg_smb_90s, avg_smb_00s, avg_smb_10s])
avg_decadal_smb_marzeion = np.array([avg_smb_marzeion_70s, avg_smb_marzeion_80s, avg_smb_marzeion_90s, avg_smb_marzeion_00s, avg_smb_marzeion_10s])
xmin = np.array([1970, 1980, 1990, 2000, 2010])
xmax = np.array([1980, 1990, 2000, 2010, 2015])
total_avg_smb = a_avg_smb.mean()
total_avg_smb_marzeion = a_avg_smb_marzeion.mean()

### Decade average SMB (computed glacier by glacier)
avg_smb_70s_g, avg_smb_80s_g, avg_smb_90s_g, avg_smb_00s_g, avg_smb_10s_g = [],[],[],[],[]
bool_mask_10s = []
for glacier in all_glaciers_smb:
    avg_smb_70s_g.append(np.nanmean(glacier[3:13]))
    avg_smb_80s_g.append(np.nanmean(glacier[13:23]))
    avg_smb_90s_g.append(np.nanmean(glacier[23:33]))
    avg_smb_00s_g.append(np.nanmean(glacier[33:43]))
    avg_smb_10s_g.append(np.nanmean(glacier[43:])) 

finite_mask = np.isfinite(avg_smb_10s_g)
avg_smb_70s_g = np.average(avg_smb_70s_g, weights=area_glaciers)
avg_smb_80s_g = np.average(avg_smb_80s_g, weights=area_glaciers)
avg_smb_90s_g = np.average(avg_smb_90s_g, weights=area_glaciers)
avg_smb_00s_g = np.average(avg_smb_00s_g, weights=area_glaciers)

avg_smb_10s_g = np.asarray(avg_smb_10s_g)
avg_smb_10s_g = np.average(avg_smb_10s_g[finite_mask], weights=area_glaciers[finite_mask])

avg_decadal_smb_g = np.array([avg_smb_70s_g, avg_smb_80s_g, avg_smb_90s_g, avg_smb_00s_g, avg_smb_10s_g])
xmin = np.array([1970, 1980, 1990, 2000, 2010])
xmax = np.array([1980, 1990, 2000, 2010, 2015])
total_avg_smb_g = np.average(avg_glacier_smb, weights=area_glaciers)

fig9, ax9 = plt.subplots(figsize=(6, 4))
#fig9, ax9 = plt.subplots()
ax9.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
#ax9.axvline(x=2015, color='grey', linewidth=0.7, linestyle='-')
ax9.set_ylabel('Average glacier-wide SMB (m.w.e. $a^{-1}$)')
ax9.set_xlabel('Year')
#ax9.set_title("Average glacier-wide SMB per decade")
ax9.fill_between(range(1967, 2016), total_avg_smb_g-standard_deviation, total_avg_smb_g+standard_deviation, facecolor = "teal", alpha=0.25, label=r'$\sigma$')
#ax9.hlines(total_avg_smb_g, 1967, 2015, color='darkblue', linewidth=7, label='Total average SMB (this study)')
ax9.hlines(total_avg_smb_g, 1967, 2015, color='darkblue', linewidth=7, label='Total average SMB')
#ax9.hlines(total_avg_smb_marzeion, 1967, 2015, color='darkred', linewidth=4, label='Total average SMB (update of Marzeion et al., 2015)')
#ax9.hlines(avg_decadal_smb_g, xmin, xmax, color='steelblue', linewidth=6, label='Decadal average SMB (this study)')
ax9.hlines(avg_decadal_smb_g, xmin, xmax, color='steelblue', linewidth=6, label='Decadal average SMB')
#ax9.hlines(avg_decadal_smb_marzeion, xmin, xmax, color='darkgoldenrod', linewidth=6, label='Decadal average SMB (update of Marzeion et al., 2015)')
ax9.set_xticks([1970, 1980, 1990, 2000, 2010, 2015])
ax9.set_xticklabels(('1970', '1980', '1990', '2000', '2010', '2015'))
ax9.xaxis.grid(True)
#ax9.legend(loc='lower left', mode= 'expand', bbox_to_anchor=(0,1.02,1,0.2))
#plt.subplots_adjust(top=0.80, left=0.15)
ax9.legend()

fig92, ax92 = plt.subplots(figsize=(6, 4))
#fig9, ax9 = plt.subplots()
ax92.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax92.set_ylabel('Average glacier-wide SMB (m.w.e. $a^{-1}$)')
ax92.set_xlabel('Year')
#ax92.hlines(total_avg_smb_g, 1967, 2015, color='darkblue', linewidth=7, label='Total average SMB (this study)')
#ax92.hlines(total_avg_smb_marzeion, 1967, 2015, color='darkred', linewidth=4, label='Total average SMB (update of Marzeion et al., 2015)')
ax92.hlines(avg_decadal_smb_g, xmin, xmax, color='steelblue', linewidth=6, label='Decadal average SMB (this study)')
ax92.hlines(avg_decadal_smb_marzeion, xmin, xmax, color='darkgoldenrod', linewidth=6, label='Decadal average SMB (update of Marzeion et al., 2015)')
ax92.set_xticks([1970, 1980, 1990, 2000, 2010, 2015])
ax92.set_xticklabels(('1970', '1980', '1990', '2000', '2010', '2015'))
ax92.xaxis.grid(True)
ax92.legend(loc='lower left', mode= 'expand', bbox_to_anchor=(0,1.02,1,0.2))
plt.subplots_adjust(top=0.80, left=0.15)
#ax9.legend()

# Get colors / markers / functions
#colors = ('C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = np.concatenate((colors, colors))
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass
    
# SMB per massif
fig10, (ax10, ax11) = plt.subplots(2,1, figsize=(6,12))
#fig10.suptitle("Annual glacier-wide SMB by massif")
ax10.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax10.set_ylabel('Glacier-wide SMB (m.w.e. $a^{-1}$)', fontsize=13)
ax10.tick_params(labelsize=11)
#ax10.set_xlabel('Year')

avg_smb_per_massif = copy.deepcopy(smb_massif_template)  
avg_glacier_smb_per_massif = copy.deepcopy(smb_massif_template)

for massif, color, marker in zip(avg_smb_massif, colors, markers):
    format_str = "{color}{marker}".format(color=color, marker=marker)
    # SMB mean per year per massif
    avg_smb_per_massif[massif][0] = np.nanmean(avg_smb_massif[massif])
    # SMB mean per glacier per massif
    finite_mask_glacier = np.isfinite(glacier_smb_per_massif[massif])
    finite_mask_glacier = np.where(finite_mask_glacier == True)[0]
    avg_glacier_smb_per_massif[massif][0] = np.average(np.asarray(glacier_smb_per_massif[massif])[finite_mask_glacier], weights=np.asarray(glacier_area_per_massif[massif])[finite_mask_glacier])
    avg_glacier_smb_per_massif[massif][1] = avg_glacier_smb_per_massif[massif][0]*49
    line101, = ax10.plot(range(1967, 2016), avg_smb_massif[massif], color=color, marker=marker, linewidth=1, label=massif)
    
ax11.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax11.set_ylabel('Cumulative glacier-wide SMB (m.w.e.)', fontsize=13)
ax11.set_xlabel('Year', fontsize=13)
ax11.tick_params(labelsize=11)
for massif, color, marker in zip(avg_smb_massif, colors, markers):
    format_str = "{color}{marker}-".format(color=color, marker=marker)
    line111, = ax11.plot(range(1967, 2016), np.cumsum(avg_smb_massif[massif]), color=color, marker=marker, linewidth=1, label=massif)

handles, labels = ax11.get_legend_handles_labels()
#ax11.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
#ax11.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=7, mode="expand", borderaxespad=0.)
ax11.legend(prop={'size': 12})
plt.tight_layout()
plt.subplots_adjust(hspace = 0.08)

# Let's print the average glacier-wide SMB per massif
print("Average glacier-wide SMB per massif: " + str(avg_glacier_smb_per_massif))
print('Average decadal glacier-wide SMB: " + ' + str(avg_decadal_smb_g))

#import pdb; pdb.set_trace()

plt.show()
