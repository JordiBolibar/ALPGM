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
from pathlib import Path
from matplotlib.lines import Line2D
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

all_glacier_smb  = []
annual_avg_smb_ = []
annual_avg_smb_big_glaciers_ = []
annual_avg_smb_small_glaciers_ = []
annual_avg_smb_very_small_glaciers_ = []
annual_avg_area, annual_avg_slope = [],[]

for year_idx in range(0, 49):
    annual_avg_smb_.append([])
    annual_avg_smb_big_glaciers_.append([])
    annual_avg_smb_small_glaciers_.append([])
    annual_avg_smb_very_small_glaciers_.append([])
    annual_avg_area.append([])
    annual_avg_slope.append([])
    
massifs_safran = {'1':'Chablais', '2':'Aravis','3':'Mont-Blanc','5':'Beaufortain','6':'Haute-Tarantaise','10':'Vanoise',
                  '9':'Maurienne','11':'Haute-Maurienne','8':'Belledonne','12':'Grandes-Rousses','15':'Oisans','16':'Pelvoux',
                  '13':'Thabor', '19':'Champsaur','18':'Devoluy','21':'Ubaye'}

smb_massif_template = {'Chablais':copy.deepcopy(annual_avg_smb_), 'Aravis':copy.deepcopy(annual_avg_smb_),'Mont-Blanc':copy.deepcopy(annual_avg_smb_),
                       'Beaufortain':copy.deepcopy(annual_avg_smb_), 'Haute-Tarantaise':copy.deepcopy(annual_avg_smb_),'Vanoise':copy.deepcopy(annual_avg_smb_), 
                       'Maurienne':copy.deepcopy(annual_avg_smb_),'Haute-Maurienne':copy.deepcopy(annual_avg_smb_), 'Belledonne':copy.deepcopy(annual_avg_smb_),
                       'Grandes-Rousses':copy.deepcopy(annual_avg_smb_),'Oisans':copy.deepcopy(annual_avg_smb_),'Pelvoux':copy.deepcopy(annual_avg_smb_), 
                       'Thabor':copy.deepcopy(annual_avg_smb_), 'Champsaur':copy.deepcopy(annual_avg_smb_), 'Ubaye':copy.deepcopy(annual_avg_smb_)}

smb_massif = copy.deepcopy(smb_massif_template)
mean_smb_glaciers, mean_area_glaciers, mean_slope_glaciers = np.zeros(661), np.zeros(661), np.zeros(661)
    
fig1, ax1 = plt.subplots()
ax1.set_ylabel('Glacier-wide SMB (m.w.e)')
ax1.set_xlabel('Year')
ax1.set_title("Annual glacier-wide SMB of all French alpine glaciers")

fig2, ax2 = plt.subplots()
ax2.set_ylabel('Cumulative glacier-wide SMB (m.w.e)')
ax2.set_xlabel('Year')
ax2.set_title("Cumulative glacier-wide SMB of all French alpine glaciers")


# Iterate all glaciers with the full simulated period
glacier_idx, glacier_idx_2015 = 0, 0
glaciers_not_2015 = 0
big_glaciers, small_glaciers, very_small_glaciers = 0, 0, 0
big_glaciers_2015, small_glaciers_2015, very_small_glaciers_2015 = 0, 0, 0
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
    
    
#    import pdb; pdb.set_trace()
    current_massif = massifs_safran[str(glacier_info['massif_SAFRAN'])[:-2]]
    
    all_glacier_smb.append(np.asarray(smb_glacier))
    
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
        linewidth = 0.1
    elif(area_glacier_i < 0.5):
        linewidth = 0.2
    elif(area_glacier_i < 5):
        linewidth = 0.5
    else:
        linewidth = 0.6
    
    if(area_glacier_i < 0.1):
        mean_smb_glaciers[glacier_idx] = np.nanmean(smb_glacier)
    else:
        mean_smb_glaciers[glacier_idx] = np.nanmean(smb_glacier)
    mean_area_glaciers[glacier_idx] = np.nanmean(area_glacier)
    mean_slope_glaciers[glacier_idx] = slope_glacier[0]
    
    line1, = ax1.plot(range(1967, 2016), smb_glacier, linewidth=linewidth)
    line2, = ax2.plot(range(1967, 2016), np.cumsum(smb_glacier), linewidth=linewidth)
    
    big_glacier, small_glacier, very_small_glacier = False, False, False
    
    for year_idx in range(0, 49):
        if(not np.isnan(smb_glacier[year_idx])):
            annual_avg_smb_[year_idx].append(smb_glacier[year_idx])
            annual_avg_area[year_idx].append(area_glacier_i)
            smb_massif[current_massif][year_idx].append(smb_glacier[year_idx])
            
            if(area_glacier_i >= 1):
                annual_avg_smb_big_glaciers_[year_idx].append(smb_glacier[year_idx])
                big_glacier = True
            elif(area_glacier_i > 0.1):
                annual_avg_smb_small_glaciers_[year_idx].append(smb_glacier[year_idx])
                small_glacier = True
            else:
                annual_avg_smb_very_small_glaciers_[year_idx].append(smb_glacier[year_idx] + 0.2)
                very_small_glacier = True
    
    # All glaciers indexes
    glacier_idx = glacier_idx+1
    if(big_glacier):
        big_glaciers = big_glaciers+1
    elif(small_glacier):
        small_glaciers = small_glaciers+1
    elif(very_small_glacier):
        very_small_glaciers = very_small_glaciers+1
    
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

#import pdb; pdb.set_trace()
  
# All glaciers
all_glacier_smb = np.asarray(all_glacier_smb) 
a_avg_smb, a_avg_smb_big, a_avg_smb_small, a_avg_smb_v_small = [],[],[],[]
for avg_smb, avg_smb_big, avg_smb_medium, avg_smb_small, avg_area in zip(annual_avg_smb_, annual_avg_smb_big_glaciers_, annual_avg_smb_small_glaciers_, annual_avg_smb_very_small_glaciers_, annual_avg_area):
    a_avg_smb.append(np.average(avg_smb, weights=avg_area))
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
a_avg_smb_big = np.asarray(a_avg_smb_big)
a_avg_smb_small = np.asarray(a_avg_smb_small)
a_avg_smb_v_small = np.asarray(a_avg_smb_v_small)

all_glacier_smb = np.asarray(all_glacier_smb)

# We compute the correlation of each glacier with respect the weighted mean
glacier_correlation = []
for glacier_smb in all_glacier_smb:
    if(glacier_smb.size > 37):
        glacier_correlation.append(np.corrcoef(a_avg_smb, glacier_smb)[0,1]**2)
glacier_correlation = np.asarray(glacier_correlation)

print("\nMax SMB common variance: " + str(glacier_correlation.max()))
print("\nMin SMB common variance: " + str(glacier_correlation.min()))
print("\nAverage SMB common variance: " + str(glacier_correlation.mean()))

line1, = ax1.plot(range(1967, 2016), a_avg_smb, linewidth=2, c='black', label='Area weighted mean')
line2, = ax2.plot(range(1967, 2016), np.cumsum(a_avg_smb), linewidth=2, c='black', label='Area weighted mean')

ax1.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax1.legend()
ax2.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax2.legend()

print("\nNumber of glaciers disappeared between 2003 and 2015: " + str(glaciers_not_2015))

#import pdb; pdb.set_trace()

### Average plots for glacier size
ax3 = plt.subplot(1,2,1)
plt.title("Annual glacier-wide SMB of French alpine glaciers depending on size", y=1.03, x=1.1)
ax3.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax3.set_ylabel('Glacier-wide SMB (m.w.e)')
ax3.set_xlabel('Year')
line14, = ax3.plot(range(1967, 2016), a_avg_smb_v_small, linewidth=1, label='Glaciers < 0.1 km$^2$', c='darkred')
line13, = ax3.plot(range(1967, 2016), a_avg_smb_small, linewidth=1, label='Glaciers 0.1 - 1 km$^2$', c='C1')
line12, = ax3.plot(range(1967, 2016), a_avg_smb_big, linewidth=1, label='Glaciers > 1 km$^2$', c='C0')
ax3.legend()

ax4 = plt.subplot(1,2,2)
ax4.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax4.set_ylabel('Cumulative glacier-wide SMB (m.w.e)')
ax4.set_xlabel('Year')
line24, = ax4.plot(range(1967, 2016), np.cumsum(a_avg_smb_v_small), linewidth=1, label='Glaciers < 0.1 km$^2$', c='darkred')
line23, = ax4.plot(range(1967, 2016), np.cumsum(a_avg_smb_small), linewidth=1, label='Glaciers 0.1 - 1 km$^2$', c='C1')
line22, = ax4.plot(range(1967, 2016), np.cumsum(a_avg_smb_big), linewidth=1, label='Glaciers > 1 km$^2$', c='C0')
ax4.legend()

print("\nMean annual glacier-wide SMB per glacier size: ")
print("Big glaciers: " + str(np.average(a_avg_smb_big)))
print("Medium glaciers: " + str(np.average(a_avg_smb_small)))
print("Small glaciers: " + str(np.average(a_avg_smb_v_small)))

print("\nArea weighted mean annual glacier-wide SMB: " + str(np.average(a_avg_smb)))

### Average SMB with uncertainties
avg_a_uncertainty = 0.32 # m.w.e (MAE)

fig5, ax5 = plt.subplots()
ax5.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax5.set_ylabel('Glacier-wide SMB (m.w.e)')
ax5.set_xlabel('Year')
ax5.set_title("Mean annual glacier-wide SMB of all French alpine glaciers")
ax5.fill_between(range(1967, 2016), a_avg_smb-avg_a_uncertainty, a_avg_smb+avg_a_uncertainty, facecolor = "red", alpha=0.3)
line12, = ax5.plot(range(1967, 2016), a_avg_smb, linewidth=2, label='Area weighted average')
ax5.legend()

fig6, ax6 = plt.subplots()
ax6.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax6.set_ylabel('Cumulative glacier-wide SMB (m.w.e)')
ax6.set_xlabel('Year')
ax6.set_title("Cumulative weighted mean annual glacier-wide SMB of all French alpine glaciers")
ax6.fill_between(range(1967, 2016), np.cumsum(a_avg_smb-avg_a_uncertainty), np.cumsum(a_avg_smb+avg_a_uncertainty), facecolor = "red", alpha=0.3)
line22, = ax6.plot(range(1967, 2016), np.cumsum(a_avg_smb), linewidth=2, label='Area weighted average')
ax6.legend()

# Scatter plots

ax7 = plt.subplot(1,2,1)
plt.title("Average annual glacier-wide SMB vs Surface area and Slope", y=1.03, x=1.1)
ax7.set_xlabel('Glacier surface area (km$^2$)')
ax7.set_ylabel('Annual glacier-wide SMB (m.w.e)')
#ax7.set_title("Average annual glacier-wide SMB vs Glacier surface area", y=1.03)
ax7.scatter(mean_area_glaciers, mean_smb_glaciers, s=4, alpha=0.7)
log_area = np.log10(mean_area_glaciers)
ax7.plot(np.unique(mean_area_glaciers), np.poly1d(np.polyfit(log_area, mean_smb_glaciers, 1))(np.unique(log_area)), c='darkred')
ax7.set_xscale('log')


ax8 = plt.subplot(1,2,2)
ax8.set_xlabel('Lowermost 20% altitudinal range slope (°)')
#ax8.set_ylabel('Annual glacier-wide SMB (m.w.e)')
#ax8.set_title("Average annual glacier-wide SMB vs Glacier slope")
ax8.scatter(mean_slope_glaciers, mean_smb_glaciers, s=4, alpha=0.7)
ax8.plot(np.unique(mean_slope_glaciers), np.poly1d(np.polyfit(mean_slope_glaciers, mean_smb_glaciers, 1))(np.unique(mean_slope_glaciers)), c='darkred')

#import pdb; pdb.set_trace()


### Decade average SMB
avg_smb_70s = a_avg_smb[3:13].mean()
avg_smb_80s = a_avg_smb[13:23].mean()
avg_smb_90s = a_avg_smb[23:33].mean()
avg_smb_00s = a_avg_smb[33:43].mean()
avg_smb_10s = a_avg_smb[43:].mean()

avg_decadal_smb = np.array([avg_smb_70s, avg_smb_80s, avg_smb_90s, avg_smb_00s, avg_smb_10s])
xmin = np.array([1970, 1980, 1990, 2000, 2010])
xmax = np.array([1980, 1990, 2000, 2010, 2015])
total_avg_smb = a_avg_smb.mean()

fig9, ax9 = plt.subplots()
ax9.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
#ax9.axvline(x=2015, color='grey', linewidth=0.7, linestyle='-')
ax9.set_ylabel('Average glacier-wide SMB (m.w.e)')
ax9.set_xlabel('Year')
ax9.set_title("Average glacier-wide SMB per decade")
ax9.hlines(total_avg_smb, 1967, 2015, color='darkred', linewidth=6, label='Total average SMB')
ax9.hlines(avg_decadal_smb, xmin, xmax, color='C0', linewidth=6, label='Decadal average SMB')
ax9.set_xticks([1970, 1980, 1990, 2000, 2010, 2015])
ax9.set_xticklabels(('1970', '1980', '1990', '2000', '2010', '2015'))
ax9.xaxis.grid(True)
ax9.legend()

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
ax10 = plt.subplot(1,2,1)
plt.title("Annual glacier-wide SMB by massif", y=1.03, x=1.1)
ax10.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax10.set_ylabel('Glacier-wide SMB (m.w.e)')
ax10.set_xlabel('Year')

for massif, color, marker in zip(avg_smb_massif, colors, markers):
    format_str = "{color}{marker}".format(color=color, marker=marker)
#    import pdb; pdb.set_trace()
    line101, = ax10.plot(range(1967, 2016), avg_smb_massif[massif], color=color, marker=marker, linewidth=1, label=massif)

ax11 = plt.subplot(1,2,2)
ax11.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax11.set_ylabel('Cumulative glacier-wide SMB (m.w.e)')
ax11.set_xlabel('Year')
for massif, color, marker in zip(avg_smb_massif, colors, markers):
    format_str = "{color}{marker}-".format(color=color, marker=marker)
    line111, = ax11.plot(range(1967, 2016), np.cumsum(avg_smb_massif[massif]), color=color, marker=marker, linewidth=1, label=massif)

handles, labels = ax11.get_legend_handles_labels()
ax11.legend()
#ax11.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
#          fancybox=True, shadow=True, ncol=5)
#ax11.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=7, col=2, mode="expand", borderaxespad=0.)

plt.show()
