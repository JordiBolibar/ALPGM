# -*- coding: utf-8 -*-
"""

@author: Jordi Bolibar

PROCESSING DATA FOR MAPS OF FRENCH ALPINE GLACIERS EVOLUTION (2015-2100)

"""

## Dependencies: ##
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import proplot as plot
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path
import pickle
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from shutil import copyfile
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

mpl.rcParams['axes.xmargin'] = 0.02
mpl.rcParams['axes.ymargin'] = 0.1
#mpl.rcParams['font.sans-serif'] = 'DejaVu'

# Folders     
workspace = str(Path(os.getcwd()).parent) 
path_glims = os.path.join(workspace, 'glacier_data', 'GLIMS') 
path_smb = os.path.join(workspace, 'glacier_data', 'smb')
path_glacier_evolution = os.path.join(workspace, 'glacier_data', 'glacier_evolution')
path_glacier_coordinates = os.path.join(workspace, 'glacier_data', 'glacier_coordinates')
path_smb_simulations = os.path.join(path_smb, 'smb_simulations')
path_glacier_thickness = os.path.join(workspace, 'glacier_data', 'glacier_rasters', 'glacier_thickness', 'thickness_tif', 'glacier_evolution')
path_individual_thickness = os.path.join(path_glacier_thickness, 'projections', 'FORCING_CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5_RCP45_alp_2005080106_2100080106', '1')
path_start_thickness = os.path.join(path_glacier_thickness, 'SAFRAN', '1')
path_gis_projections = 'C:\Jordi\PhD\GIS\glacier_projections'

glims_2003 = pd.read_csv(os.path.join(path_glims, 'GLIMS_2003_ID_massif.csv'), sep=';')

#######################    FUNCTIONS    ##########################################################



##################################################################################################
        
        
###############################################################################
###                           MAIN                                          ###
###############################################################################

# Open netCDF dataset with glacier evolution projections
ds_glacier_projections = xr.open_dataset(os.path.join(path_glacier_evolution, 'glacier_evolution_2015_2100.nc'))

# Processing xarray dataset with mean values for RCP 4.5

volume_2015_massif_groups, volume_2100_massif_groups, total_volume_2100 = [],[],[]
for member in ds_glacier_projections.member:
    volume_2015_massif_groups.append(ds_glacier_projections.volume.sel(member=member, RCP='45', year=2015).groupby('massif_ID').sum(...))
    volume_2100_massif_groups.append(ds_glacier_projections.volume.sel(member=member, RCP='45', year=2099).groupby('massif_ID').sum(...))
    
    total_volume_2100.append(ds_glacier_projections.volume.sel(member=member, RCP='45', year=2099).sum(...).data)
    print("\nRCP 4.5 " + "- member: " + str(member))
    print("Volume 2100 (all massifs): " + str(total_volume_2100[-1]))
    
# Compute the mean by massif 
volume_2015_massif_groups = np.mean(volume_2015_massif_groups, axis=0)
volume_2100_massif_groups = np.mean(volume_2100_massif_groups, axis=0)

print("\nAverage volume 2100 (all massifs): " + str(np.sum(volume_2100_massif_groups)))

CPDD_2015_massif_groups = ds_glacier_projections.CPDD.sel(RCP='45', year=2015).groupby('massif_ID').mean(...)
CPDD_2100_massif_groups = ds_glacier_projections.CPDD.sel(RCP='45', year=2099).groupby('massif_ID').mean(...)

zmean_2015_massif_groups = ds_glacier_projections.zmean.sel(RCP='45', year=2015).groupby('massif_ID').mean(...)
zmean_2100_massif_groups = ds_glacier_projections.zmean.sel(RCP='45', year=2099).groupby('massif_ID').mean(...)

snowfall_45_massif_groups = ds_glacier_projections.snowfall.sel(RCP='45').groupby('massif_ID').mean(dim=['GLIMS_ID', 'member'])
cpdd_45_massif_groups = ds_glacier_projections.CPDD.sel(RCP='45').groupby('massif_ID').mean(dim=['GLIMS_ID', 'member'])
MB_45_massif_groups = ds_glacier_projections.MB.sel(RCP='45').groupby('massif_ID').mean(dim=['GLIMS_ID', 'member'])

massif_order_S_N = {'Names': ['1. Ubaye','2. Champsaur','3. Pelvoux', '4. Oisans', '5. Grandes Rousses','6. Belledonne', '7. Haute Maurienne', '8. Vanoise', '9. Haute Tarentaise', '10. Mont Blanc', '11. Chablais'], 
                    'ID': [21,19,16,15,12,8,11,10,6,3, 1], 'ID_str':[]}

subset_massifs = np.array([3, 19, 16, 15])

zmean_2015_massifs_S_N, zmean_2100_massifs_S_N = [],[]
snowfall_45_massif_S_N, cpdd_45_massif_S_N, MB_45_massif_S_N = [],[],[]

for ID in massif_order_S_N['ID']:
    idx = np.where(zmean_2015_massif_groups.massif_ID == ID)[0][0]
    zmean_2015_massifs_S_N.append(zmean_2015_massif_groups.data[idx])
    zmean_2100_massifs_S_N.append(zmean_2100_massif_groups.data[idx])
    snowfall_45_massif_S_N.append(snowfall_45_massif_groups.values[idx,:])
    cpdd_45_massif_S_N.append(cpdd_45_massif_groups.values[idx,:])
    MB_45_massif_S_N.append(MB_45_massif_groups.values[idx,:])
    
    massif_order_S_N['ID_str'].append(str(ID))
    
zmean_2015_massifs_S_N = np.asarray(zmean_2015_massifs_S_N)
zmean_2100_massifs_S_N = np.asarray(zmean_2100_massifs_S_N)
snowfall_45_massif_S_N = np.asarray(snowfall_45_massif_S_N)
cpdd_45_massif_S_N = np.asarray(cpdd_45_massif_S_N)
MB_45_massif_S_N = np.asarray(MB_45_massif_S_N)

######  Glacier survival factors statistical analysis   ########

volume_2015_glaciers = ds_glacier_projections.volume.sel(RCP='45', year=2015).groupby('GLIMS_ID').mean(...)
volume_2100_glaciers = ds_glacier_projections.volume.sel(RCP='45', year=2099).groupby('GLIMS_ID').mean(...)

slope_2015_glaciers = ds_glacier_projections.slope20.sel(RCP='45', year=2015).groupby('GLIMS_ID').mean(...)
slope_2100_glaciers = ds_glacier_projections.slope20.sel(RCP='45', year=2099).groupby('GLIMS_ID').mean(...)

area_2015_massifs = ds_glacier_projections.area.sel(RCP='45', year=2015).groupby('massif_ID').mean(dim=['member']).sum(dim='GLIMS_ID')
area_2100_massifs = ds_glacier_projections.area.sel(RCP='45', year=2099).groupby('massif_ID').mean(dim=['member']).sum(dim='GLIMS_ID')

avg_slopes = (slope_2015_glaciers + slope_2100_glaciers)/2

y = (volume_2100_glaciers/volume_2015_glaciers)*100
#y = volume_2100_glaciers

max_alt, lat, lon, x = [],[],[],[]
for glacier, glacier_slope in zip(y.GLIMS_ID, avg_slopes):
    max_alt.append(glims_2003['MAX_PixelV'][glims_2003['GLIMS_ID'].values == glacier.GLIMS_ID.values[()][:14]].values[0])
    lat.append(glims_2003['y_coord'][glims_2003['GLIMS_ID'].values == glacier.GLIMS_ID.values[()][:14]].values[0])
    lon.append(glims_2003['x_coord'][glims_2003['GLIMS_ID'].values == glacier.GLIMS_ID.values[()][:14]].values[0])
    
#    x.append(np.array([max_alt[-1], lat[-1], lon[-1], glacier_slope.values]))
    x.append(np.array([max_alt[-1], lat[-1], lon[-1]]))

y = np.where(y > 100, np.nan, y)
max_alt = np.asarray(max_alt)
lat = np.asarray(lat)
lon = np.asarray(lon)
x = np.asarray(x)
mask = np.isfinite(y)

#plt.scatter(max_alt[mask], y[mask])
#plt.show()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = sm.OLS(y[mask], x_scaled[mask,:])
model_fit = model.fit()

print(model_fit.summary())

import pdb; pdb.set_trace()

###  PLOTS   ######

#### Zmean massif for map   #######

x = np.linspace(0, zmean_2015_massifs_S_N.size, num=11, endpoint=True)
xnew = np.linspace(0, zmean_2015_massifs_S_N.size, num=101, endpoint=True)

fsmooth_2015 = interp1d(x, zmean_2015_massifs_S_N, kind='cubic')
fsmooth_2100 = interp1d(x, zmean_2100_massifs_S_N, kind='cubic')

zmean_2100_massifs_S_N_gone = np.nan_to_num(zmean_2100_massifs_S_N, nan=2600)
zmean_2100_massifs_S_N_gone = np.where(zmean_2100_massifs_S_N_gone != 2600, np.nan, zmean_2100_massifs_S_N_gone)

fig1, axs1 = plot.subplots(ncols=1, nrows=1, width=8, height=2)
axs1.format(xtickminor=False)

zline = axs1.plot(xnew, fsmooth_2015(xnew), c='brown', linewidth=1, zorder=1, label='2015')
axs1.fill_between(xnew, 2500, fsmooth_2015(xnew), zorder=1, facecolor='brown', alpha=0.4)
z2100 = axs1.scatter(x, zmean_2100_massifs_S_N, c='darkred', marker="^", size=70, zorder=10, label='2100')
axs1.scatter(x, zmean_2100_massifs_S_N_gone, c='darkred', marker="x", size=70, zorder=10)
#axs1.plot(xnew, fsmooth_2100(xnew), c='darkred')
plt.xticks(x, massif_order_S_N['Names'], rotation='270')
axs1.xaxis.tick_top()
axs1.spines['top'].set_visible(False)
axs1.spines['right'].set_visible(False)
axs1.spines['bottom'].set_visible(False)
#axs1.spines['left'].set_visible(False)

axs1.legend(zline, loc='r', ncols=1, frame=False)
axs1.legend(z2100, loc='r', ncols=1, frame=False)

#plt.yticks(rotation='90')
#plt.subplots_adjust(top=0.15)
#fig1.tight_layout(False)
fig1.savefig("C:\\Jordi\\PhD\\Publications\\Third article\\Bolibar_et_al_Science_Advances\\maps\\" + "zmean_massif.pdf")

#plt.show()

#### Climate stripes per  massifs  ######

#fig2, ax2 = plot.subplots(ncols=2, axwidth=3, share=3, aspect=0.75, wspace='7em')

fig2, ax2 = plot.subplots([[1, 1], [2, 3]], ncols=2, nrows=2, axwidth=7, aspect=3.3, share=3)

ax2.format(
#        abc=True, abcloc='lr',
        ylocator=1,
        ytickminor=False,
        yticklabelloc='left'
)

mb_stripes = ax2[0].pcolormesh(MB_45_massif_groups.year.values, range(1,12), MB_45_massif_S_N, cmap='vikO_r', cmap_kw={'right': 0.7})
fig2.colorbar(mb_stripes, ax=ax2[0])
ax2[0].set_title('Annual glacier-wide MB (m.w.e.)')
ax2[0].set_ylabel('Glacierized massif')
ax2[0].set_xlabel('Year')

snow_stripes = ax2[1].pcolormesh(snowfall_45_massif_groups.year.values, range(1,12), snowfall_45_massif_S_N, cmap='lapaz_r')
fig2.colorbar(snow_stripes, ax=ax2[1])
ax2[1].set_title('Annual snowfall (mm)')
#ax2[1].set_ylabel('Glacierized massif')
ax2[1].set_xlabel('Year')

snow_stripes = ax2[2].pcolormesh(cpdd_45_massif_groups.year.values, range(1,12), cpdd_45_massif_S_N, cmap='Matter')
fig2.colorbar(snow_stripes, ax=ax2[2])
ax2[2].set_title('Annual CPDD')
ax2[2].set_ylabel('Glacierized massif')
ax2[2].set_xlabel('Year')

plt.show()

#### Raster processing for maps  #############################3

#########################################################################3

### Process raster data
thick_raster_files = np.asarray(os.listdir(path_individual_thickness))
thick_start_raster_files = np.asarray(os.listdir(path_start_thickness))

year_marks = np.array([2015,2030,2040,2050,2060,2070,2080,2090,2099])

for raster in thick_raster_files:
    extension = raster[-3:]
    if(extension == 'tif'):
        ID = float(raster[-14:-9])
        year_file = int(raster[-8:-4:])
            
        # If glacier is in Mont-Blanc or Écrins
        file_massif_ID = glims_2003[glims_2003['ID'] == ID]['massif_SAF'].values
        if(file_massif_ID.size != 0):
            if(np.any(file_massif_ID[0] == subset_massifs)):
                # We copy the files
                if(np.any(year_file == year_marks)):
                    path_year = os.path.join(path_gis_projections, str(year_file))
                    
                    if(not os.path.exists(path_year)):
                        os.makedirs(path_year)
                    copyfile(os.path.join(path_individual_thickness, raster), os.path.join(path_year, raster))

year_start = 2014

for raster in thick_start_raster_files:
    extension = raster[-3:]
    if(extension == 'tif'):
        ID = float(raster[-14:-9])
        year_file = int(raster[-8:-4:])
            
        # If glacier is in Mont-Blanc or Écrins
        file_massif_ID = glims_2003[glims_2003['ID'] == ID]['massif_SAF'].values
        if(file_massif_ID.size != 0):
            if(np.any(file_massif_ID[0] == subset_massifs)):
                # We copy the files
                if(year_file == year_start):
                    path_year = os.path.join(path_gis_projections, str(year_file))
                    
                    if(not os.path.exists(path_year)):
                        os.makedirs(path_year)
                    copyfile(os.path.join(path_start_thickness, raster), os.path.join(path_year, raster))

print("\nRaster ice thickness data filtered")

# Exporting data
massif_glacier_volume = pd.DataFrame({'ID': ds_glacier_projections.massif_ID.data, 'volume_2015': volume_2015_massif_groups.data, 'volume_2100': volume_2100_massif_groups.data})
massif_glacier_volume = massif_glacier_volume.fillna(0)

massif_glacier_volume.to_csv(os.path.join(path_glacier_evolution, 'plots', 'data', 'glacier_volume_2015_2100.csv'), sep=';')



