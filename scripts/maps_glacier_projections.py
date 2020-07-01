# -*- coding: utf-8 -*-
"""

@author: Jordi Bolibar

PROCESSING DATA FOR MAPS OF FRENCH ALPINE GLACIERS EVOLUTION (2015-2100)

"""

## Dependencies: ##
import matplotlib.pyplot as plt
import matplotlib as mpl
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

mpl.rcParams['axes.xmargin'] = 0.02
mpl.rcParams['axes.ymargin'] = 0.1
mpl.rcParams['font.sans-serif'] = 'DejaVu'

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

massif_order_S_N = {'Names': ['1. Ubaye','2. Champsaur','3. Pelvoux', '4. Oisans', '5. Grandes Rousses','6. Belledonne', '7. Haute Maurienne', '8. Vanoise', '9. Haute Tarentaise', '10. Mont Blanc', '11. Chablais'], 
                    'ID': [21,19,16,15,12,8,11,10,6,3, 1], 'ID_str':[]}

subset_massifs = np.array([3, 19, 16, 15])

zmean_2015_massifs_S_N, zmean_2100_massifs_S_N = [],[]
for ID in massif_order_S_N['ID']:
    idx = np.where(zmean_2015_massif_groups.massif_ID == ID)[0][0]
    zmean_2015_massifs_S_N.append(zmean_2015_massif_groups.data[idx])
    zmean_2100_massifs_S_N.append(zmean_2100_massif_groups.data[idx])
    massif_order_S_N['ID_str'].append(str(ID))
    
zmean_2015_massifs_S_N = np.asarray(zmean_2015_massifs_S_N)
zmean_2100_massifs_S_N = np.asarray(zmean_2100_massifs_S_N)

###  PLOTS   ######

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
fig1.savefig("C:\\Jordi\\PhD\\Publications\\Third article\\maps\\" + "zmean_massif.pdf")

#plt.show()

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



