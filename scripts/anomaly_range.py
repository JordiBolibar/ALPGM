# -*- coding: utf-8 -*-
"""
Created on Tue Jul 06 19:05:49 2021

@author: bolibarj
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path
import xarray as xr
import proplot as plot

# Folders     
workspace = str(Path(os.getcwd()).parent) 
path_glacier_evolution = os.path.join(workspace, 'glacier_data', 'glacier_evolution')
path_smb = os.path.join(workspace, 'glacier_data', 'smb')
path_smb_function = os.path.join(path_smb, 'smb_function')
path_safran_forcings = os.path.join(path_smb_function, 'SAFRAN')
path_glacier_evolution = os.path.join(workspace, 'glacier_data', 'glacier_evolution')

with open(os.path.join(path_safran_forcings, 'season_meteo.txt'), 'rb') as season_f:
    season_meteo = np.load(season_f,  allow_pickle=True)[()]
    
# Retrieves the mean meteo values to compute the anomalies
def get_meteo_references(season_meteo_SMB, glimsIDs):
    glacier_CPDDs = season_meteo_SMB['CPDD']
    glacier_winter_snow = season_meteo_SMB['winter_snow']
    glacier_summer_snow = season_meteo_SMB['summer_snow']   
    
    CPDD_refs, w_snow_refs, s_snow_refs = [],[],[]
    for glimsID in glimsIDs:
        for cpdd, w_snow, s_snow in zip(glacier_CPDDs, glacier_winter_snow, glacier_summer_snow):
            if(cpdd['GLIMS_ID'] == glimsID):
                CPDD_refs.append(cpdd['Mean'])
                w_snow_refs.append(w_snow['Mean'])
                s_snow_refs.append(s_snow['Mean']) 
            
    return CPDD_refs, w_snow_refs, s_snow_refs

# Open netCDF dataset with glacier evolution projections
ds_glacier_projections = xr.open_dataset(os.path.join(path_glacier_evolution, 'glacier_evolution_2015_2100.nc'))

# We get the glacier climatic references
CPDD_refs, w_snow_refs, s_snow_refs = get_meteo_references(season_meteo, ds_glacier_projections.GLIMS_ID)

# We group the glacier projections by GLIMS ID
CPDD_ID = ds_glacier_projections.CPDD.groupby("GLIMS_ID")
w_snow_ID = ds_glacier_projections.w_snowfall.groupby("GLIMS_ID")
s_snow_ID = ds_glacier_projections.s_snowfall.groupby("GLIMS_ID")

CPDD_anom, w_snow_anom, s_snow_anom = {"mean":[], "max":[], "min":[]},{"mean":[], "max":[], "min":[]},{"mean":[], "max":[], "min":[]}

count = 1
for CPDD_ref, w_snow_ref, s_snow_ref, CPDD_group, w_snow_group, s_snow_group in zip(CPDD_refs, w_snow_refs, s_snow_refs, CPDD_ID, w_snow_ID, s_snow_ID):
    print(CPDD_group[0], " - #", count, "\n")
    CPDD_anom["mean"].append(CPDD_group[1].mean(...).data - CPDD_ref)
    w_snow_anom["mean"].append(w_snow_group[1].mean(...).data - w_snow_ref)
    s_snow_anom["mean"].append(s_snow_group[1].mean(...).data - s_snow_ref)
    
    CPDD_anom["max"].append(CPDD_group[1].max(...).data - CPDD_ref)
    w_snow_anom["max"].append(w_snow_group[1].max(...).data - w_snow_ref)
    s_snow_anom["max"].append(s_snow_group[1].max(...).data - s_snow_ref)
    
    CPDD_anom["min"].append(CPDD_group[1].min(...).data - CPDD_ref)
    w_snow_anom["min"].append(w_snow_group[1].min(...).data - w_snow_ref)
    s_snow_anom["min"].append(s_snow_group[1].min(...).data - s_snow_ref)
    
    count +=1
    
import pdb; pdb.set_trace()
    
CPDD_avg_max_anom = CPDD_anom["max"].mean()

with open(os.path.join(path_glacier_evolution, "CPDD_projected_anomalies"), 'wb') as cpdd_f:
    np.save(cpdd_f, CPDD_anom)

with open(os.path.join(path_glacier_evolution, "w_snow_projected_anomalies"), 'wb') as wsnow_f:
    np.save(wsnow_f, w_snow_anom)
    
with open(os.path.join(path_glacier_evolution, "s_snow_projected_anomalies"), 'wb') as ssnow_f:
    np.save(ssnow_f, s_snow_anom)


#ds_glacier_projections.s_snowfall.groupby("RCP").max(...).mean()