# -*- coding: utf-8 -*-
"""

@author: Jordi Bolibar
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
# Folders     
workspace = str(Path(os.getcwd()).parent) 
path_glims = os.path.join(workspace, 'glacier_data', 'GLIMS') 
path_smb = os.path.join(workspace, 'glacier_data', 'smb')
path_glacier_evolution = os.path.join(workspace, 'glacier_data', 'glacier_evolution')
path_glacier_coordinates = os.path.join(workspace, 'glacier_data', 'glacier_coordinates')
path_smb_simulations = os.path.join(path_smb, 'smb_simulations')

#######################    FUNCTIONS    ##########################################################



##################################################################################################
        
        
###############################################################################
###                           MAIN                                          ###
###############################################################################


# Open netCDF dataset with glacier evolution projections
ds_glacier_projections = xr.open_dataset(os.path.join(path_glacier_evolution, 'glacier_evolution_2015_2100.nc'))

volume_2015_massif_groups = ds_glacier_projections.volume.sel(member= 'CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5', RCP='45', year=2015).groupby('massif_ID').sum(...)
volume_2100_massif_groups = ds_glacier_projections.volume.sel(member= 'CLMcom-CCLM4-8-17_CNRM-CERFACS-CNRM-CM5', RCP='45', year=2099).groupby('massif_ID').sum(...)

massif_glacier_volume = pd.DataFrame({'ID': volume_2015_massif_groups.massif_ID.data, 'volume_2015': volume_2015_massif_groups.data, 'volume_2100': volume_2100_massif_groups.data})
massif_glacier_volume = massif_glacier_volume.fillna(0)

massif_glacier_volume.to_csv(os.path.join(path_glacier_evolution, 'plots', 'data', 'glacier_volume_2015_2100.csv'), sep=';')

import pdb; pdb.set_trace()

