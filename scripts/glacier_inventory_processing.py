# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:07:32 2019

@author: bolibarj
"""


from numpy import genfromtxt
import os
from pathlib import Path  
import numpy as np

# Folders   
workspace_path = Path(os.getcwd()).parent 
workspace = str(Path(os.getcwd()).parent) + '\\'
root = str(workspace_path.parent) + '\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 
path_sazm = 'C:\\Jordi\\PhD\\GIS\\SPAZM\\'


glims_1967 = genfromtxt(path_glims + 'GLIMS_1967-71.csv', delimiter=';', skip_header=1,  dtype=[('Glacier', '<a50'), ('Latitude', '<f8'), ('Longitude', '<f8'), ('Massif', '<a50'),  ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('Year', '<f8'), ('Perimeter', '<f8'), ('Area', '<f8'),  ('Code_WGI', '<a50'), ('Length', '<f8'), ('MEAN_Pixel', '<f8'), ('Slope', '<f8'), ('Aspect', '<a50'), ('Code', '<a50'), ('BV', '<a50'), ('GLIMS_ID', '<a50')])
glims_2015_L93 = genfromtxt(path_sazm + 'glims_2015_WGI.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Glacier', '<a50'), ('Code_WGI', '<a50'), ('GLIMS_ID', '<a50')])

for glacier_67 in glims_1967:
    found = False
    WGI_ID_67 = glacier_67['Code_WGI'].decode('ascii')
#    print("WGI_ID_67: " + str(WGI_ID_67))
    for glacier_15 in glims_2015_L93:
        WGI_ID_15 = glacier_15['Code_WGI'].decode('ascii')[:11]
#        print("WGI_ID_15: " + str(WGI_ID_15))
        if(WGI_ID_15 == WGI_ID_67):
            glacier_67['GLIMS_ID'] = glacier_15['GLIMS_ID'].decode('ascii')
            found = True
    if(not found):
        print("/!\ Glacier " + str(glacier_67['Glacier']) + " not found")
    else:
        print("Glacier " + str(glacier_67['Glacier']) + " found")

print("Saving processed 1967 file")
#import pdb; pdb.set_trace()
np.savetxt(path_glims + 'GLIMS_1967_GLIMS_ID.csv', glims_1967['GLIMS_ID'], delimiter=";", fmt="%s")
