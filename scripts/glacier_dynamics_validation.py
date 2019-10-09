# -*- coding: utf-8 -*-
"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

VALIDATION OF GLACIER DYNAMICS PARAMETERIZATION

"""

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

workspace = Path(os.getcwd()).parent 
root = str(workspace.parent) + '\\'
workspace = str(workspace) + '\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 
path_areas = workspace + 'glacier_data\\glacier_evolution\\BACKUP\\glacier_area\\SAFRAN\\'
path_area_1 = path_areas + '1\\'
path_area_07 = path_areas + '0.7\\'
path_area_13 = path_areas + '1.3\\'
path_area_delta_110 = path_areas + 'delta_110\\'
path_area_delta_90 = path_areas + 'delta_90\\'

glims_2015 = genfromtxt(path_glims + 'GLIMS_2015.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50')])
glims_rabatel = genfromtxt(path_glims + 'GLIMS_Rabatel_30_2015.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('slope20', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8')])        

################################

ref_areas, sim_areas_1, sim_areas_07, sim_areas_13, sim_areas_delta_110, sim_areas_delta_90 = [],[],[],[],[],[]
sim_area_errors, sim_area_errors_perc = [], []
for glacier in glims_rabatel[:-1]:
    print("\nGlacier: " + str(glacier['Glacier']))
    print("Area: " + str(glacier['Area']))
    area_ref = glacier['Area']
    glims_ID = glacier['GLIMS_ID'].decode('ascii')
    
    print("glims_ID: " + str(glims_ID))
    
#    if(glims_ID == "G006985E45951N"):
#        glims_ID = "G006985E45951N_2"
    
    area_sim_1 = genfromtxt(path_area_1 + str(glims_ID) + '_area.csv', delimiter=';')[-1, 1]
    area_sim_07 = genfromtxt(path_area_07 + str(glims_ID) + '_area.csv', delimiter=';')[-1, 1]
#    if(glims_ID == "G006149E45143N"):
#        glims_ID = "G006149E45143N_2"
    area_sim_13 = genfromtxt(path_area_13 + str(glims_ID) + '_area.csv', delimiter=';')[-1, 1]
    
    area_delta_110 = genfromtxt(path_area_delta_110 + str(glims_ID) + '_area.csv', delimiter=';')[-1, 1]
    area_delta_90 = genfromtxt(path_area_delta_90 + str(glims_ID) + '_area.csv', delimiter=';')[-1, 1]
    
#    import pdb; pdb.set_trace()
    
    print("Simulated area: " + str(area_sim_1))
    
    ref_areas.append(area_ref)
    sim_areas_1.append(area_sim_1)
    sim_areas_07.append(area_sim_07)
    sim_areas_13.append(area_sim_13)
    
    sim_areas_delta_110.append(area_delta_110)
    sim_areas_delta_90.append(area_delta_90)
    
    sim_area_errors.append(area_sim_1 - area_ref)
    sim_area_errors_perc.append((area_sim_1 - area_ref)/area_ref)
    
    print("\nError (%): " + str((area_sim_1 - area_ref)/area_ref))
    
ref_areas = np.asarray(ref_areas)
sim_areas_1 = np.asarray(sim_areas_1)
sim_areas_07 = np.asarray(sim_areas_07)
sim_areas_13 = np.asarray(sim_areas_13)
sim_areas_delta_110 = np.asarray(sim_areas_delta_110)
sim_areas_delta_90 = np.asarray(sim_areas_delta_90)
sim_area_errors = np.asarray(sim_area_errors)
sim_area_errors_perc = np.asarray(sim_area_errors_perc)

print("\nAverage area error (km2): " + str(sim_area_errors.mean()) )
print("\nAverage area error (%): " + str(sim_area_errors_perc.mean()*100) )

### Plot results

#plt.figure(1, figsize=(6,6))
plt.subplot(1,2,1)
#plt.title("Simulated glacier area for 2003-2015", fontsize=16, y=1.03, x=1.1)
plt.ylabel('ALPGM (km$^2)$', fontsize=16)
plt.xlabel('Reference glacier area (km$^2)$', fontsize=16, x=1.1)
sc = plt.scatter(ref_areas, sim_areas_07 , s=200, alpha = 0.5, c='red', edgecolors='none', marker='s', label='Ice thickness -30%')
sc = plt.scatter(ref_areas, sim_areas_13 , s=200, alpha = 0.5, edgecolors='none',  marker='s', label='Ice thickness + 30%')
sc = plt.scatter(ref_areas, sim_areas_1, s=200, alpha = 0.5, c='C2', edgecolors='none',  marker='s', label='Original ice thickness')
sc = plt.scatter(ref_areas, sim_areas_delta_110, s=100, alpha = 0.5, c='C4', edgecolors='none', marker='^', label='Δh+10%')
sc = plt.scatter(ref_areas, sim_areas_delta_90, s=100, alpha = 0.5, c='C5', edgecolors='none', marker='v', label='Δh-10%')
#lineStart = sim_areas_13.min() 
#lineEnd = sim_areas_13.max()  
lineStart = 0
lineEnd = 9 
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
plt.tick_params(labelsize=14)
plt.legend(prop={'size': 12})

plt.subplot(1,2,2)
sc = plt.scatter(ref_areas, sim_areas_07 , s=200, alpha = 0.5, c='red', edgecolors='none',  marker='s', label='Ice thickness -30%')
sc = plt.scatter(ref_areas, sim_areas_13 , s=200, alpha = 0.5, edgecolors='none',  marker='s', label='Ice thickness + 30%')
sc = plt.scatter(ref_areas, sim_areas_1, s=200, alpha = 0.5, c='C2', edgecolors='none',  marker='s', label='Original ice thickness')
sc = plt.scatter(ref_areas, sim_areas_delta_110, s=100, alpha = 0.5, c='C4', edgecolors='none', marker='^', label='Δh+10%')
sc = plt.scatter(ref_areas, sim_areas_delta_90, s=100, alpha = 0.5, c='C5', edgecolors='none', marker='v', label='Δh-10%')
#lineStart = sim_areas_13.min() 
#lineEnd = sim_areas_13.max()  
lineStart = 9
lineEnd = 32
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
plt.tick_params(labelsize=14)
plt.legend(prop={'size': 12})
plt.show()

