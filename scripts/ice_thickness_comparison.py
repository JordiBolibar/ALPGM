# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:18:24 2018

@author: bolibarj
"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import numpy.polynomial.polynomial as poly

######   FILE PATHS    #######
    
# Folders     
workspace = 'C:\\Jordi\\PhD\\Python\\'
#path_obs = 'C:\\Jordi\\PhD\\Data\\Obs\\'
path_smb = workspace + 'ALPGM\\SMB\\'
#path_glacier_coordinates = workspace + 'ALPGM\\glacier_coordinates\\'     
path_delta_h_param = workspace + "ALPGM\\delta_h_param\\"
path_glacier_shapefiles = workspace + 'ALPGM\\glacier_shapefiles\\'
path_glacier_rasters = workspace + 'ALPGM\\glacier_rasters\\dh_2015'   
path_ice_thick_obs = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\Obs_Argentiere-Mer_de_Glace\\csv\\'
path_slopes = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\slopes_MontBlanc\\csv\\'
path_new_arg = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\Obs_Argentiere-Mer_de_Glace\\Argentiere\\'
path_tongue_mdg = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\Obs_Argentiere-Mer_de_Glace\\ablation_mer_de_glace\\csv\\'
path_new_mdg = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\Obs_Argentiere-Mer_de_Glace\\ablation_mer_de_glace\\new_sample_points\\Results\\'
path_flowline_mdg = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\Obs_Argentiere-Mer_de_Glace\\ablation_mer_de_glace\\new_sample_points\\flowline\\'
path_saintsorlin = 'C:\\Jordi\\PhD\\GIS\\Saint_Sorlin\\Results\\'
path_saintsorlin_flowline = 'C:\\Jordi\\PhD\\GIS\\Saint_Sorlin\\Results\\flowline_25\\'
path_glacier_blanc = 'C:\\Jordi\\PhD\\GIS\\Glacier Blanc\\Results\\'

path_consensus_argentiere = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\Comparison_Consensus_Data\\Argentiere\\'
path_consensus_mer_de_glace = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\Comparison_Consensus_Data\\Mer_de_Glace\\'
path_consensus_saint_sorlin = 'C:\\Jordi\\PhD\\GIS\\dh_huss\\Comparison_Consensus_Data\\Saint_Sorlin\\'

plt.close('all')

##### FUNCTIONS #####

# Interval must be a multiple of 25m
def get_slope_at_25m(altitude, interval):
    n = interval/25
    slope = []
    i = 0
    for i_altitude in altitude[n*2:]:
        i_1_altitude = altitude[i]
        slope.append(np.arctan([(i_altitude - i_1_altitude)/(n*2*25)])[0]*(-180/np.pi))
        i = i+1
    slope = np.asarray(slope)
    return slope
        

# Depth/Altitude points
# Argentiere
obs_argentiere = genfromtxt(path_ice_thick_obs + 'depth_argentiere_merged.csv', delimiter=',', skip_header=1)
dh_Huss_argentiere = genfromtxt(path_ice_thick_obs + 'dh_Huss_argentiere_2003.csv', delimiter=',', skip_header=1) 
distance_slope_argentiere = genfromtxt(path_new_arg + 'argentiere_distance_slope_25_flowline.csv', delimiter=';', skip_header=1) 

depth_obs_argentiere = obs_argentiere[:,5]
altitude_obs_argentiere = obs_argentiere[:,2]
depth_Huss_argentiere = dh_Huss_argentiere[:,1]
altitude_Huss_argentiere = dh_Huss_argentiere[:,0]

argentiere_srtm_25 = distance_slope_argentiere[:,0]
argentiere_huss_25 = distance_slope_argentiere[:,1]
argentiere_obs_25 = distance_slope_argentiere[:,2]
argentiere_dem2003_25 = distance_slope_argentiere[:,3]

# We compute the slope at a certain distance step
slope_100_argentiere = get_slope_at_25m(argentiere_srtm_25, 50)
slope_200_argentiere = get_slope_at_25m(argentiere_srtm_25, 100)
slope_300_argentiere = get_slope_at_25m(argentiere_srtm_25, 150)
slope_400_argentiere = get_slope_at_25m(argentiere_srtm_25, 200)

# Argentiere 25m flowline intervals
argentiere_consensus_25 = genfromtxt(path_consensus_argentiere + 'argentiere_consensus_2003.csv', delimiter=';', skip_header=1)
argentiere_obs_25 = genfromtxt(path_consensus_argentiere + 'argentiere_obs_2013.csv', delimiter=';', skip_header=1)
argentiere_srtm_25 = genfromtxt(path_consensus_argentiere + 'argentiere_srtm_2003.csv', delimiter=';', skip_header=1)

argentiere_consensus_25 = np.asarray(argentiere_consensus_25)
argentiere_obs_25 = np.asarray(argentiere_obs_25)
argentiere_srtm_25 = np.asarray(argentiere_srtm_25)

#print("argentiere_consensus_25: " + str(argentiere_consensus_25))
#print("argentiere_obs_25: " + str(argentiere_obs_25))

arg_start = np.where(argentiere_obs_25[0,0] == argentiere_consensus_25)[0][0]
arg_end = np.where(argentiere_obs_25[-1,-2] == argentiere_consensus_25)[0][0]

argentiere_consensus_25 = argentiere_consensus_25[arg_start:arg_end+1]

arg_start = np.where(argentiere_obs_25[0,0] == argentiere_srtm_25)[0][0]
arg_end = np.where(argentiere_obs_25[-1,-2] == argentiere_srtm_25)[0][0]

argentiere_srtm_25 = argentiere_srtm_25[arg_start:arg_end+1]

argentiere_consensus_25 = argentiere_consensus_25[:,1]
argentiere_obs_25 = argentiere_obs_25[:,1]
argentiere_srtm_25 = argentiere_srtm_25[:,1]

obs_argentiere = genfromtxt(path_ice_thick_obs + 'depth_argentiere_merged.csv', delimiter=',', skip_header=1)
dh_Huss_argentiere = genfromtxt(path_ice_thick_obs + 'dh_Huss_argentiere_2003.csv', delimiter=',', skip_header=1) 

# Mer de Glace
obs_mer_de_glace = genfromtxt(path_ice_thick_obs + 'depth_mdg_2016.csv', delimiter=',', skip_header=1) 
obs_tongue_mdg1 = genfromtxt(path_tongue_mdg + 'depth_altitude_tongue_mdg_2001_CVincent.csv', delimiter=',', skip_header=1) 
dh_Huss_mer_de_glace = genfromtxt(path_ice_thick_obs + 'dh_Huss_mdg_2003.csv', delimiter=',', skip_header=1)
dh_Huss_tongue_mdg = genfromtxt(path_tongue_mdg + 'dh_Huss_altitude_tongue_mdg_2003.csv', delimiter=',', skip_header=1)

depth_obs_mer_de_glace = obs_mer_de_glace[:,5]
altitude_obs_mer_de_glace = obs_mer_de_glace[:,2]
depth_obs_tongue_mdg = obs_tongue_mdg1[:,0]
altitude_obs_tongue_mdg = obs_tongue_mdg1[:,1]
depth_Huss_mer_de_glace = dh_Huss_mer_de_glace[:,1]
altitude_Huss_mer_de_glace = dh_Huss_mer_de_glace[:,0]
depth_Huss_tongue_mdg = dh_Huss_tongue_mdg[:,1]
altitude_Huss_tongue_mdg = dh_Huss_tongue_mdg[:,0]

# Mer de Glace 25m flowline intervals
mdg_consensus_25 = genfromtxt(path_consensus_mer_de_glace + 'mer_de_glace_consensus_2003.csv', delimiter=';', skip_header=1)
mdg_obs_25 = genfromtxt(path_consensus_mer_de_glace + 'mer_de_glace_obs_tongue_2003.csv', delimiter=';', skip_header=1)
mdg_srtm_25 = genfromtxt(path_consensus_mer_de_glace + 'mer_de_glace_srtm_2003.csv', delimiter=';', skip_header=1)

mdg_consensus_25 = np.asarray(mdg_consensus_25)
mdg_obs_25 = np.asarray(mdg_obs_25)

#print("mdg_consensus_25: " + str(mdg_consensus_25))
#print("mdg_obs_25: " + str(mdg_obs_25))

mdg_start = np.where(mdg_obs_25[0,0] == mdg_consensus_25[:,0])[0][0]
mdg_end = np.where(mdg_obs_25[-1,-2] == mdg_consensus_25[:,0])[0][0]
mdg_consensus_25 = mdg_consensus_25[mdg_start:mdg_end+1]

mdg_start = np.where(mdg_obs_25[0,0] == mdg_srtm_25[:,0])[0][0]
mdg_end = np.where(mdg_obs_25[-1,-2] == mdg_srtm_25[:,0])[0][0]
mdg_srtm_25 = mdg_srtm_25[mdg_start:mdg_end+1]

mdg_consensus_25 = mdg_consensus_25[:,1]
mdg_obs_25 = mdg_obs_25[:,1]
mdg_srtm_25 = mdg_srtm_25[:,1]

print("mdg_consensus_25.shape: " + str(mdg_consensus_25.shape))
print("mdg_obs_25.shape: " + str(mdg_obs_25.shape))

# We compute the slope at a certain distance step
slope_100_mdg = get_slope_at_25m(mdg_srtm_25, 50)
slope_200_mdg = get_slope_at_25m(mdg_srtm_25, 100)
slope_300_mdg = get_slope_at_25m(mdg_srtm_25, 150)
slope_400_mdg = get_slope_at_25m(mdg_srtm_25, 200)

# New samples Mer de Glace
mdg_all = genfromtxt(path_new_mdg + 'mdg_all.csv', delimiter=';', skip_header=1) 

mdg_obs_2003 = mdg_all[:,0]
mdg_huss_2003 = mdg_all[:,1]
mdg_slope_2003 = mdg_all[:,2]
mdg_altitudes_2003 = mdg_all[:,3]
mdg_srtm_2000 = mdg_all[:,4]


# Saint Sorlin
dh_Huss_obs_slope_SaintSorlin = genfromtxt(path_saintsorlin + 'SaintSorlin_all.csv', delimiter=';', skip_header=1)
distance_slope_SaintSorlin = genfromtxt(path_saintsorlin_flowline + 'ss_flowline_25.csv', delimiter=';', skip_header=1)

altitude_SaintSorlin = dh_Huss_obs_slope_SaintSorlin[:,0]
slope_2011_SaintSorlin = dh_Huss_obs_slope_SaintSorlin[:,1]
slope_2003_SaintSorlin = dh_Huss_obs_slope_SaintSorlin[:,2]
depth_Obs_SaintSorlin = dh_Huss_obs_slope_SaintSorlin[:,3]
depth_Huss_SaintSorlin = dh_Huss_obs_slope_SaintSorlin[:,4]

ss_srtm2000_25 = distance_slope_SaintSorlin[:,1]
ss_dem2003_25 = distance_slope_SaintSorlin[:,2]
ss_huss_25 = distance_slope_SaintSorlin[:,3]
ss_obs2000_25 = distance_slope_SaintSorlin[:,4]

# We compute the slope at a certain distance step
slope_100_ss = get_slope_at_25m(ss_srtm2000_25, 50)
slope_200_ss = get_slope_at_25m(ss_srtm2000_25, 100)
slope_300_ss = get_slope_at_25m(ss_srtm2000_25, 150)
slope_400_ss = get_slope_at_25m(ss_srtm2000_25, 200)

# Mer de Glace 25m flowline intervals
saint_sorlin_consensus_25 = genfromtxt(path_consensus_saint_sorlin + 'saint_sorlin_consensus_2003.csv', delimiter=';', skip_header=1)
saint_sorlin_obs_25 = genfromtxt(path_consensus_saint_sorlin + 'saint_sorlin_obs_2000.csv', delimiter=';', skip_header=1)
saint_sorlin_srtm_25 = genfromtxt(path_consensus_saint_sorlin + 'saint_sorlin_srtm_2003.csv', delimiter=';', skip_header=1)

saint_sorlin_consensus_25 = saint_sorlin_consensus_25[:,1]
saint_sorlin_obs_25 = saint_sorlin_obs_25[:,1]
saint_sorlin_srtm_25 = saint_sorlin_srtm_25[:,1]

# Glacier Blanc
GlacierBlanc_all = genfromtxt(path_glacier_blanc + 'GlacierBlanc_all.csv', delimiter=';', skip_header=1)

glacierblanc_altitude_2002 = GlacierBlanc_all[:,3]
glacierblanc_obs_2002 = GlacierBlanc_all[:,4]
glacierblanc_huss_2003 = GlacierBlanc_all[:,5]
glacierblanc_slope_2011 = GlacierBlanc_all[:,6]


# Depth/Slope points
# Mer de Glace
obs_slope_argentiere = genfromtxt(path_slopes + 'depth_argentiere_slope.csv', delimiter=',', skip_header=1)
obs_slope_mer_de_glace = genfromtxt(path_slopes + 'depth_mdg_slope_2016.csv', delimiter=',', skip_header=1) 
obs_tongue_mdg2 = genfromtxt(path_tongue_mdg + 'depth_slope_tongue_mdg_2001_CVincent.csv', delimiter=',', skip_header=1) 
# Argentiere
dh_Huss_slope_argentiere = genfromtxt(path_slopes + 'dh_Huss_argentiere_slope.csv', delimiter=',', skip_header=1) 
dh_Huss_slope_mer_de_glace = genfromtxt(path_slopes + 'dh_Huss_mdg_slope.csv', delimiter=',', skip_header=1)
dh_Huss_slope_tongue_mdg = genfromtxt(path_tongue_mdg + 'dh_Huss_slope_tongue_mdg_2003.csv', delimiter=',', skip_header=1)


depth_obs_argentiere2 = obs_slope_argentiere[:,0]
slope_obs_argentiere = obs_slope_argentiere[:,1]
depth_obs_mer_de_glace2 = obs_slope_mer_de_glace[:,0]
slope_obs_mer_de_glace = obs_slope_mer_de_glace[:,1]
depth_obs_tongue_mdg = obs_tongue_mdg2[:,0]
slope_obs_tongue_mdg = obs_tongue_mdg2[:,1]
depth_Huss_argentiere2 = dh_Huss_slope_argentiere[:,0]
slope_Huss_argentiere = dh_Huss_slope_argentiere[:,1]
depth_Huss_mer_de_glace2 = dh_Huss_slope_mer_de_glace[:,0]
slope_Huss_mer_de_glace = dh_Huss_slope_mer_de_glace[:,1]
depth_Huss_tongue_mdg = dh_Huss_slope_tongue_mdg[:,1]
slope_Huss_tongue_mdg = dh_Huss_slope_tongue_mdg[:,0]



########################   PLOTS   #########################################

nfigure = 1

#############  Ice depth vs Altitude  #######################

# Argentiere radar points
plt.figure(nfigure)
plt.scatter(altitude_obs_argentiere, depth_obs_argentiere, s=2, label="Observations (2013-2015)")
p_obs_arg1 = poly.Polynomial.fit(altitude_obs_argentiere, depth_obs_argentiere, 6)
x_p_obs_arg1 = np.asarray(*p_obs_arg1.linspace(n=altitude_obs_argentiere.size)[:1]).flatten()
y_p_obs_arg1 = np.asarray(*p_obs_arg1.linspace(n=depth_obs_argentiere.size)[1:]).flatten()
plt.plot(x_p_obs_arg1, y_p_obs_arg1, linewidth=4)
plt.scatter(altitude_Huss_argentiere, depth_Huss_argentiere, s=2, label="HF12 (2003)", color='mediumorchid')
p_huss_arg1 = poly.Polynomial.fit(altitude_Huss_argentiere, depth_Huss_argentiere, 6)
x_p_huss_arg1 = np.asarray(*p_huss_arg1.linspace(n=altitude_Huss_argentiere.size)[:1]).flatten()
y_p_huss_arg1 = np.asarray(*p_huss_arg1.linspace(n=depth_Huss_argentiere.size)[1:]).flatten()
plt.plot(x_p_huss_arg1, y_p_huss_arg1, linewidth=4, color='mediumorchid')
plt.title("Argentiere", fontsize=20, y=1.05)
plt.xlabel("Altitude (m)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

# Argentiere 25m flowline intervals
plt.figure(nfigure)
plt.scatter(argentiere_srtm_25, argentiere_obs_25, s=3, label="Observations (2013-2015)")
p_obs_arg2 = poly.Polynomial.fit(argentiere_srtm_25, argentiere_obs_25, 20)
x_p_obs_arg2 = np.asarray(*p_obs_arg2.linspace(n=argentiere_srtm_25.size)[:1]).flatten()
y_p_obs_arg2 = np.asarray(*p_obs_arg2.linspace(n=argentiere_obs_25.size)[1:]).flatten()
plt.plot(x_p_obs_arg2, y_p_obs_arg2, linewidth=4)
plt.scatter(argentiere_srtm_25, argentiere_consensus_25, s=3, label="F19 (2003)", color='#49bad3ff')
p_huss_arg2 = poly.Polynomial.fit(argentiere_srtm_25, argentiere_consensus_25, 20)
x_p_huss_arg2 = np.asarray(*p_huss_arg2.linspace(n=argentiere_srtm_25.size)[:1]).flatten()
y_p_huss_arg2 = np.asarray(*p_huss_arg2.linspace(n=argentiere_consensus_25.size)[1:]).flatten()
plt.plot(x_p_huss_arg2, y_p_huss_arg2, linewidth=4, color='#49bad3ff')
plt.title("Argentiere (25m intervals)", fontsize=20, y=1.05)
plt.xlabel("Altitude (m)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(fontsize='x-large')
plt.tick_params(labelsize=16)
plt.show()
nfigure = nfigure+1


# Mer de Glace
plt.figure(nfigure)
altitude_obs_mer_de_glace = np.append(altitude_obs_mer_de_glace, altitude_obs_tongue_mdg)
depth_obs_mer_de_glace = np.append(depth_obs_mer_de_glace, depth_obs_tongue_mdg)
altitude_Huss_mer_de_glace = np.append(altitude_Huss_mer_de_glace, altitude_Huss_tongue_mdg)
depth_Huss_mer_de_glace = np.append(depth_Huss_mer_de_glace, depth_Huss_tongue_mdg)
plt.scatter(altitude_obs_mer_de_glace, depth_obs_mer_de_glace, s=2, label="Obs(Tongue: 2011, Accum: 2016)")
p_obs_mdg1 = poly.Polynomial.fit(altitude_obs_mer_de_glace, depth_obs_mer_de_glace, 2)
x_p_obs_mdg1 = np.asarray(*p_obs_mdg1.linspace(n=altitude_obs_mer_de_glace.size)[:1]).flatten()
y_p_obs_mdg1 = np.asarray(*p_obs_mdg1.linspace(n=depth_obs_mer_de_glace.size)[1:]).flatten()
plt.plot(x_p_obs_mdg1, y_p_obs_mdg1, linewidth=4)
plt.scatter(altitude_Huss_mer_de_glace, depth_Huss_mer_de_glace, s=2, label="HF12 (2003)", color='mediumorchid')
p_huss_mdg1 = poly.Polynomial.fit(altitude_Huss_mer_de_glace, depth_Huss_mer_de_glace, 2)
x_p_huss_mdg1 = np.asarray(*p_huss_mdg1.linspace(n=altitude_Huss_mer_de_glace.size)[:1]).flatten()
y_p_huss_mdg1 = np.asarray(*p_huss_mdg1.linspace(n=depth_Huss_mer_de_glace.size)[1:]).flatten()
plt.plot(x_p_huss_mdg1, y_p_huss_mdg1, linewidth=4, color='mediumorchid')
plt.title("Mer de Glace", fontsize=20, y=1.05)
plt.xlabel("Altitude (m)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

# Mer de Glace tongue (tongue new sampling)
plt.figure(nfigure)
plt.scatter(mdg_altitudes_2003, mdg_obs_2003, s=2, label="Observations (2003)")
p_obs_mdg_2003 = poly.Polynomial.fit(mdg_altitudes_2003, mdg_obs_2003, 6)
x_p_obs_mdg_2003 = np.asarray(*p_obs_mdg_2003.linspace(n=mdg_altitudes_2003.size)[:1]).flatten()
y_p_obs_mdg_2003 = np.asarray(*p_obs_mdg_2003.linspace(n=mdg_obs_2003.size)[1:]).flatten()
plt.plot(x_p_obs_mdg_2003, y_p_obs_mdg_2003, linewidth=4)
plt.scatter(mdg_srtm_2000, mdg_huss_2003, s=2, label="HF12 (2003)", color='mediumorchid')
p_huss_mdg_2003 = poly.Polynomial.fit(mdg_srtm_2000, mdg_huss_2003, 6)
x_p_huss_mdg_2003 = np.asarray(*p_huss_mdg_2003.linspace(n=mdg_srtm_2000.size)[:1]).flatten()
y_p_huss_mdg_2003 = np.asarray(*p_huss_mdg_2003.linspace(n=mdg_huss_2003.size)[1:]).flatten()
plt.plot(x_p_huss_mdg_2003, y_p_huss_mdg_2003, linewidth=4, color='mediumorchid')
plt.title("Mer de Glace (tongue)", fontsize=20, y=1.05)
plt.xlabel("Altitude (m)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

print("mdg_srtm_25.shape: " + str(mdg_srtm_25.shape))
print("mdg_obs_25.shape: " + str(mdg_obs_25.shape))
print("mdg_consensus_25.shape: " + str(mdg_consensus_25.shape))

# Mer de Glace tongue (25m intervals)
plt.figure(nfigure)
plt.scatter(mdg_srtm_25, mdg_obs_25, s=3, label="Observations (2003)")
p_obs_mdg_25 = poly.Polynomial.fit(mdg_srtm_25, mdg_obs_25, 20)
x_p_obs_mdg_25 = np.asarray(*p_obs_mdg_25.linspace(n=mdg_srtm_25.size)[:1]).flatten()
y_p_obs_mdg_25 = np.asarray(*p_obs_mdg_25.linspace(n=mdg_obs_25.size)[1:]).flatten()
plt.plot(x_p_obs_mdg_25, y_p_obs_mdg_25, linewidth=4)
plt.scatter(mdg_srtm_25, mdg_consensus_25, s=3, label="F19 (2003)", color='#49bad3ff')
p_huss_mdg_25 = poly.Polynomial.fit(mdg_srtm_25, mdg_consensus_25, 20)
x_p_huss_mdg_25 = np.asarray(*p_huss_mdg_25.linspace(n=mdg_srtm_25.size)[:1]).flatten()
y_p_huss_mdg_25 = np.asarray(*p_huss_mdg_25.linspace(n=mdg_consensus_25.size)[1:]).flatten()
plt.plot(x_p_huss_mdg_25, y_p_huss_mdg_25, linewidth=4, color='#49bad3ff')
plt.title("Mer de Glace (tongue, 25m intervals)", fontsize=20, y=1.05)
plt.xlabel("Altitude (m)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(fontsize='x-large')
plt.tick_params(labelsize=16)
plt.show()
nfigure = nfigure+1


# Saint Sorlin 25m flowline intervals
plt.figure(nfigure)
plt.scatter(saint_sorlin_srtm_25, saint_sorlin_obs_25, s=2, label="Observations/SRTM (2000)")
p_obs_ss = poly.Polynomial.fit(saint_sorlin_srtm_25, saint_sorlin_obs_25, 6)
x_p_obs_ss = np.asarray(*p_obs_ss.linspace(n=saint_sorlin_srtm_25.size)[:1]).flatten()
y_p_obs_ss = np.asarray(*p_obs_ss.linspace(n=saint_sorlin_obs_25.size)[1:]).flatten()
plt.plot(x_p_obs_ss, y_p_obs_ss, linewidth=4)
plt.scatter(saint_sorlin_srtm_25, saint_sorlin_consensus_25, s=2, label="F19 (2003)", color='#49bad3ff')
p_huss_ss = poly.Polynomial.fit(saint_sorlin_srtm_25, saint_sorlin_consensus_25, 6)
x_p_huss_ss = np.asarray(*p_huss_ss.linspace(n=saint_sorlin_srtm_25.size)[:1]).flatten()
y_p_huss_ss = np.asarray(*p_huss_ss.linspace(n=saint_sorlin_consensus_25.size)[1:]).flatten()
plt.plot(x_p_huss_ss, y_p_huss_ss, linewidth=4, color='#49bad3ff')
plt.title("Saint Sorlin (25m intervals)", fontsize=20, y=1.05)
plt.xlabel("Altitude (m)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(fontsize='x-large')
plt.tick_params(labelsize=16)
plt.show()
nfigure = nfigure+1

# Glacier Blanc
plt.figure(nfigure)
plt.scatter(glacierblanc_altitude_2002, glacierblanc_obs_2002, s=2, label="Observations (2002)")
p_obs_gb = poly.Polynomial.fit(glacierblanc_altitude_2002, glacierblanc_obs_2002, 2)
x_p_obs_gb = np.asarray(*p_obs_gb.linspace(n=glacierblanc_altitude_2002.size)[:1]).flatten()
y_p_obs_gb = np.asarray(*p_obs_gb.linspace(n=glacierblanc_obs_2002.size)[1:]).flatten()
plt.plot(x_p_obs_gb, y_p_obs_gb, linewidth=4)
plt.scatter(glacierblanc_altitude_2002, glacierblanc_huss_2003, s=2, label="HF12 (2003)", color='mediumorchid')
p_huss_gb = poly.Polynomial.fit(glacierblanc_altitude_2002, glacierblanc_huss_2003, 2)
x_p_huss_gb = np.asarray(*p_huss_gb.linspace(n=glacierblanc_altitude_2002.size)[:1]).flatten()
y_p_huss_gb = np.asarray(*p_huss_gb.linspace(n=glacierblanc_huss_2003.size)[1:]).flatten()
plt.plot(x_p_huss_gb, y_p_huss_gb, linewidth=4, color='mediumorchid')
plt.title("Glacier Blanc", fontsize=20, y=1.05)
plt.xlabel("Altitude (m)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

##############  Ice depth vs Slope  #########################

# Argentiere
plt.figure(nfigure)
plt.scatter(slope_obs_argentiere, depth_obs_argentiere2, s=2, label="Observations (2013-2015)")
p_obs_arg = poly.Polynomial.fit(slope_obs_argentiere, depth_obs_argentiere2, 2)
x_p_obs_arg = np.asarray(*p_obs_arg.linspace(n=slope_obs_argentiere.size)[:1]).flatten()
y_p_obs_arg = np.asarray(*p_obs_arg.linspace(n=depth_obs_argentiere2.size)[1:]).flatten()
plt.plot(x_p_obs_arg, y_p_obs_arg, linewidth=4)
plt.scatter(slope_Huss_argentiere, depth_Huss_argentiere2, s=2, label="HF12 (2003)", color='mediumorchid')
p_huss_arg = poly.Polynomial.fit(slope_Huss_argentiere, depth_Huss_argentiere2, 2)
x_p_huss_arg = np.asarray(*p_huss_arg.linspace(n=slope_Huss_argentiere.size)[:1]).flatten()
y_p_huss_arg = np.asarray(*p_huss_arg.linspace(n=depth_Huss_argentiere2.size)[1:]).flatten()
plt.plot(x_p_huss_arg, y_p_huss_arg, linewidth=4, color='mediumorchid')
plt.title("Glacier d'Argentiere (avg flowline slope = 7.7 deg)", fontsize=20, y=1.05)
plt.xlabel("Slope (deg)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

n = argentiere_obs_25.size - slope_200_argentiere.size
print("slope_200_argentiere.shape: " + str(slope_200_argentiere.shape))
print("argentiere_obs_25[n/2:-n/2].shape: " + str(argentiere_obs_25[n/2:-n/2].shape))

# Argentiere 25m
n = argentiere_obs_25.size - slope_200_argentiere.size
plt.figure(nfigure)
plt.scatter(slope_200_argentiere, argentiere_obs_25[n/2:-n/2], s=3, label="Observations (2013-2015)")
p_obs_arg = poly.Polynomial.fit(slope_200_argentiere, argentiere_obs_25[n/2:-n/2], 2)
x_p_obs_arg = np.asarray(*p_obs_arg.linspace(n=slope_200_argentiere.size)[:1]).flatten()
y_p_obs_arg = np.asarray(*p_obs_arg.linspace(n=argentiere_obs_25[n/2:-n/2].size)[1:]).flatten()
plt.plot(x_p_obs_arg, y_p_obs_arg, linewidth=4)
plt.scatter(slope_200_argentiere, argentiere_consensus_25[n/2:-n/2], s=3, label="F19 (2003)", color='#49bad3ff')
p_huss_arg = poly.Polynomial.fit(slope_200_argentiere, argentiere_consensus_25[n/2:-n/2], 2)
x_p_huss_arg = np.asarray(*p_huss_arg.linspace(n=slope_200_argentiere.size)[:1]).flatten()
y_p_huss_arg = np.asarray(*p_huss_arg.linspace(n=argentiere_consensus_25[n/2:-n/2].size)[1:]).flatten()
plt.plot(x_p_huss_arg, y_p_huss_arg, linewidth=4, color='#49bad3ff')
plt.title("Argentiere (25m intervals)", fontsize=20, y=1.05)
plt.xlabel("Slope (deg)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(fontsize='x-large')
plt.tick_params(labelsize=16)
plt.show()
nfigure = nfigure+1


# Mer de Glace
plt.figure(nfigure)
slope_obs_mer_de_glace = np.append(slope_obs_mer_de_glace, slope_obs_tongue_mdg)
depth_obs_mer_de_glace2 = np.append(depth_obs_mer_de_glace2, depth_obs_tongue_mdg)
slope_Huss_mer_de_glace = np.append(slope_Huss_mer_de_glace, slope_Huss_tongue_mdg)
depth_Huss_mer_de_glace2 = np.append(depth_Huss_mer_de_glace2, depth_Huss_tongue_mdg)
plt.scatter(slope_obs_mer_de_glace, depth_obs_mer_de_glace, s=2, label="Obs(Tongue: 2011, Accum: 2016)")
p_obs_mdg = poly.Polynomial.fit(slope_obs_mer_de_glace, depth_obs_mer_de_glace, 2)
x_p_obs_mdg = np.asarray(*p_obs_mdg.linspace(n=slope_obs_mer_de_glace.size)[:1]).flatten()
y_p_obs_mdg = np.asarray(*p_obs_mdg.linspace(n=depth_obs_mer_de_glace.size)[1:]).flatten()
plt.plot(x_p_obs_mdg, y_p_obs_mdg, linewidth=4)
plt.scatter(slope_Huss_mer_de_glace, depth_Huss_mer_de_glace, s=2, label="HF12 (2003)", color='mediumorchid')
p_huss_mdg = poly.Polynomial.fit(slope_Huss_mer_de_glace, depth_Huss_mer_de_glace, 2)
x_p_huss_mdg = np.asarray(*p_huss_mdg.linspace(n=slope_Huss_mer_de_glace.size)[:1]).flatten()
y_p_huss_mdg = np.asarray(*p_huss_mdg.linspace(n=depth_Huss_mer_de_glace.size)[1:]).flatten()
plt.plot(x_p_huss_mdg, y_p_huss_mdg, linewidth=4, color='mediumorchid')
plt.title("Mer de Glace", fontsize=20, y=1.05)
plt.xlabel("Slope (deg)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

# Mer de Glace tongue (new sampling)
plt.figure(nfigure)
plt.scatter(mdg_slope_2003, mdg_obs_2003, s=2, label="Observations (2003)")
p_obs_mdg_2003 = poly.Polynomial.fit(mdg_slope_2003, mdg_obs_2003, 2)
x_p_obs_mdg_2003 = np.asarray(*p_obs_mdg_2003.linspace(n=mdg_slope_2003.size)[:1]).flatten()
y_p_obs_mdg_2003 = np.asarray(*p_obs_mdg_2003.linspace(n=mdg_obs_2003.size)[1:]).flatten()
plt.plot(x_p_obs_mdg_2003, y_p_obs_mdg_2003, linewidth=4)
plt.scatter(mdg_slope_2003, mdg_huss_2003, s=2, label="HF12 (2003)", color='mediumorchid')
p_huss_mdg_2003 = poly.Polynomial.fit(mdg_slope_2003, mdg_huss_2003, 2)
x_p_huss_mdg_2003 = np.asarray(*p_huss_mdg_2003.linspace(n=mdg_slope_2003.size)[:1]).flatten()
y_p_huss_mdg_2003 = np.asarray(*p_huss_mdg_2003.linspace(n=mdg_huss_2003.size)[1:]).flatten()
plt.plot(x_p_huss_mdg_2003, y_p_huss_mdg_2003, linewidth=4, color='mediumorchid')
plt.title("Mer de Glace (tongue) (avg flowline slope = 6.5 deg)", fontsize=20, y=1.05)
plt.xlabel("Slope (deg)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

# Mer de Glace tongue (new sampling) 25m intervals
plt.figure(nfigure)
n = mdg_obs_25.size - slope_200_mdg.size
plt.scatter(slope_200_mdg, mdg_obs_25[n/2:-n/2], s=3, label="Observations (2003)")
p_obs_mdg_2003 = poly.Polynomial.fit(slope_200_mdg, mdg_obs_25[n/2:-n/2], 2)
x_p_obs_mdg_2003 = np.asarray(*p_obs_mdg_2003.linspace(n=slope_200_mdg.size)[:1]).flatten()
y_p_obs_mdg_2003 = np.asarray(*p_obs_mdg_2003.linspace(n=mdg_obs_25[n/2:-n/2].size)[1:]).flatten()
plt.plot(x_p_obs_mdg_2003, y_p_obs_mdg_2003, linewidth=4)
plt.scatter(slope_200_mdg, mdg_consensus_25[n/2:-n/2], s=3, label="F19 (2003)", color='#49bad3ff')
p_huss_mdg_2003 = poly.Polynomial.fit(slope_200_mdg, mdg_consensus_25[n/2:-n/2], 2)
x_p_huss_mdg_2003 = np.asarray(*p_huss_mdg_2003.linspace(n=slope_200_mdg.size)[:1]).flatten()
y_p_huss_mdg_2003 = np.asarray(*p_huss_mdg_2003.linspace(n=mdg_consensus_25[n/2:-n/2].size)[1:]).flatten()
plt.plot(x_p_huss_mdg_2003, y_p_huss_mdg_2003, linewidth=4, color='#49bad3ff')
plt.title("Mer de Glace (tongue, 25m intervals)", fontsize=20, y=1.05)
plt.xlabel("Slope (deg)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(fontsize='x-large')
plt.tick_params(labelsize=16)
plt.show()
nfigure = nfigure+1

# Saint Sorlin
plt.figure(nfigure)
n = saint_sorlin_obs_25.size - slope_200_ss.size
plt.scatter(slope_200_ss, saint_sorlin_obs_25[n/2:-n/2], s=3, label="Observations/SRTM (2000)")
plt.scatter(slope_200_ss, saint_sorlin_consensus_25[n/2:-n/2], s=3, label="F19 (2003)", color='#49bad3ff')

p_obs_ss = poly.Polynomial.fit(slope_200_ss, saint_sorlin_obs_25[n/2:-n/2], 4)
x_p_obs_ss = np.asarray(*p_obs_ss.linspace(n=slope_200_ss.size)[:1]).flatten()
y_p_obs_ss = np.asarray(*p_obs_ss.linspace(n=saint_sorlin_obs_25[n/2:-n/2].size)[1:]).flatten()
plt.plot(x_p_obs_ss, y_p_obs_ss, linewidth=4)
p_huss_ss = poly.Polynomial.fit(slope_200_ss, saint_sorlin_consensus_25[n/2:-n/2], 4)
x_p_huss_ss = np.asarray(*p_huss_ss.linspace(n=slope_200_ss.size)[:1]).flatten()
y_p_huss_ss = np.asarray(*p_huss_ss.linspace(n=saint_sorlin_consensus_25[n/2:-n/2].size)[1:]).flatten()
plt.plot(x_p_huss_ss, y_p_huss_ss, linewidth=4, color='#49bad3ff')

plt.title("Saint Sorlin (25m intervals)", fontsize=20, y=1.05)
plt.xlabel("Slope (deg)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(fontsize='x-large')
plt.tick_params(labelsize=16)
plt.show()
nfigure = nfigure+1

# Glacier Blanc
plt.figure(nfigure)
plt.scatter(glacierblanc_slope_2011, glacierblanc_obs_2002, s=2, label="Observations (2002)")
plt.scatter(glacierblanc_slope_2011, glacierblanc_huss_2003, s=2, label="HF12 (2003)", color='mediumorchid')

p_obs_gb = poly.Polynomial.fit(glacierblanc_slope_2011, glacierblanc_obs_2002, 2)
x_p_obs_gb = np.asarray(*p_obs_gb.linspace(n=glacierblanc_slope_2011.size)[:1]).flatten()
y_p_obs_gb = np.asarray(*p_obs_gb.linspace(n=glacierblanc_obs_2002.size)[1:]).flatten()
plt.plot(x_p_obs_gb, y_p_obs_gb, linewidth=4)
p_huss_gb = poly.Polynomial.fit(glacierblanc_slope_2011, glacierblanc_huss_2003, 2)
x_p_huss_gb = np.asarray(*p_huss_gb.linspace(n=glacierblanc_slope_2011.size)[:1]).flatten()
y_p_huss_gb = np.asarray(*p_huss_gb.linspace(n=glacierblanc_huss_2003.size)[1:]).flatten()
plt.plot(x_p_huss_gb, y_p_huss_gb, linewidth=4, color='mediumorchid')

plt.title("Glacier Blanc (avg flowline slope = 15.3 deg)", fontsize=20, y=1.05)
plt.xlabel("Slope (deg)", fontsize=20)
plt.ylabel("Ice thickness (m)", fontsize=20)
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

################# Ice depth vs Ice depth  ################################

plt.figure(nfigure)
plt.scatter(depth_Huss_argentiere, depth_obs_argentiere, s=2)
plt.title("Argentiere", fontsize=20, y=1.05)
plt.xlabel("Depth HF12 (m)", fontsize=20)
plt.ylabel("Depth Observations (m)", fontsize=20)
axes = plt.gca()
axes.set_xlim([0, 500])
axes.set_ylim([0, 500])
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1
 
plt.figure(nfigure)
plt.scatter(depth_Huss_mer_de_glace, depth_obs_mer_de_glace, s=2)
plt.title("Mer de Glace", fontsize=20, y=1.05)
plt.xlabel("Depth HF12 (m)", fontsize=20)
plt.ylabel("Depth Observations (m)", fontsize=20)
axes = plt.gca()
axes.set_xlim([0, 400])
axes.set_ylim([0, 400])
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

plt.figure(nfigure)
plt.scatter(mdg_huss_2003, mdg_obs_2003, s=2)
plt.title("Mer de Glace (tongue)", fontsize=20, y=1.05)
plt.xlabel("Depth HF12 (m)", fontsize=20)
plt.ylabel("Depth Observations (m)", fontsize=20)
axes = plt.gca()
axes.set_xlim([0, 400])
axes.set_ylim([0, 400])
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

plt.figure(nfigure)
plt.scatter(depth_Huss_SaintSorlin, depth_Obs_SaintSorlin, s=2)
plt.title("Saint Sorlin", fontsize=20, y=1.05)
plt.xlabel("Depth HF12 (m)", fontsize=20)
plt.ylabel("Depth Observations (m)", fontsize=20)
axes = plt.gca()
axes.set_xlim([0, 120])
axes.set_ylim([0, 120])
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1

plt.figure(nfigure)
plt.scatter(glacierblanc_huss_2003, glacierblanc_obs_2002, s=2)
plt.title("Glacier Blanc", fontsize=20, y=1.05)
plt.xlabel("Depth HF12 (m)", fontsize=20)
plt.ylabel("Depth Observations (m)", fontsize=20)
axes = plt.gca()
axes.set_xlim([0, 350])
axes.set_ylim([0, 350])
plt.legend(loc=2, fontsize='x-large')
plt.show()
nfigure = nfigure+1


