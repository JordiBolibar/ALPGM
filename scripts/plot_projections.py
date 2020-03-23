# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:30:22 2020

@author: bolibarj
"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import copy
from pathlib import Path

###### FLAGS  #########
with_26 = False
filter_glacier = True
filter_member = False
# mer de glace
#glacier_ID_filter = "G006934E45883N"
# argentiere
#glacier_ID_filter = "G006985E45951N"
# Tré-la-Tête
glacier_ID_filter = "G006784E45784N"

# Member index to be filtered
# Set to -1 to turn filtering off
filtered_member = -1

######   FILE PATHS    #######

# Folders     
workspace = str(Path(os.getcwd()).parent) + '\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 
#path_obs = 'C:\\Jordi\\PhD\\Data\\Obs\\'
path_smb = workspace + 'glacier_data\\smb\\'
path_glacier_evolution = workspace + 'glacier_data\\glacier_evolution\\'
#path_glacier_evolution = 'C:\\Jordi\\PhD\\Simulations\\AGU 2018\\'
path_smb_simulations = path_smb + 'smb_simulations\\'
path_glacier_evolution_plots = path_glacier_evolution + 'plots\\'
path_glacier_area = path_glacier_evolution + 'glacier_area\\'
path_glacier_volume = path_glacier_evolution + 'glacier_volume\\'
path_glacier_zmean = path_glacier_evolution + 'glacier_zmean\\'
path_glacier_slope20 = path_glacier_evolution + 'glacier_slope20\\'
path_glacier_melt_years = path_glacier_evolution + 'glacier_melt_years\\'
path_glacier_w_errors = path_glacier_evolution + 'glacier_w_errors\\'
path_glacier_CPDDs = path_glacier_evolution + 'glacier_CPDDs\\'
path_glacier_snowfall = path_glacier_evolution + 'glacier_snowfall\\'
path_glacier_discharge = path_glacier_evolution + 'glacier_meltwater_discharge\\'

#### We fetch all the data from the simulations
path_area_root = np.asarray(os.listdir(path_glacier_area + "projections\\"))
path_melt_years_root = np.asarray(os.listdir(path_glacier_melt_years + "projections\\"))
path_slope20_root = np.asarray(os.listdir(path_glacier_slope20 + "projections\\"))
path_volume_root = np.asarray(os.listdir(path_glacier_volume + "projections\\"))
path_errors_root = np.asarray(os.listdir(path_glacier_w_errors + "projections\\"))
path_zmean_root = np.asarray(os.listdir(path_glacier_zmean + "projections\\"))
path_CPDDs_root = np.asarray(os.listdir(path_glacier_CPDDs + "projections\\"))
path_snowfall_root = np.asarray(os.listdir(path_glacier_snowfall + "projections\\"))
path_SMB_root = np.asarray(os.listdir(path_smb_simulations + "projections\\"))

glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8')])
glacier_name_filter = glims_2003['Glacier'][glims_2003['GLIMS_ID'] == glacier_ID_filter.encode('UTF-8')]
glacier_name_filter = glacier_name_filter[0].decode('UTF-8')
#print("\nFiltered glacier name: " + str(glacier_name_filter))

proj_blob = {'year':[], 'data':[]}
annual_mean = {'SMB':copy.deepcopy(proj_blob), 'area':copy.deepcopy(proj_blob), 'volume':copy.deepcopy(proj_blob), 'zmean':copy.deepcopy(proj_blob), 'slope20':copy.deepcopy(proj_blob), 'CPDD':copy.deepcopy(proj_blob), 'snowfall':copy.deepcopy(proj_blob), 'discharge':copy.deepcopy(proj_blob)}
# Data structure composed by annual values clusters
RCP_data = {'26':copy.deepcopy(annual_mean), '45':copy.deepcopy(annual_mean), '85':copy.deepcopy(annual_mean)}
multiple_RCP_data = {'26':[], '45':[], '85':[]}
first_26, first_45, first_85 = True, True, True
# Data structure composed by member clusters
RCP_members = copy.deepcopy(multiple_RCP_data)
RCP_member_means = copy.deepcopy(multiple_RCP_data)

# Array of indexes of the data structure to iterate
data_idxs = ['SMB','area','volume','zmean','slope20','CPDD','snowfall', 'discharge']

members_with_26 = np.array(['KNMI-RACMO22E_MOHC-HadGEM2-ES', 'MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR', 'SMHI-RCA4_ICHEC-EC-EARTH'])

#######################    FUNCTIONS    ##########################################################

def save_plot_as_pdf(fig, variables, with_26):
    # Save as PDF
    if(with_26):
        fig.savefig(path_glacier_evolution_plots + 'summary\\pdf\\glacier_' + str(variables) + '_evolution_with_26.pdf')
        fig.savefig(path_glacier_evolution_plots + 'summary\\png\\glacier_' + str(variables) + '_evolution_with_26.png')
    else:
        fig.savefig(path_glacier_evolution_plots + 'summary\\pdf\\glacier_ ' + str(variables) + '_evolution.pdf')
        fig.savefig(path_glacier_evolution_plots + 'summary\\png\\glacier_ ' + str(variables) + '_evolution.png')
        
        
# Store the RCP means in CSV files
def store_RCP_mean(path_variable, variable, RCP_means):
    
    path_RCP_means = path_variable + "RCP_means\\"
    if not os.path.exists(path_RCP_means):
            os.makedirs(path_RCP_means)
    RCPs = ['26', '45', '85']
    for RCP in RCPs:
        if((with_26 and RCP == '26') or RCP != '26'):
            data = np.asarray(RCP_means[RCP][variable]['data'][:-1])
            years = np.asarray(RCP_means[RCP][variable]['year'][:-1])
            data_years = np.transpose(np.stack((data,years)))
            
            np.savetxt(path_RCP_means + "RCP" + str(RCP) + "_glacier_" + str(variable) + "_" + str(years[0])+ "_" + str(years[-1]) + '.csv', data_years, delimiter=";", fmt="%s")


##################################################################################################

# Data reading and processing
print("\♣nReading files and creating data structures...")

# Iterate different RCP-GCM-RCM combinations
member_26_idx, member_45_idx, member_85_idx = 0, 0, 0
for path_forcing_SMB, path_forcing_area, path_forcing_melt_years, path_forcing_slope20, path_forcing_volume, path_forcing_zmean, path_forcing_CPDDs, path_forcing_snowfall in zip(path_SMB_root, path_area_root, path_melt_years_root, path_slope20_root, path_volume_root, path_zmean_root, path_CPDDs_root, path_snowfall_root):
    
    current_RCP = path_forcing_area[-28:-26]
    current_member = path_forcing_area[8:-32]
    
    # Filter members depending if we want to include RCP 2.6 or not
    if((with_26 and np.any(current_member == members_with_26)) or (not with_26 and current_RCP != '26')):
        print("\nProcessing " + str(path_forcing_area))
        
        # Assign the right member idx
        if(current_RCP == '26'):
            member_idx = member_26_idx
        elif(current_RCP == '45'):
            member_idx = member_45_idx
        if(current_RCP == '85'):
            member_idx = member_85_idx
            
#        print("member_idx: " + str(member_idx))
            
        path_area_glaciers = np.asarray(os.listdir(path_glacier_area + "projections\\" + path_forcing_area))
        path_melt_years_glaciers = np.asarray(os.listdir(path_glacier_melt_years + "projections\\" + path_forcing_melt_years))
        path_slope20_glaciers = np.asarray(os.listdir(path_glacier_slope20 + "projections\\" + path_forcing_slope20))
        path_volume_glaciers = np.asarray(os.listdir(path_glacier_volume + "projections\\" + path_forcing_volume))
        path_zmean_glaciers = np.asarray(os.listdir(path_glacier_zmean + "projections\\" + path_forcing_zmean))
        path_CPDDs_glaciers = np.asarray(os.listdir(path_glacier_CPDDs + "projections\\" + path_forcing_CPDDs))
        path_snowfall_glaciers = np.asarray(os.listdir(path_glacier_snowfall + "projections\\" + path_forcing_snowfall))
        path_SMB_glaciers = np.asarray(os.listdir(path_smb_simulations + "projections\\" + path_forcing_SMB))
        
        # Initialize data structures
        # We add a new member to the RCP group
        RCP_members[current_RCP].append(copy.deepcopy(annual_mean))
        RCP_member_means[current_RCP].append(copy.deepcopy(annual_mean))
                            
        for year in range(2015, 2100):
            for data_idx in data_idxs:
                RCP_members[current_RCP][member_idx][data_idx]['data'].append([])
                RCP_members[current_RCP][member_idx][data_idx]['year'].append(year)
                RCP_member_means[current_RCP][member_idx][data_idx]['year'].append(year)
        
        bump_member = False
        # Iterate volume scaling folders
        for path_SMB_scaled, path_area_scaled, path_volume_scaled, path_zmean_scaled, path_slope20_scaled, path_CPDDs_scaled, path_snowfall_scaled in zip(path_SMB_glaciers, path_area_glaciers, path_volume_glaciers, path_zmean_glaciers, path_slope20_glaciers, path_CPDDs_glaciers, path_snowfall_glaciers):
            
            path_area_glaciers_scaled = np.asarray(os.listdir(path_glacier_area + "projections\\" + path_forcing_area + "\\" + path_area_scaled))
            path_volume_glaciers_scaled = np.asarray(os.listdir(path_glacier_volume + "projections\\" + path_forcing_volume + "\\" +path_volume_scaled))
            path_zmean_glaciers_scaled = np.asarray(os.listdir(path_glacier_zmean + "projections\\" + path_forcing_zmean + "\\" + path_zmean_scaled))
            path_slope20_glaciers_scaled = np.asarray(os.listdir(path_glacier_slope20 + "projections\\" + path_forcing_slope20 + "\\" + path_slope20_scaled))
            path_CPDDs_glaciers_scaled = np.asarray(os.listdir(path_glacier_CPDDs + "projections\\" + path_forcing_CPDDs + "\\" + path_CPDDs_scaled))
            path_snowfall_glaciers_scaled = np.asarray(os.listdir(path_glacier_snowfall + "projections\\" + path_forcing_snowfall + "\\" + path_snowfall_scaled))
            path_SMB_glaciers_scaled = np.asarray(os.listdir(path_smb_simulations + "projections\\" + path_forcing_SMB + "\\" + path_SMB_scaled))
            
            glacier_count = 0
            if(path_area_scaled == '1' and path_area_glaciers_scaled.size > 369):
                bump_member = True
                for path_SMB, path_area, path_volume, path_zmean, path_slope20, path_CPDD, path_snowfall in zip(path_SMB_glaciers_scaled, path_area_glaciers_scaled, path_volume_glaciers_scaled, path_zmean_glaciers_scaled, path_slope20_glaciers_scaled, path_CPDDs_glaciers_scaled, path_snowfall_glaciers_scaled):
                    
                    
    #                print("path_SMB[:13]: " + str(path_SMB[:13]))
    #                print("glacier_ID_filter: " + str(glacier_ID_filter))
    #                if(path_SMB[:14] == glacier_ID_filter):
                    if((filter_glacier and glacier_ID_filter == path_SMB[:14]) or not filter_glacier):
                        area_glacier = genfromtxt(path_glacier_area + "projections\\" + path_forcing_area + "\\" + path_area_scaled + "\\" + path_area, delimiter=';')
                        volume_glacier = genfromtxt(path_glacier_volume + "projections\\" + path_forcing_volume + "\\" + path_volume_scaled + "\\" + path_volume, delimiter=';')
                        zmean_glacier = genfromtxt(path_glacier_zmean + "projections\\" + path_forcing_zmean + "\\" + path_zmean_scaled + "\\" + path_zmean, delimiter=';')
                        slope20_glacier = genfromtxt(path_glacier_slope20 + "projections\\" + path_forcing_slope20 + "\\" + path_slope20_scaled + "\\" + path_slope20, delimiter=';')
                        CPDD_glacier = genfromtxt(path_glacier_CPDDs + "projections\\" + path_forcing_CPDDs + "\\" + path_CPDDs_scaled + "\\" + path_CPDD, delimiter=';')
                        snowfall_glacier = genfromtxt(path_glacier_snowfall + "projections\\" + path_forcing_snowfall + "\\" + path_snowfall_scaled + "\\" + path_snowfall, delimiter=';')
                        SMB_glacier = genfromtxt(path_smb_simulations + "projections\\" + path_forcing_SMB + "\\" + path_SMB_scaled + "\\" + path_SMB, delimiter=';')
                        
                        if(len(SMB_glacier.shape) > 1):
                            for year in range(2015, 2100):
                                for data_idx in data_idxs:
                                    if((current_RCP == '26' and first_26) or (current_RCP == '45' and first_45) or (current_RCP == '85' and first_85)):
                                        RCP_data[current_RCP][data_idx]['data'].append([])
                                        RCP_data[current_RCP][data_idx]['year'].append(year)
                            
                            # Add glacier data to blob separated by year
                            year_idx = 0
                            
#                            print("\nmember_idx: " + str(member_idx))
        
                            for SMB_y, area_y, volume_y, zmean_y, slope20_y, CPDD_y, snowfall_y in zip(SMB_glacier, area_glacier, volume_glacier, zmean_glacier, slope20_glacier, CPDD_glacier, snowfall_glacier):
                                RCP_data[current_RCP]['SMB']['data'][year_idx].append(SMB_y[1])
                                RCP_data[current_RCP]['area']['data'][year_idx].append(area_y[1])
                                RCP_data[current_RCP]['volume']['data'][year_idx].append(volume_y[1])
                                RCP_data[current_RCP]['zmean']['data'][year_idx].append(zmean_y[1])
                                RCP_data[current_RCP]['slope20']['data'][year_idx].append(slope20_y[1])
                                RCP_data[current_RCP]['CPDD']['data'][year_idx].append(CPDD_y[1])
                                RCP_data[current_RCP]['snowfall']['data'][year_idx].append(snowfall_y[1])
                                annual_discharge = -1*area_y[1]*SMB_y[1]
                                if(annual_discharge < 0):
                                    annual_discharge = 0
                                RCP_data[current_RCP]['discharge']['data'][year_idx].append(annual_discharge)
                            
                                # Add data to blob separated by RCP-GCM-RCM members
                                RCP_members[current_RCP][member_idx]['SMB']['data'][year_idx].append(SMB_y[1])
                                RCP_members[current_RCP][member_idx]['area']['data'][year_idx].append(area_y[1])
                                RCP_members[current_RCP][member_idx]['volume']['data'][year_idx].append(volume_y[1])
                                RCP_members[current_RCP][member_idx]['zmean']['data'][year_idx].append(zmean_y[1])
                                RCP_members[current_RCP][member_idx]['slope20']['data'][year_idx].append(slope20_y[1])
                                RCP_members[current_RCP][member_idx]['CPDD']['data'][year_idx].append(CPDD_y[1])
                                RCP_members[current_RCP][member_idx]['snowfall']['data'][year_idx].append(snowfall_y[1])
                                RCP_members[current_RCP][member_idx]['discharge']['data'][year_idx].append(annual_discharge)
                                
                                year_idx = year_idx+1
                            
                            if(current_RCP == '26'):
                                first_26 = False
                            elif(current_RCP == '45'):
                                first_45 = False
                            elif(current_RCP == '85'):
                                first_85 = False
                                
                            glacier_count = glacier_count+1
                            
        ### End of if RCP-GCM-RCM is part of the subgroup
        # Bump the right member idx 
        if(bump_member):
            if(current_RCP == '26'):
                member_26_idx = member_26_idx+1
            elif(current_RCP == '45'):
                member_45_idx = member_45_idx+1
            if(current_RCP == '85'):
                member_85_idx = member_85_idx+1
        
print("\nPost-processing data...")

# Compute overall average values per year
RCP_means = {'26':copy.deepcopy(annual_mean), '45':copy.deepcopy(annual_mean), '85':copy.deepcopy(annual_mean)}

year_size = copy.deepcopy(year_idx)

if(with_26):
    RCP_array = ['26', '45', '85']
else:
    RCP_array = ['45', '85']
    
for RCP in RCP_array:
    print("\nCurrent RCP: " + str(RCP))
    if(current_RCP == '26'):
        member_idx = member_26_idx
    elif(current_RCP == '45'):
        member_idx = member_45_idx
    if(current_RCP == '85'):
        member_idx = member_85_idx
    
    year_range = np.asarray(range(2015, 2100))
    for year_idx in range(0, year_range.size):
        
        # RCP_means
        RCP_means[RCP]['SMB']['data'].append(np.nanmean(RCP_data[RCP]['SMB']['data'][year_idx]))
        RCP_means[RCP]['SMB']['year'] = np.array(RCP_data[RCP]['SMB']['year'], dtype=int)
        RCP_means[RCP]['zmean']['data'].append(np.nanmean(RCP_data[RCP]['zmean']['data'][year_idx]))
        RCP_means[RCP]['zmean']['year'] = np.array(RCP_data[RCP]['zmean']['year'], dtype=int)
        RCP_means[RCP]['slope20']['data'].append(np.nanmean(RCP_data[RCP]['slope20']['data'][year_idx]))
        RCP_means[RCP]['slope20']['year'] = np.array(RCP_data[RCP]['slope20']['year'], dtype=int)
        RCP_means[RCP]['CPDD']['data'].append(np.nanmean(RCP_data[RCP]['CPDD']['data'][year_idx]))
        RCP_means[RCP]['CPDD']['year'] = np.array(RCP_data[RCP]['CPDD']['year'], dtype=int)
        RCP_means[RCP]['snowfall']['data'].append(np.nanmean(RCP_data[RCP]['snowfall']['data'][year_idx]))
        RCP_means[RCP]['snowfall']['year'] = np.array(RCP_data[RCP]['snowfall']['year'], dtype=int)
        annual_discharge = -1*np.nansum(np.asarray(RCP_data[RCP]['SMB']['data'][year_idx])*np.asarray(RCP_data[RCP]['area']['data'][year_idx]))/member_idx
        annual_discharge = np.where(annual_discharge < 0, 0, annual_discharge)
        RCP_means[RCP]['discharge']['data'].append(annual_discharge)
        RCP_means[RCP]['discharge']['year'] = np.array(RCP_data[RCP]['snowfall']['year'], dtype=int)
#        
        # RCP_member_means
        if(filter_glacier):
            member_idx = len(RCP_members[RCP])
#        print("len(RCP_members[RCP]): " + str(len(RCP_members[RCP])))
#        print("member_idx: " + str(member_idx))
        for member in range(0, member_idx-1):
#            if(year_idx == 83):
#                print("member: " + str(member))
#            print("year_idx: " + str(year_idx))
#            if(member == 12):
#                import pdb; pdb.set_trace()
            
            RCP_member_means[RCP][member]['SMB']['year'] = np.array(RCP_members[RCP][member]['SMB']['year'], dtype=int)
            RCP_member_means[RCP][member]['area']['year'] = np.array(RCP_members[RCP][member]['area']['year'], dtype=int)
            RCP_member_means[RCP][member]['volume']['year'] = np.array(RCP_members[RCP][member]['volume']['year'], dtype=int)
            RCP_member_means[RCP][member]['zmean']['year'] = np.array(RCP_members[RCP][member]['zmean']['year'], dtype=int)
            RCP_member_means[RCP][member]['slope20']['year'] = np.array(RCP_members[RCP][member]['slope20']['year'], dtype=int)
            RCP_member_means[RCP][member]['CPDD']['year'] = np.array(RCP_members[RCP][member]['CPDD']['year'], dtype=int)
            RCP_member_means[RCP][member]['snowfall']['year'] = np.array(RCP_members[RCP][member]['snowfall']['year'], dtype=int)
            RCP_member_means[RCP][member]['discharge']['year'] = np.array(RCP_members[RCP][member]['snowfall']['year'], dtype=int)
            
            if(len(RCP_members[RCP][member]['SMB']['data'][year_idx]) > 0):
                RCP_member_means[RCP][member]['SMB']['data'].append(np.nanmean(RCP_members[RCP][member]['SMB']['data'][year_idx]))
                RCP_member_means[RCP][member]['area']['data'].append(np.nansum(RCP_members[RCP][member]['area']['data'][year_idx]))
                RCP_member_means[RCP][member]['volume']['data'].append(np.nansum(RCP_members[RCP][member]['volume']['data'][year_idx]))
                RCP_member_means[RCP][member]['zmean']['data'].append(np.nanmean(RCP_members[RCP][member]['zmean']['data'][year_idx]))
                RCP_member_means[RCP][member]['slope20']['data'].append(np.nanmean(RCP_members[RCP][member]['slope20']['data'][year_idx]))
                RCP_member_means[RCP][member]['CPDD']['data'].append(np.nanmean(RCP_members[RCP][member]['CPDD']['data'][year_idx]))
                RCP_member_means[RCP][member]['snowfall']['data'].append(np.nanmean(RCP_members[RCP][member]['snowfall']['data'][year_idx]))
                member_annual_discharge = -1*np.nansum(np.asarray(RCP_members[RCP][member]['SMB']['data'][year_idx])*np.asarray(RCP_members[RCP][member]['area']['data'][year_idx]))
                member_annual_discharge = np.where(member_annual_discharge < 0, 0, member_annual_discharge)
                RCP_member_means[RCP][member]['discharge']['data'].append(member_annual_discharge)
                
            else:
                RCP_member_means[RCP][member]['SMB']['data'].append(np.nan)
                RCP_member_means[RCP][member]['area']['data'].append(np.nan)
                RCP_member_means[RCP][member]['volume']['data'].append(np.nan)
                RCP_member_means[RCP][member]['zmean']['data'].append(np.nan)
                RCP_member_means[RCP][member]['slope20']['data'].append(np.nan)
                RCP_member_means[RCP][member]['CPDD']['data'].append(np.nan)
                RCP_member_means[RCP][member]['snowfall']['data'].append(np.nan)
                RCP_member_means[RCP][member]['discharge']['data'].append(np.nan)
                
#            if(year_idx == 83):
#                print("\nFinal volume: " + str(np.nansum(RCP_members[RCP][member]['volume']['data'][year_idx])))

    for year_idx in range(0, year_range.size):
        area_year, volume_year = [],[]
        if(filter_glacier):
            member_idx = len(RCP_members[RCP])
        #        print("len(RCP_members[RCP]): " + str(len(RCP_members[RCP])))
        #        print("member_idx: " + str(member_idx))
        for member in range(0, member_idx-1):
            # RCP_means
            area_year.append(np.nansum(RCP_members[RCP][member]['area']['data'][year_idx]))
            volume_year.append(np.nansum(RCP_members[RCP][member]['volume']['data'][year_idx]))
        
        RCP_means[RCP]['area']['data'].append(np.nanmean(area_year))
        RCP_means[RCP]['area']['year'] = np.array(RCP_data[RCP]['area']['year'], dtype=int)
        RCP_means[RCP]['volume']['data'].append(np.nanmean(volume_year))
        RCP_means[RCP]['volume']['year'] = np.array(RCP_data[RCP]['volume']['year'], dtype=int)
 
#print(overall_annual_mean)

##########    PLOTS    #######################
if(filter_glacier):
    header = glacier_ID_filter + "_"
else:
    header = "french_alps_avg_"

#############       Plot each one of the RCP-GCM-RCM combinations       #############################################
    
fig1, (ax11, ax12) = plt.subplots(1,2, figsize=(10, 6))
if(filter_glacier):
    fig1.suptitle("Glacier " + glacier_name_filter + " glacier projections under climate change")
else:
    fig1.suptitle("Regional average French alpine glacier projections under climate change")
ax11.set_ylabel('Volume (m$^3$ 10$^6$)')
ax11.set_xlabel('Year')

member_idx = 0
for member_26 in RCP_member_means['26']:
    data_26 = member_26['volume']['data']
    if(len(data_26) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['26']['volume']['year']) > len(data_26)):
            ax11.plot(RCP_means['26']['volume']['year'][:-1], data_26, linewidth=0.1, alpha=0.5, c='blue')
        else:
            ax11.plot(RCP_means['26']['volume']['year'], data_26, linewidth=0.1, alpha=0.5, c='blue')
    member_idx=member_idx+1
member_idx = 0
for member_45 in RCP_member_means['45']:
    data_45 = member_45['volume']['data']
    if(len(data_45) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['45']['volume']['year']) > len(data_45)):
            ax11.plot(RCP_means['45']['volume']['year'][:-1], data_45, linewidth=0.1, alpha=0.5, c='green')
        else:
            ax11.plot(RCP_means['45']['volume']['year'], data_45, linewidth=0.1, alpha=0.5, c='green')
    member_idx=member_idx+1
member_idx = 0
for member_85 in RCP_member_means['85']:
    data_85 = member_85['volume']['data']
    if(len(data_85) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['85']['volume']['year']) > len(data_85)):
            ax11.plot(RCP_means['85']['volume']['year'][:-1], data_85, linewidth=0.1, alpha=0.5, c='red')
        else:
            ax11.plot(RCP_means['85']['volume']['year'], data_85, linewidth=0.1, alpha=0.5, c='red')
    member_idx=member_idx+1
    
# Plot the average of each RCP
if(with_26):
    line111, = ax11.plot(RCP_means['26']['volume']['year'][:-1], np.asarray(RCP_means['26']['volume']['data'][:-1]), linewidth=3, label='RCP 2.6', c='blue')
line112, = ax11.plot(RCP_means['45']['volume']['year'][:-1], np.asarray(RCP_means['45']['volume']['data'][:-1]), linewidth=3, label='RCP 4.5', c='green')
line113, = ax11.plot(RCP_means['85']['volume']['year'][:-1], np.asarray(RCP_means['85']['volume']['data'][:-1]), linewidth=3, label='RCP 8.5', c='red')
ax11.legend()


ax12.set_ylabel('Area (km$^2$)')
ax12.set_xlabel('Year')

member_idx = 0
for member_26 in RCP_member_means['26']:
    data_26 = member_26['area']['data']
    if(len(data_26) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['26']['area']['year']) > len(data_26)):
            ax12.plot(RCP_means['26']['area']['year'][:-1], data_26, linewidth=0.1, alpha=0.5, c='blue')
        else:
            ax12.plot(RCP_means['26']['area']['year'], data_26, linewidth=0.1, alpha=0.5, c='blue')
    member_idx=member_idx+1
member_idx = 0
for member_45 in RCP_member_means['45']:
    data_45 = member_45['area']['data']
    if(len(data_45) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['45']['area']['year']) > len(data_45)):
            ax12.plot(RCP_means['45']['area']['year'][:-1], data_45, linewidth=0.1, alpha=0.5, c='green')
        else:
            ax12.plot(RCP_means['45']['area']['year'], data_45, linewidth=0.1, alpha=0.5, c='green')
    member_idx=member_idx+1
member_idx = 0
for member_85 in RCP_member_means['85']:
    data_85 = member_85['area']['data']
    if(len(data_85) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['85']['area']['year']) > len(data_85)):
            ax12.plot(RCP_means['85']['area']['year'][:-1], data_85, linewidth=0.1, alpha=0.5, c='red')
        else:
            ax12.plot(RCP_means['85']['area']['year'], data_85, linewidth=0.1, alpha=0.5, c='red')
    member_idx=member_idx+1

if(with_26):
    line121, = ax12.plot(RCP_means['26']['area']['year'][:-1], np.asarray(RCP_means['26']['area']['data'][:-1]), linewidth=3, label='RCP 2.6', c='blue')
line122, = ax12.plot(RCP_means['45']['area']['year'][:-1], np.asarray(RCP_means['45']['area']['data'][:-1]), linewidth=3, label='RCP 4.5', c='green')
line123, = ax12.plot(RCP_means['85']['area']['year'][:-1], np.asarray(RCP_means['85']['area']['data'][:-1]), linewidth=3, label='RCP 8.5', c='red')
ax12.legend()

# Save as PDF
save_plot_as_pdf(fig1, header + 'volume_area', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_glacier_area, 'area', RCP_means)
store_RCP_mean(path_glacier_volume, 'volume', RCP_means)

###############     Plot the evolution of topographical parameters    ####################################
fig2, (ax21, ax22) = plt.subplots(1,2, figsize=(10, 6))
if(filter_glacier):
    fig2.suptitle("Glacier " + glacier_name_filter + " glacier projections under climate change")
else:
    fig2.suptitle("Regional average French alpine glacier projections under climate change")
ax21.set_ylabel('Mean glacier altitude (m)')
ax21.set_xlabel('Year')

# Mean altitude
if(with_26):
    line211, = ax21.plot(RCP_means['26']['zmean']['year'][1:-1], RCP_means['26']['zmean']['data'][1:-1], linewidth=3, label='RCP 2.6', c='blue')
line212, = ax21.plot(RCP_means['45']['zmean']['year'][1:-1], RCP_means['45']['zmean']['data'][1:-1], linewidth=3, label='RCP 4.5', c='green')
line213, = ax21.plot(RCP_means['85']['zmean']['year'][1:-1], RCP_means['85']['zmean']['data'][1:-1], linewidth=3, label='RCP 8.5', c='red')
ax21.legend()

# Slope 20% altitudinal range
ax22.set_ylabel('Slope of 20% altitudinal range (°)')
ax22.set_xlabel('Year')
if(with_26):
    line221, = ax22.plot(RCP_means['26']['slope20']['year'][1:-1], RCP_means['26']['slope20']['data'][1:-1], linewidth=3, label='RCP 2.6', c='blue')
line222, = ax22.plot(RCP_means['45']['slope20']['year'][1:-1], RCP_means['45']['slope20']['data'][1:-1], linewidth=3, label='RCP 4.5', c='green')
line223, = ax22.plot(RCP_means['85']['slope20']['year'][1:-1], RCP_means['85']['slope20']['data'][1:-1], linewidth=3, label='RCP 8.5', c='red')
ax22.legend()

# Save as PDF
save_plot_as_pdf(fig2, header + 'zmean_slope', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_glacier_zmean, 'zmean', RCP_means)
store_RCP_mean(path_glacier_slope20, 'slope20', RCP_means)

###############     Plot the evolution of temperature and snowfall    ####################################
fig3, (ax31, ax32) = plt.subplots(1,2, figsize=(14, 6))
if(filter_glacier):
    fig3.suptitle("Glacier " + glacier_name_filter + " climate projections")
else:
    fig3.suptitle("Regional average French alpine glacier projections under climate change")
ax31.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
ax31.set_ylabel('Cumulative positive degree days anomaly (1984-2015)')
ax31.set_xlabel('Year')

# CPDD
member_idx = 0
for member_26 in RCP_member_means['26']:
    data_26 = member_26['CPDD']['data'][1:]
    if(len(data_26) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['26']['CPDD']['year']) > len(data_26)):
            ax31.plot(RCP_means['26']['CPDD']['year'][:-1], data_26, linewidth=0.1, alpha=0.5, c='blue')
        else:
            ax31.plot(RCP_means['26']['CPDD']['year'][1:], data_26, linewidth=0.1, alpha=0.5, c='blue')
    member_idx=member_idx+1
member_idx = 0
for member_45 in RCP_member_means['45']:
    data_45 = member_45['CPDD']['data'][1:]
    if(len(data_45) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['45']['CPDD']['year']) > len(data_45)):
            ax31.plot(RCP_means['45']['CPDD']['year'][:-1], data_45, linewidth=0.1, alpha=0.5, c='green')
        else:
            ax31.plot(RCP_means['45']['CPDD']['year'][1:], data_45, linewidth=0.1, alpha=0.5, c='green')
    member_idx=member_idx+1
member_idx = 0
for member_85 in RCP_member_means['85']:
    data_85 = member_85['CPDD']['data'][1:]
    if(len(data_85) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['85']['CPDD']['year']) > len(data_85)):
            ax31.plot(RCP_means['85']['CPDD']['year'][:-1], data_85, linewidth=0.1, alpha=0.5, c='red')
        else:
            ax31.plot(RCP_means['85']['CPDD']['year'][1:], data_85, linewidth=0.1, alpha=0.5, c='red')
    member_idx=member_idx+1

if(with_26):
    line311, = ax31.plot(RCP_means['26']['CPDD']['year'][1:-1], RCP_means['26']['CPDD']['data'][1:-1], linewidth=3, label='RCP 2.6', c='blue')
line312, = ax31.plot(RCP_means['45']['CPDD']['year'][1:-1], RCP_means['45']['CPDD']['data'][1:-1], linewidth=3, label='RCP 4.5', c='green')
line313, = ax31.plot(RCP_means['85']['CPDD']['year'][1:-1], RCP_means['85']['CPDD']['data'][1:-1], linewidth=3, label='RCP 8.5', c='red')
ax31.legend()

# Snowfall
member_idx = 0
for member_26 in RCP_member_means['26']:
    data_26 = member_26['snowfall']['data'][1:]
    if(len(data_26) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['26']['snowfall']['year']) > len(data_26)):
            ax32.plot(RCP_means['26']['snowfall']['year'][:-1], data_26, linewidth=0.1, alpha=0.5, c='blue')
        else:
            ax32.plot(RCP_means['26']['snowfall']['year'][1:], data_26, linewidth=0.1, alpha=0.5, c='blue')
    member_idx=member_idx+1
member_idx = 0
for member_45 in RCP_member_means['45']:
    data_45 = member_45['snowfall']['data'][1:]
    if(len(data_45) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['45']['snowfall']['year']) > len(data_45)):
            ax32.plot(RCP_means['45']['snowfall']['year'][:-1], data_45, linewidth=0.1, alpha=0.5, c='green')
        else:
            ax32.plot(RCP_means['45']['snowfall']['year'][1:], data_45, linewidth=0.1, alpha=0.5, c='green')
    member_idx=member_idx+1
member_idx = 0
for member_85 in RCP_member_means['85']:
    data_85 = member_85['snowfall']['data'][1:]
    if(len(data_85) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['85']['snowfall']['year']) > len(data_85)):
            ax32.plot(RCP_means['85']['snowfall']['year'][:-1], data_85, linewidth=0.1, alpha=0.5, c='red')
        else:
            ax32.plot(RCP_means['85']['snowfall']['year'][1:], data_85, linewidth=0.1, alpha=0.5, c='red')
    member_idx=member_idx+1

ax32.set_ylabel('Annual cumulative snowfall anomaly (1984-2015)')
ax32.set_xlabel('Year')
ax32.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
if(with_26):
    line321, = ax32.plot(RCP_means['26']['snowfall']['year'][1:-1], RCP_means['26']['snowfall']['data'][1:-1], linewidth=3, label='RCP 2.6', c='blue')
line322, = ax32.plot(RCP_means['45']['snowfall']['year'][1:-1], RCP_means['45']['snowfall']['data'][1:-1], linewidth=3, label='RCP 4.5', c='green')
line323, = ax32.plot(RCP_means['85']['snowfall']['year'][1:-1], RCP_means['85']['snowfall']['data'][1:-1], linewidth=3, label='RCP 8.5', c='red')
ax32.legend()

# Save as PDF
save_plot_as_pdf(fig3, header + 'CPDD_snowfall', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_glacier_CPDDs, 'CPDD', RCP_means)
store_RCP_mean(path_glacier_snowfall, 'snowfall', RCP_means)

###############     Plot the glacier-wide SMB   ####################################
fig4, (ax41) = plt.subplots(1,1, figsize=(10, 6))
ax41.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
if(filter_glacier):
    fig4.suptitle("Glacier " + glacier_name_filter + " glacier-wide SMB evolution under climate change")
else:
    fig4.suptitle("Average glacier-wide SMB projections of French alpine glaciers under climate change")
ax41.set_ylabel('Glacier-wide SMB (m.w.e. a$^-1$)')
ax41.set_xlabel('Year')

member_idx = 0
for member_26 in RCP_member_means['26']:
    data_26 = member_26['SMB']['data'][1:]
    if(len(data_26) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['26']['SMB']['year']) > len(data_26)):
            ax41.plot(RCP_means['26']['SMB']['year'][:-1], data_26, linewidth=0.1, alpha=0.5, c='blue')
        else:
            ax41.plot(RCP_means['26']['SMB']['year'][1:], data_26, linewidth=0.1, alpha=0.5, c='blue')
    member_idx=member_idx+1
member_idx = 0
for member_45 in RCP_member_means['45']:
    data_45 = member_45['SMB']['data'][1:]
    if(len(data_45) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['45']['SMB']['year']) > len(data_45)):
            ax41.plot(RCP_means['45']['SMB']['year'][:-1], data_45, linewidth=0.1, alpha=0.5, c='green')
        else:
            ax41.plot(RCP_means['45']['SMB']['year'][1:], data_45, linewidth=0.1, alpha=0.5, c='green')
    member_idx=member_idx+1
member_idx = 0
for member_85 in RCP_member_means['85']:
    data_85 = member_85['SMB']['data'][1:]
    if(len(data_85) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['85']['SMB']['year']) > len(data_85)):
            ax41.plot(RCP_means['85']['SMB']['year'][:-1], data_85, linewidth=0.1, alpha=0.5, c='red')
        else:
            ax41.plot(RCP_means['85']['SMB']['year'][1:], data_85, linewidth=0.1, alpha=0.5, c='red')
    member_idx=member_idx+1

if(with_26):
    line41, = ax41.plot(RCP_means['26']['SMB']['year'][1:-1], RCP_means['26']['SMB']['data'][1:-1], linewidth=3, label='RCP 2.6', c='blue')
line42, = ax41.plot(RCP_means['45']['SMB']['year'][1:-1], RCP_means['45']['SMB']['data'][1:-1], linewidth=3, label='RCP 4.5', c='green')
line43, = ax41.plot(RCP_means['85']['SMB']['year'][1:-1], RCP_means['85']['SMB']['data'][1:-1], linewidth=3, label='RCP 8.5', c='red')
ax41.legend()

# Save as PDF
save_plot_as_pdf(fig4, header + 'SMB', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_smb_simulations, 'SMB', RCP_means)

###############     Plot the glacier meltwater discharge   ####################################
fig5, (ax51) = plt.subplots(1,1, figsize=(10, 6))
ax51.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
if(filter_glacier):
    fig5.suptitle("Glacier " + glacier_name_filter + " meltwater discharge evolution under climate change")
else:
    fig5.suptitle("Average meltwater discharge projections of French alpine glacier under climate change")
ax51.set_ylabel('Meltwater discharge (m$^3$ 10$^6$)')
ax51.set_xlabel('Year')

member_idx = 0
for member_26 in RCP_member_means['26']:
    data_26 = member_26['discharge']['data'][1:]
    if(len(data_26) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['26']['discharge']['year']) > len(data_26)):
            ax51.plot(RCP_means['26']['discharge']['year'][:-1], data_26, linewidth=0.1, alpha=0.5, c='blue')
        else:
            ax51.plot(RCP_means['26']['discharge']['year'][1:], data_26, linewidth=0.1, alpha=0.5, c='blue')
    member_idx=member_idx+1
member_idx = 0
for member_45 in RCP_member_means['45']:
    data_45 = member_45['discharge']['data'][1:]
    if(len(data_45) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['45']['discharge']['year']) > len(data_45)):
            ax51.plot(RCP_means['45']['discharge']['year'][:-1], data_45, linewidth=0.1, alpha=0.5, c='green')
        else:
            ax51.plot(RCP_means['45']['discharge']['year'][1:], data_45, linewidth=0.1, alpha=0.5, c='green')
    member_idx=member_idx+1
member_idx = 0
for member_85 in RCP_member_means['85']:
    data_85 = member_85['discharge']['data'][1:]
    if(len(data_85) > 0 and (member_idx == filtered_member or filtered_member == -1)):
        if(len(RCP_means['85']['discharge']['year']) > len(data_85)):
            ax51.plot(RCP_means['85']['discharge']['year'][:-1], data_85, linewidth=0.1, alpha=0.5, c='red')
        else:
            ax51.plot(RCP_means['85']['discharge']['year'][1:], data_85, linewidth=0.1, alpha=0.5, c='red')
    member_idx=member_idx+1

if(with_26):
    line41, = ax51.plot(RCP_means['26']['discharge']['year'][1:-1], RCP_means['26']['discharge']['data'][1:-1], linewidth=3, label='RCP 2.6', c='blue')
line42, = ax51.plot(RCP_means['45']['discharge']['year'][1:-1], RCP_means['45']['discharge']['data'][1:-1], linewidth=3, label='RCP 4.5', c='green')
line43, = ax51.plot(RCP_means['85']['discharge']['year'][1:-1], RCP_means['85']['discharge']['data'][1:-1], linewidth=3, label='RCP 8.5', c='red')
ax51.legend()

# Save as PDF
save_plot_as_pdf(fig5, header + 'meltwater_discharge', with_26)

# Store RCP means in CSV file
store_RCP_mean(path_glacier_discharge, 'discharge', RCP_means)

plt.show()