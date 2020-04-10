# -*- coding: utf-8 -*-

"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

GLACIER ICE VOLUME AND SURFACE AREA PROJECTION AND EVOLUTION


"""

## Dependencies: 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import xarray as xr
from numpy import genfromtxt
from numba import jit
import unicodedata
import subprocess
import os
import shutil
import sys
import time
from osgeo import gdal, ogr, osr
import copy
from difflib import SequenceMatcher
from netCDF4 import Dataset
import settings
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, normalize

from keras import backend as K
from keras.models import load_model

# To ignore errors when raster has only one pixel and normalization is divided by 0
np.seterr(divide='ignore', invalid='ignore') 

#import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#matplotlib.use("GTKCairo", warn=False)

# Turn interactive plotting off
plt.ioff()

os.environ['SHAPE_ENCODING'] = "latin-1"



######   FILE PATHS    #######
    
# Folders     

workspace = str(Path(os.getcwd()).parent) + '\\'
#workspace = 'C:\\Jordi\\PhD\\Python\\'
path_smb = workspace + 'glacier_data\\smb\\'
path_glacier_evolution = workspace + 'glacier_data\\glacier_evolution\\'
# Delta h parameterization functions
path_delta_h_param = workspace + "glacier_data\\delta_h_param\\"
# Shapefiles
path_glacier_shapefiles = workspace + 'glacier_data\\glacier_shapefiles\\'
path_glacier_2003_shapefiles = workspace + 'glacier_data\\glacier_shapefiles\\2003\\'
path_glacier_2015_shapefiles = workspace + 'glacier_data\\glacier_shapefiles\\2015\\'
path_glacier_flowlines_shapefile = path_glacier_2003_shapefiles + 'GLIMS_flowlines_2003' + '.shp'
# Rasters
path_glacier_ID_rasters = workspace + 'glacier_data\\glacier_rasters\\glacier_thickness\\thickness_tif\\'
path_glacier_DEM_rasters = workspace + 'glacier_data\\glacier_rasters\\glacier_thickness\\dem_tif\\'
path_glacier_evolution_DEM_rasters = path_glacier_DEM_rasters + 'glacier_evolution\\' 
path_glacier_evolution_ID_rasters = path_glacier_ID_rasters + 'glacier_evolution\\'
# Glacier evolution files
path_glacier_evolution_plots = path_glacier_evolution + 'plots\\'
path_glacier_area = path_glacier_evolution + 'glacier_area\\'
path_glacier_volume = path_glacier_evolution + 'glacier_volume\\'
path_glacier_zmean = path_glacier_evolution + 'glacier_zmean\\'
path_glacier_slope20 = path_glacier_evolution + 'glacier_slope20\\'
path_glacier_melt_years = path_glacier_evolution + 'glacier_melt_years\\'
path_glacier_w_errors = path_glacier_evolution + 'glacier_w_errors\\'
path_glacier_CPDDs = path_glacier_evolution + 'glacier_CPDDs\\'
path_glacier_snowfall = path_glacier_evolution + 'glacier_snowfall\\'
# GLIMS data
path_glims = workspace + 'glacier_data\\GLIMS\\' 
# SAFRAN climate forcings
path_safran_forcings = 'C:\\Jordi\\PhD\\Data\\SAFRAN-Nivo-2017\\'
global path_smb_function_safran 
path_smb_function_safran = path_smb + 'smb_function\\SAFRAN\\'
global path_smb_function_adamont
path_smb_function_adamont = path_smb + 'smb_function\\ADAMONT\\'
# Path to be updated with ADAMONT forcings local path
path_adamont_forcings = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\treated\\'
#path_adamont_forcings = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\\FORCING_ADAMONT_IGE_BERGER\\projections\\'
# SMB simulation files
path_smb_simulations = path_smb + 'smb_simulations\\'
path_smb_function = path_smb + 'smb_function\\'



################     TOOLS    ###########################

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def shorten_name(glacierName):
    if(len(glacierName) > 30):
        ch_idx = glacierName.find('-')
        glacierName = glacierName[:ch_idx-1]
    return glacierName

def empty_folder(path):
    if(os.path.exists(path)):
        shutil.rmtree(path)
        
def automatic_file_name_save(file_name_h, file_name_t, data, f_format):
    file_name = file_name_h + file_name_t
    appendix = 2
    stored = False
    while not stored:
        if not os.path.exists(file_name):
            try:
                if(f_format == 'csv'):
                    np.savetxt(file_name, data, delimiter=";", fmt="%.7f")
                else:
                    with open(file_name, 'wb') as file_f:
                        np.save(file_f, data)
                stored = True
            except IOError:
                print("File currently opened. Please close it to proceed.")
                os.system('pause')
                # We try again
                try:
                    print("\nRetrying storing " + str(file_name))
                    if(f_format == 'csv'):
                        np.savetxt(file_name, data, delimiter=";", fmt="%.7f")
                    else:
                        with open(file_name, 'wb') as file_f:
                            np.save(file_f, data)
                    stored = True
                except IOError:
                    print("File still not available. Aborting simulations.")
        else:
            file_name = file_name_h + str(appendix) + "_" + file_name_t
            appendix = appendix+1

def store_file(data, path, midfolder, file_description, glimsID, year_start, year):
    year_range = np.asarray(range(year_start, year))
    data =  np.asarray(data).reshape(-1,1)
    data_w_years = np.column_stack((year_range, data))
    path_midfolder = path + midfolder
    if not os.path.exists(path_midfolder):
        os.makedirs(path_midfolder)
        
#    file_name = path + midfolder + glimsID + "_" + str(file_description) + '.csv'
    file_name_h = path + midfolder + str(glimsID) + "_"
    file_name_t = str(file_description) + '.csv'
    # We save the file with an unexisting name
    automatic_file_name_save(file_name_h, file_name_t, data_w_years, 'csv')
    
@jit
def similar(a, b):
    ratios = []
    for glacier_name in a:
        ratios.append(SequenceMatcher(None, glacier_name, b).ratio())
    ratios = np.asarray(ratios)
    return ratios.max(), ratios.argmax()
 
# Remove all accents from string
def strip_accents(unicode_or_str):
    
    if isinstance(unicode_or_str, str):
        text = unicode_or_str
    else:
        text = unicode_or_str.decode('utf-8')
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return text

# Clips a raster with the shape of a polygon   
def clipRaster_with_polygon(output_cropped_raster, input_raster, shapefile_mask):
    if os.path.exists(output_cropped_raster):
        os.remove(output_cropped_raster)
        
    try:
        subprocess.check_output("gdalwarp --config GDALWARP_IGNORE_BAD_CUTLINE YES -q -cutline \"" + shapefile_mask 
                                           + "\" -of GTiff \"" 
                                           + input_raster + "\" \"" + output_cropped_raster +"\"",
                                     stderr=subprocess.PIPE,
                                     shell=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

def normalize_dem(dem):
    dem_n = dem.copy()
    dem_n = (float(dem.max()) - dem)/(float(dem.max()) - float(dem.min()))
    return dem_n

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    diff = np.amin(np.abs(array-value))
    return idx, diff

def find_nearest_altitude(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

@jit
def create_input_array(season_anomalies_y, monthly_anomalies_y, mean_alt_y, max_alt, slope20, area_y, lat, lon, aspect):
    cpdd_y = season_anomalies_y['CPDD']
    w_snow_y = season_anomalies_y['winter_snow']
    s_snow_y = season_anomalies_y['summer_snow']
    mon_temp_anomaly_y = monthly_anomalies_y['temps']
    mon_snow_anomaly_y = monthly_anomalies_y['snow']
    
    # Lasso input features
    input_variables_array = np.array([cpdd_y, w_snow_y, s_snow_y, mean_alt_y, max_alt, slope20, area_y, lon, lat, np.cos(aspect), mean_alt_y*cpdd_y, slope20*cpdd_y, max_alt*cpdd_y, area_y*cpdd_y, lat*cpdd_y, lon*cpdd_y, aspect*cpdd_y, mean_alt_y*w_snow_y, slope20*w_snow_y, max_alt*w_snow_y, area_y*w_snow_y, lat*w_snow_y, lon*w_snow_y, aspect*w_snow_y, mean_alt_y*s_snow_y, slope20*s_snow_y, max_alt*s_snow_y, area_y*s_snow_y, lat*s_snow_y, lon*s_snow_y, aspect*s_snow_y])
#    input_variables_array = np.append(input_variables_array, mon_temp_anomaly_y)
#    input_variables_array = np.append(input_variables_array, mon_snow_anomaly_y)
    
    # ANN input features
#    input_features_nn_array = np.array([cpdd_y, w_snow_y, s_snow_y, mean_alt_y, max_alt, slope20, area_y, lon, lat, aspect])
    input_features_nn_array = np.array([cpdd_y, w_snow_y, s_snow_y, mean_alt_y, max_alt, slope20, area_y, lon, lat, np.cos(aspect)])
    input_features_nn_array = np.append(input_features_nn_array, mon_temp_anomaly_y)
    input_features_nn_array = np.append(input_features_nn_array, mon_snow_anomaly_y)
    
    return np.asarray(input_variables_array), np.asarray(input_features_nn_array)

# Preloads in memory all the ANN SMB ensemble models to speed up the simulations
def preload_ensemble_SMB_models():
    # CV ensemble
    path_CV_ensemble = settings.path_cv_ann
    path_CV_ensemble_members = np.asarray(os.listdir(path_CV_ensemble))
    
    print("\nTaking CV ensemble models from: " + str(path_CV_ensemble))
    
    # Full model ensemble
    path_ensemble = settings.path_ensemble_ann
    path_ensemble_members = np.asarray(os.listdir(path_ensemble))
    
    print("\nTaking full ensemble models from: " + str(path_ensemble))
    
    CV_ensemble_members = np.ndarray(path_CV_ensemble_members.shape, dtype=np.object)
    ensemble_members = np.ndarray(path_ensemble_members.shape, dtype=np.object)
    
    member_idx = 0
    print("\nPreloading CV ensemble SMB models...")
    for path_CV_member in path_CV_ensemble_members:
        # We retrieve the ensemble member ANN model
        ann_CV_member_model = load_model(path_CV_ensemble + path_CV_member, custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error}, compile=False)
#        CV_ensemble_members.append(ann_CV_member_model)
        CV_ensemble_members[member_idx] = ann_CV_member_model
        print("|", end="", flush=True)
        member_idx = member_idx+1
    
    member_idx = 0
    print("\n\nPreloading ensemble full SMB models...")
    for path_member in path_ensemble_members:
        # We retrieve the ensemble member ANN model
        ann_member_model = load_model(path_ensemble + path_member + '\\ann_glacier_model.h5', custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error}, compile=False)
#        ensemble_members.append(ann_member_model)
        ensemble_members[member_idx] = ann_member_model
        print("|", end="", flush=True)
        member_idx = member_idx+1
        
    CV_ensemble_members = np.asarray(CV_ensemble_members)
    ensemble_members = np.asarray(ensemble_members)
    print("\n")
    
    # We pack all the ensemble models
    ensemble_member_models = {'CV': CV_ensemble_members, 'full':ensemble_members}
    
    return ensemble_member_models

#def compute_generalization_error_weights(CV_RMSE, SMB_members, SMB_mean):
#    # We compute the generalization errors to weight the average
#    ge_weights, ambiguities = [],[]
#    # Pre-compute the ambiguities
#    for SMB_member in SMB_members:
#        ambiguities.append((SMB_member - SMB_mean)**2)
#    ambiguities = np.asarray(ambiguities)
#    
#    for RMSE_i, ambiguity_i in zip(CV_RMSE, ambiguities):
#        # Generalization error = RMSE_ensemble  - ensemble_ambiguity
#        # Both values are normalized in order to make them comparable
#        # RMSE is inversed for weighting
#        inv_RMSE_i_norm = (1/RMSE_i - np.min(1/CV_RMSE))/np.ptp(1/CV_RMSE)
#        ambiguity_i_norm = (ambiguity_i - np.min(ambiguities))/np.ptp(ambiguities)
#        # The generalization error is inversed in order to be used as weights
#        ge_weights.append(inv_RMSE_i_norm + ambiguity_i_norm)
#    
#    ge_weights = np.asarray(ge_weights)
#    
#    return ge_weights

# Makes an ANN glacier-wide SMB simulation using an ensemble approach
# Evolution flag = True for glacier_evolution.py format and False for smb_validation.py format
def make_ensemble_simulation(ensemble_SMB_models, x_ann, batch_size, glimsID, glims_rabatel, evolution):
    SMB_ensemble = []
    training_slopes = glims_rabatel['slope20']
    if(len(x_ann.shape) == 2):
        ref_slope = np.median(x_ann[:,5])
    else:
        ref_slope = np.median(x_ann[5])
    first = True
    CV_ensemble = False
    member_idx = 0
    
    # We compute the quadratic slope difference 
    slope_dist = (training_slopes - ref_slope)**2
    
    # Depending if glacier is present in training dataset we use the CV or full model
    if(np.any(glims_rabatel['GLIMS_ID'] == glimsID.encode('utf-8'))):
        SMB_ensemble_members = ensemble_SMB_models['full']
        print("\nFull ensemble models")
    else:
        SMB_ensemble_members = ensemble_SMB_models['CV']
        CV_ensemble = True
        print("\nCross-validation ensemble models")
    
    # We iterate the previously loaded ensemble models
    for ensemble_model in SMB_ensemble_members:
        # We retrieve the ensemble member ANN model
        # Make single member prediction
        if(evolution):
            x_ann = x_ann.reshape(1,-1)
            SMB_member = ensemble_model.predict(x_ann, batch_size=batch_size)[0][0]
        else:
            SMB_member = ensemble_model.predict(x_ann, batch_size=batch_size).flatten()
        if(first):
            print("Running ensemble SMB simulation", end="", flush=True)
        print(".", end="", flush=True)
        
        # Add member simulation to ensemble
        SMB_ensemble.append(SMB_member)
        first = False
        member_idx = member_idx+1
    
    # We compute the ensemble average value
    SMB_ensemble = np.asarray(SMB_ensemble)
    
    if(evolution):
    # Glacier evolution modelling
        if(CV_ensemble):
            # We generate the weights for the ensemble averaging
#            ge_weights = compute_generalization_error_weights(CV_RMSE, SMB_ensemble, SMB_ensemble.mean())
            # Generalization error weighted ensemble
#            ensemble_simulation = np.average(SMB_ensemble, weights=ge_weights)
            # Inverse slope difference weighted ensemble
#            ensemble_simulation = np.average(SMB_ensemble, weights=1/slope_dist)
            # Average of all ensemble members
            ensemble_simulation = np.average(SMB_ensemble)
        else:
            # Unweighted ensemble average
            ensemble_simulation = np.average(SMB_ensemble)
    else:
    # SMB reconstruction
        # We initialize the empty struture to fill with annual data
        ensemble_data, ensemble_simulation = [],[]
        for year in SMB_ensemble[0]:
            ensemble_data.append([])
        # We fill the data structure
        for member in SMB_ensemble:
            year_idx = 0
            for year in member:
                ensemble_data[year_idx].append(year)
                year_idx = year_idx+1
                
        # We compute the average annual value
        for year in ensemble_data:
            if(CV_ensemble):
                # We generate the weights for the ensemble averaging
#                ge_weights = compute_generalization_error_weights(CV_RMSE, year, np.average(year))
                # Generalization error weighted ensemble
#                ensemble_simulation.append(np.average(year, weights=ge_weights))
                # Inverse slope difference weighted ensemble
#                ensemble_simulation.append(np.average(year, weights=1/slope_dist))
                ensemble_simulation.append(np.average(year))
            else:
                # Unweighted ensemble average
                ensemble_simulation.append(np.average(year))
                
        
            
        ensemble_simulation = np.asarray(ensemble_simulation)
    
#    print("\nAverage simulation: " + str(ensemble_simulation))
    
    # We return the average value of all the ensemble members
    return ensemble_simulation, SMB_ensemble
        
    
def getRasterInfo(raster_current_F19):
    global r_projection
    r_projection = raster_current_F19.GetProjection()
    global r_geotransform
    r_geotransform = raster_current_F19.GetGeoTransform()
    global r_pixelwidth
    r_pixelwidth = r_geotransform[1]
    global r_pixelheight
    r_pixelheight = r_geotransform[-1]
    global r_Xorigin
    r_Xorigin = r_geotransform[0]
    global r_Yorigin
    r_Yorigin = r_geotransform[3]
    global r_origin
    r_origin = (r_Xorigin, r_Yorigin)
    
    return r_projection, r_pixelwidth, r_pixelheight, r_origin

def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]
    
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float64)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(32632)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
# Compute the scaling factor fs for the delta h function
def generate_fs(year_smb, _year_start, DEM_sorted_current_glacier_u, DEM_sorted_CG_n_u, delta_h_DEM_current_glacier, 
                masked_DEM_current_glacier_u, delta_h_dh_current_glacier, _masked_ID_current_glacier_u, pixel_area, glacierArea):
    # We compute the fs factor in order to scale the normalized delta h parameters
    vol_distrib = 0
    
    # If delta-h function is all 0, swap for a flat function 
    if(not np.any(delta_h_dh_current_glacier)):
        delta_h_dh_current_glacier = np.ones(delta_h_dh_current_glacier.shape)
    
    for alt_band, alt_band_n in zip(DEM_sorted_current_glacier_u, DEM_sorted_CG_n_u):
        band_flat_idx = np.where(masked_DEM_current_glacier_u == alt_band)[0]
        area_band = pixel_area*band_flat_idx.size
        
        delta_h_idx, dh_diff = find_nearest(delta_h_DEM_current_glacier, alt_band_n)
        delta_h_i = delta_h_dh_current_glacier[delta_h_idx]
        
        vol_distrib = vol_distrib + area_band*delta_h_i  
        
    fs_id = year_smb*(glacierArea*1000) / (ice_density * vol_distrib) 
    
    return fs_id, delta_h_dh_current_glacier


#####################  TOPOGRAPHICAL ADJUSTMENT  #####################################

# Gets the flowline shapefile for a raster
def get_flowline(glimsID, Length, layer_flowlines):
    found = False
    chosen_diff = 0
    for flowline in layer_flowlines:
        length_diff = abs(Length - flowline.GetField("Length"))
        if((glimsID == flowline.GetField("GLIMS_ID") and (length_diff < chosen_diff)) or glimsID == flowline.GetField("GLIMS_ID") and (length_diff == 0)): 
            found = True
            chosen_diff = length_diff
            chosen_flowline = flowline
        elif(glimsID == flowline.GetField("GLIMS_ID") and (length_diff < Length/100) and (not found)):
            found = True
            chosen_diff = length_diff
            chosen_flowline = flowline
            
    if(found):
        return chosen_flowline
    else:
        print("/!\ NO FLOWLINE FOUND WITH THIS GLIMS_ID")
     
# Retrieves the glacier aspect and converts it to degrees
def get_aspect_deg(aspect):
    if(aspect == 'N'): aspect_deg = 0
    elif(aspect == 'NNE'): aspect_deg = 22.5
    elif(aspect == 'NE'): aspect_deg = 45
    elif(aspect == 'ENE'): aspect_deg = 67.5
    elif(aspect == 'E'): aspect_deg = 90
    elif(aspect == 'ESE'): aspect_deg = 112.5
    elif(aspect == 'SE'): aspect_deg = 135
    elif(aspect == 'SSE'): aspect_deg = 157.5
    elif(aspect == 'S'): aspect_deg = 180
    elif(aspect == 'SSW'): aspect_deg = 202.5
    elif(aspect == 'SW'): aspect_deg = 225
    elif(aspect == 'WSW'): aspect_deg = 247.5
    elif(aspect == 'W'): aspect_deg = 270
    elif(aspect == 'WNW'): aspect_deg = 292.5
    elif(aspect == 'NW'): aspect_deg = 315
    elif(aspect == 'NNW'): aspect_deg = 337.5
    
    return aspect_deg
            
        
 # We crop the initial rasters to the extent of the GLIMS 2003 or 2015 database
def crop_inital_rasters_to_GLIMS(path_glacier_ID_rasters, path_glacier_DEM_rasters, path_glacier_outline, _glacier_shapefile, glacierID, _midfolder, year_start):
    if(year_start == 2004):
        print("glacierID: " + str(glacierID))
        
        if(glacierID == 3638): # If Argentiere glacier, use field data
            current_glacier_ice_depth = path_glacier_ID_rasters + "argentiere_2003_glacioclim.tif"
        else:
            current_glacier_ice_depth = path_glacier_ID_rasters + "RGI60-11.0" + str(glacierID) + "_thickness.tif"
        
        current_glacier_DEM = path_glacier_DEM_rasters + "dem_0" + str(glacierID) + ".asc.tif"
#        
        path_glacier_ID_GLIMS = path_glacier_ID_rasters + "thick_0" + str(glacierID) + "_GLIMS2003.tif"
        path_glacier_DEM_GLIMS = current_glacier_DEM
        
        path_glacier_DEM_2003 = path_glacier_DEM_GLIMS
        path_glacier_ID_2003 = current_glacier_ice_depth
        
        print("Clipping raster to GLIMS 2003 extent... ")
        clipRaster_with_polygon(path_glacier_ID_GLIMS, current_glacier_ice_depth, path_glacier_outline)
        
#        path_glacier_ID_GLIMS = current_glacier_ice_depth
#        clipRaster_with_polygon(path_glacier_DEM_GLIMS, current_glacier_DEM, path_glacier_outline)
        
    elif(year_start == 2015):
        # We open the 2015 projected F19 files
        path_glacier_ID_GLIMS = path_glacier_ID_rasters + "glacier_evolution\\SAFRAN\\1\\" + "IceDepth_Glacier_0" + str(glacierID) + "_2014.tif"
        path_glacier_DEM_GLIMS = path_glacier_DEM_rasters + "glacier_evolution\\SAFRAN\\1\\" + "DEM_Glacier_0" + str(glacierID) + "_2014.tif"
        
        path_glacier_DEM_2003 = path_glacier_DEM_rasters + "dem_0" + str(glacierID) + ".asc.tif"
        path_glacier_ID_2003 = path_glacier_ID_rasters + "RGI60-11.0" + str(glacierID) + "_thickness.tif"
#        path_glacier_ID_2003 = path_glacier_ID_rasters + "thick_0" + str(glacierID) + "_GLIMS2003.tif"
        
#        print("Clipping raster to GLIMS extent... ")
#        clipRaster_with_polygon(path_glacier_ID_GLIMS, current_glacier_ice_depth, path_glacier_outline)
#        clipRaster_with_polygon(path_glacier_DEM_GLIMS, current_glacier_DEM, path_glacier_outline)
        
    elif(year_start == 2019):
        if(glacierID == 3651): # If Tré la Tête glacier, use field data
            path_glacier_ID_GLIMS = path_glacier_ID_rasters + "interp_cleaned_masked_newh_25m.tif"
            path_glacier_DEM_GLIMS = path_glacier_DEM_rasters + "masked_dem_lidar_25m.tif"
            path_glacier_DEM_2003 = ''
            path_glacier_ID_2003 = ''
        else:
            print("\nWARNING: No ice thickness data for this glacier!")
            
        
    return path_glacier_ID_GLIMS, path_glacier_DEM_GLIMS, path_glacier_DEM_2003, path_glacier_ID_2003

# Get points from flowline 
def get_point_values(flowline_feature, dem_raster):
    
    gt=dem_raster.GetGeoTransform()
    rb=dem_raster.GetRasterBand(1)
#
    geom = flowline_feature.GetGeometryRef()
    points = geom.GetPoints()
    flowline_altitudes, flowline_coordinates = [],[]
    for point in points:
        mx,my=point[0], point[1]
        px = int((mx - gt[0]) / gt[1]) 
        py = int((my - gt[3]) / gt[5]) 
        alt = rb.ReadAsArray(px,py,1,1)
        if(alt is not None):
            flowline_altitudes.append(rb.ReadAsArray(px,py,1,1)[0][0])
            flowline_coordinates.append(point)
        else:
            break
    flowline_altitudes = np.asarray(flowline_altitudes)
    flowline_coordinates = np.asarray(flowline_coordinates)
        
    return flowline_altitudes, flowline_coordinates

# Get the lenght of the lowermost 20%
def get_flowline_20_length(flowline_feature, flowline_coords, alt_20_flowline_idx, min_flowline_idx, path_temp_shapefile):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.CreateDataSource(path_temp_shapefile)
    line = ogr.Geometry(type=flowline_feature.GetGeometryRef().GetGeometryType())
    fixed_length = False
    if(alt_20_flowline_idx+1 == flowline_coords.shape[0]):
        alt_20_flowline_idx = alt_20_flowline_idx - 1
        fixed_length = True
    for coords in flowline_coords[alt_20_flowline_idx:min_flowline_idx+1]:
        line.AddPoint(coords[0], coords[1])
    myPoly = ogr.Geometry(type=flowline_feature.GetGeometryRef().GetGeometryType())
    myPoly.AddGeometry(line)
    flowline_20_length = line.Length()
    #flush memory - very important
    datasource.Destroy()
    myPoly.Destroy()
    line.Destroy()
    
    return flowline_20_length, fixed_length

# Computes the slope of the lowermost 20% of a glacier    
def get_slope20(_masked_DEM_current_glacier_u, DEM_sorted_current_glacier_u, glacierName, flowline, path_raster_current_DEM, yearly_glacier_slope20):
    
    raster_current_DEM = gdal.Open(path_raster_current_DEM) 
    
    path_temp_shapefile = path_glacier_DEM_rasters + "aux_vector_" + str(glacierName) + ".shp"
    flowline_altitudes_full, flowline_coordinates = get_point_values(flowline, raster_current_DEM)
    if(flowline_altitudes_full.size == 0):
        print("[ ERROR ] No match between DEM and flowline. Skipping glacier...")
        return -9
    
    flowline_altitudes = flowline_altitudes_full[flowline_altitudes_full>0]
    
    max_alt = DEM_sorted_current_glacier_u.max()
    min_alt = DEM_sorted_current_glacier_u.min()
    
    min_alt_flowline_idx, diff = find_nearest(flowline_altitudes, min_alt)
    min_flowline_alt_u = flowline_altitudes[min_alt_flowline_idx]
    alt_20_threshold = min_flowline_alt_u + (max_alt - min_flowline_alt_u)*0.2
    alt_20_flowline_idx, diff = find_nearest(flowline_altitudes, alt_20_threshold)
    flowline_20_length, fixed_length = get_flowline_20_length(flowline, flowline_coordinates, alt_20_flowline_idx, min_alt_flowline_idx, path_temp_shapefile)

    # We adjust the parameters in case there are problems with the angle calculation
    if(fixed_length):
        alt_20_threshold = flowline_altitudes[alt_20_flowline_idx-1]
    counter = 0
    while((flowline_20_length < (alt_20_threshold - min_flowline_alt_u)) and counter < 20):
        alt_20_flowline_idx = alt_20_flowline_idx-1
        flowline_20_length, fixed_length = get_flowline_20_length(flowline, flowline_coordinates, alt_20_flowline_idx, min_alt_flowline_idx, path_temp_shapefile)
        counter = counter+1
    
    # If the slope cannot be computed, we take the slope from the previous year
    if(counter >= 20) and len(yearly_glacier_slope20) > 0:
        slope20 = yearly_glacier_slope20[-1]
    elif(counter >= 20) and len(yearly_glacier_slope20) < 0:
        slope20 = 50 # Standard value for few steep glaciers whose slope cannot be computed
    else:
        slope20 = np.rad2deg(np.arctan((alt_20_threshold - min_flowline_alt_u)/flowline_20_length))
        # Smooth slope transitions to avoid abrupt changes due to DEMs
        if(len(yearly_glacier_slope20) > 0):
            if(slope20 < yearly_glacier_slope20[-1]*0.8):
                slope20 = yearly_glacier_slope20[-1]*0.8
    
    if(slope20 > 55):
        slope20 = 55 # Limit slope at 55º to avoid unrealistic slopes
        print("\n/!\ GLACIER OVER 55º /!\ \n")
    
    return slope20
    
    
########################   SAFRAN CLIMATIC FORCINGS    ####################################################

# Finds the glacier index in the forcing matrixes
def find_glacier_idx(glacier_massif, massif_number, altitudes, glacier_altitude, aspects, glacier_aspect):
    massif_altitudes_idx = np.where(massif_number == float(glacier_massif))[0]
    glacier_aspect_idx = np.where(aspects == float(glacier_aspect))[0]
    massif_alt_aspect_idx = np.array(list(set(massif_altitudes_idx).intersection(glacier_aspect_idx)))
    index_alt = find_nearest(altitudes[massif_alt_aspect_idx], glacier_altitude)[0]
    final_idx = int(massif_alt_aspect_idx[index_alt])
    
    return final_idx

@jit
# Get the glacier information
def find_glacier_coordinates(massif_number, zs, aspects, glims_data):
    glacier_centroid_altitude = glims_data['MEDIAN_Pixel']
    GLIMS_IDs = glims_data['GLIMS_ID']
    glacier_massifs = glims_data['Massif_SAFRAN']
    glacier_names = glims_data['Glacier']
    glacier_aspects = glims_data['Aspect_num']
    all_glacier_coordinates = []
    
    # All glaciers loop
    for glims_id, glacier_name, glacier_massif, glacier_altitude, glacier_aspect in zip(GLIMS_IDs, glacier_names, glacier_massifs, glacier_centroid_altitude, glacier_aspects):
        all_glacier_coordinates.append([glacier_name, find_glacier_idx(glacier_massif, massif_number, zs, glacier_altitude, aspects, glacier_aspect), float(glacier_altitude), glims_id, int(glacier_massif)])
        
    return np.asarray(all_glacier_coordinates)

def get_SAFRAN_glacier_coordinates(glims_dataset):
     # We read the first year to get some basic information
    dummy_SAFRAN_forcing = Dataset(path_safran_forcings + '84\\FORCING.nc')
    
    aspects = dummy_SAFRAN_forcing.variables['aspect'][:]
    zs = dummy_SAFRAN_forcing.variables['ZS'][:]
    massif_number = dummy_SAFRAN_forcing.variables['massif_number'][:]
    
    all_glacier_coordinates = find_glacier_coordinates(massif_number, zs, aspects, glims_dataset)
    
    return all_glacier_coordinates

@jit
# Mean daily temperature
def get_mean_temps(datetimes, hourly_data):
    ref_day = -9
    daily_data = []
    idx, day_idx = 0, 0
    first = True
    for time_hour in datetimes:
        current_day = time_hour.astype(object).timetuple().tm_yday
        if(current_day == ref_day):
            daily_data[day_idx] = daily_data[day_idx] + hourly_data[idx]/24.0
        else:
            ref_day = current_day
            daily_data.append(hourly_data[idx]/24.0)
            if(not first):
                day_idx = day_idx + 1
            else:
                first = False
            
        idx = idx + 1 
        
    return np.asarray(daily_data)

@jit
# Daily precipitations
def get_precips(datetimes, hourly_data):
    ref_day = -9
    daily_data = []
    idx, day_idx = 0, 0
    isFirst = True
    for time_hour in datetimes:
        current_day = time_hour.astype(object).timetuple().tm_yday
        if(current_day == ref_day):
            daily_data[day_idx] = daily_data[day_idx] + hourly_data[idx]
        else:
            ref_day = current_day
            daily_data.append(hourly_data[idx])
            if(not isFirst):                 
                day_idx = day_idx + 1
            else:
                isFirst = False
        idx = idx + 1 
    return np.asarray(daily_data)

def get_monthly_temps(daily_data, daily_datetimes):
    
    d = {'Dates': daily_datetimes, 'Temps': daily_data}
    df_datetimes = pd.DataFrame(data=d)
    df_datetimes.set_index('Dates', inplace=True)
    df_datetimes.index = pd.to_datetime(df_datetimes.index)
    df_datetimes = df_datetimes.resample('M').mean()
    
    monthly_avg_data = df_datetimes.Temps.to_numpy()
    
    return monthly_avg_data[:12]
    
def get_monthly_snow(daily_data, daily_datetimes):
    d = {'Dates': daily_datetimes, 'Snow': daily_data}
    df_datetimes = pd.DataFrame(data=d)
    df_datetimes.set_index('Dates', inplace=True)
    df_datetimes.index = pd.to_datetime(df_datetimes.index)
    df_datetimes = df_datetimes.resample('M').sum()
    
    monthly_avg_data = df_datetimes.Snow.to_numpy()
    
    return monthly_avg_data[:12]

def compute_local_anomalies(glacier_CPDD, glacier_winter_snow, glacier_summer_snow, meteo_refs):
    
    local_CPDD_anomaly = glacier_CPDD - meteo_refs['CPDD']
    local_w_snow_anomaly = glacier_winter_snow - meteo_refs['w_snow']
    local_s_snow_anomaly = glacier_summer_snow - meteo_refs['s_snow']
    
    return local_CPDD_anomaly, local_w_snow_anomaly, local_s_snow_anomaly

def compute_monthly_anomalies(mon_temps, mon_snow, mon_temp_ref, mon_snow_ref):
    
    mon_temp_anomaly = mon_temps - mon_temp_ref
    mon_snow_anomaly = mon_snow - mon_snow_ref
    
    return mon_temp_anomaly, mon_snow_anomaly

# Fetches the preprocessed SAFRAN daily data 
def get_default_SAFRAN_forcings(safran_start, safran_end):
    
    path_temps = path_smb_function_safran +'daily_temps_years_' + str(safran_start) + '-' + str(safran_end) + '.txt'
    path_snow = path_smb_function_safran +'daily_snow_years_' + str(safran_start) + '-' + str(safran_end) + '.txt'
    path_rain = path_smb_function_safran +'daily_rain_years_' + str(safran_start) + '-' + str(safran_end) + '.txt'
    path_dates = path_smb_function_safran +'daily_dates_years_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_zs = path_smb_function_safran +'zs_years' + str(safran_start) + '-' + str(safran_end) + '.txt'
    
    if(os.path.exists(path_temps) & os.path.exists(path_snow) & os.path.exists(path_rain) & os.path.exists(path_dates) & os.path.exists(path_zs)):
        print("\nFetching SAFRAN forcings...")
        
        with open(path_temps, 'rb') as temps_f:
            daily_temps_years = np.load(temps_f, encoding='latin1',  allow_pickle=True)
        with open(path_snow, 'rb') as snow_f:
            daily_snow_years = np.load(snow_f, encoding='latin1',  allow_pickle=True)
        with open(path_rain, 'rb') as rain_f:
            daily_rain_years = np.load(rain_f, encoding='latin1',  allow_pickle=True)
        with open(path_dates, 'rb') as dates_f:
            daily_dates_years = np.load(dates_f, encoding='latin1',  allow_pickle=True)
        with open(path_zs, 'rb') as zs_f:
            zs_years = np.load(zs_f, encoding='latin1',  allow_pickle=True)[0]
            
    else:
        sys.exit("\n[ ERROR ] SAFRAN base forcing files not found. Please execute SAFRAN forcing module before")
        
    daily_meteo_data = {'temps':daily_temps_years, 'snow': daily_snow_years, 'rain': daily_rain_years, 'dates': daily_dates_years, 'zs': zs_years}
    
    return daily_meteo_data

# Adjusts the daily SAFRAN data for each glacier for a specific year
def get_adjusted_glacier_SAFRAN_forcings(year, year_start, glacier_mean_altitude, SAFRAN_idx, daily_meteo_data, meteo_refs):
    
    # We also need to fetch the previous year since data goes from 1st of August to 31st of July
    idx = year - year_start 
    glacier_idx = int(SAFRAN_idx)
    t_lim = 2.0
#    print("Year idx: " + str(idx))
    
#    import pdb; pdb.set_trace()
    
    # Retrieve raw meteo data for the current year
    zs = daily_meteo_data['zs'][idx][0]
    dates = daily_meteo_data['dates'][idx]
    
    safran_tmean_d = xr.DataArray(daily_meteo_data['temps'][idx], coords=[dates, zs], dims=['time', 'zs'])
    safran_snow_d = xr.DataArray(daily_meteo_data['snow'][idx], coords=[dates, zs], dims=['time', 'zs'])
    safran_rain_d = xr.DataArray(daily_meteo_data['rain'][idx], coords=[dates, zs], dims=['time', 'zs'])
    
    # Re-scale temperature at glacier's actual altitude
    safran_tmean_d_g = copy.deepcopy(safran_tmean_d[:, glacier_idx] + ((zs[glacier_idx] - glacier_mean_altitude)/1000.0)*6.0)
    
    # We adjust the snowfall rate at the glacier's altitude
    safran_snow_d_g = copy.deepcopy(safran_snow_d[:, glacier_idx])
    safran_rain_d_g = copy.deepcopy(safran_rain_d[:, glacier_idx])
    safran_snow_d_g.data = np.where(safran_tmean_d_g.data > t_lim, 0.0, safran_snow_d_g.data)
    safran_snow_d_g.data = np.where(safran_tmean_d_g.data < t_lim, safran_snow_d_g.data + safran_rain_d_g.data, safran_snow_d_g.data)
    
    # Monthly data during the current hydrological year
    safran_tmean_m_g = safran_tmean_d_g.resample(time="1MS").mean().data
    safran_snow_m_g = safran_snow_d_g.resample(time="1MS").sum().data
    
    # Compute CPDD
    # Compute dask arrays prior to storage
    glacier_CPDD = np.sum(np.where(safran_tmean_d_g.data < 0, 0, safran_tmean_d_g.data))
    
    # Compute snowfall
    # Compute dask arrays prior to storage
    glacier_winter_snow = np.sum(safran_snow_d_g.sel(time = slice(str(year-1)+'-10-01', str(year)+'-03-31')).data)
    glacier_summer_snow = np.sum(safran_snow_d_g.sel(time = slice(str(year)+'-04-01', str(year)+'-07-31')).data)
    
    # We compute the seasonal anomalies
    CPDD_LocalAnomaly, winter_snow_LocalAnomaly, summer_snow_LocalAnomaly = compute_local_anomalies(glacier_CPDD, 
                                                                                                    glacier_winter_snow, 
                                                                                                    glacier_summer_snow,
                                                                                                    meteo_refs)

    # We compute the monthly anomalies
    mon_temp_anomaly, mon_snow_anomaly = compute_monthly_anomalies(safran_tmean_m_g, safran_snow_m_g, 
                                                                   meteo_refs['mon_temp'], meteo_refs['mon_snow'])
    
    season_anomalies_y = {'CPDD': CPDD_LocalAnomaly, 'winter_snow':winter_snow_LocalAnomaly, 'summer_snow': summer_snow_LocalAnomaly}
    monthly_anomalies_y = {'temps': mon_temp_anomaly, 'snow': mon_snow_anomaly}
    
    return season_anomalies_y,  monthly_anomalies_y


#####  ADAMONT FORCINGS  #######
    
def find_adamont_glacier_idx(massif_idx, glacier_altitude, massif_number, altitudes):
    massif_altitudes_idx = np.where(massif_number == massif_idx)[0]
    index_alt = find_nearest_altitude(altitudes[massif_altitudes_idx], glacier_altitude)
    final_idx = massif_altitudes_idx[index_alt]
    
    return final_idx

def get_ADAMONT_idx(massif, glacier_altitude, massif_number, zs):
    ADAMONT_idx = find_adamont_glacier_idx(massif, glacier_altitude, massif_number, zs)
    return ADAMONT_idx

    
def get_default_ADAMONT_forcings(year_start, year_end, midfolder):
    
    path_temps = path_smb_function_adamont + midfolder + 'daily_temps_years_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_snow = path_smb_function_adamont + midfolder + 'daily_snow_years_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_rain = path_smb_function_adamont + midfolder + 'daily_rain_years_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_dates = path_smb_function_adamont + midfolder + 'daily_datetimes_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_zs = path_smb_function_adamont + midfolder + 'zs_' + str(year_start) + '-' + str(year_end) + '.txt'
    path_massif = path_smb_function_adamont + midfolder + 'massif_number.txt'
    path_aspect = path_smb_function_adamont + midfolder + 'aspects.txt'
    
    if(os.path.exists(path_temps) & os.path.exists(path_snow) & os.path.exists(path_rain) & os.path.exists(path_dates) & os.path.exists(path_zs) & os.path.exists(path_massif) & os.path.exists(path_aspect)):
        print("Fetching ADAMONT forcings...")
        with open(path_temps, 'rb') as temps_f:
            daily_temps_years = np.load(temps_f,  allow_pickle=True)
        with open(path_snow, 'rb') as snow_f:
            daily_snow_years = np.load(snow_f, allow_pickle=True)
        with open(path_rain, 'rb') as rain_f:
            daily_rain_years = np.load(rain_f, allow_pickle=True)
        with open(path_dates, 'rb') as dates_f:
            daily_datetimes = np.load(dates_f, allow_pickle=True)
        with open(path_zs, 'rb') as zs_f:
            zs = np.load(zs_f, allow_pickle=True)
        with open(path_massif, 'rb') as massif_f:
            massif_number = np.load(massif_f, allow_pickle=True)
        with open(path_aspect, 'rb') as aspects_f:
            aspects = np.load(aspects_f, allow_pickle=True)
        
        daily_meteo_data = {'temps':daily_temps_years, 'snow': daily_snow_years, 'rain': daily_rain_years, 'dates': daily_datetimes, 'zs': zs}
    else:
        # We read all the files
        print("\nRe-computing ADAMONT forcings...")
        
        forcing_daymean = settings.current_ADAMONT_model_daymean
        forcing_daysum = settings.current_ADAMONT_model_daysum
        print("\nCurrent ADAMONT combination: " + str(forcing_daymean) + "\\n")
        
        start = time.time()
         # We load the two ADAMONT files
        adamont_mean_climate = xr.open_dataset(forcing_daymean, parallel=True)
        adamont_sum_climate = xr.open_dataset(forcing_daysum, parallel=True)
        
        # Rename the time coordinates to match the SAFRAN format
        adamont_mean_climate = adamont_mean_climate.rename({"TIME": "time"})
        adamont_sum_climate = adamont_sum_climate.rename({"TIME": "time"})
            
        end = time.time()
        print("\n-> open SAFRAN dataset processing time: " + str(end - start) + " s")
        
        zs = adamont_mean_climate['ZS'].compute()
        massif_number = adamont_mean_climate['MASSIF_NUMBER'].compute()
        aspects = np.repeat(-1, len(zs))
        
        daily_temps_years, daily_snow_years, daily_rain_years, daily_datetimes = np.array([]),np.array([]),np.array([]),np.array([])
            
        for year in year_period: 
            print("Hydrological year: " + str(year-1) + "-" + str(year))
            
            start = time.time()
            # We load into memory only the current year to speed things up
            # Only two years are loaded: compute dask arrays in memory so computations are faster
            safran_tmean_d = (adamont_mean_climate.sel(time = slice(str(year-1)+'-10-01', str(year)+'-09-30'))['Tair'].resample(time="1D").mean() -273.15).compute()
            safran_snow_d = (adamont_sum_climate.sel(time = slice(str(year-1)+'-10-01', str(year)+'-09-30'))['SNOW'].resample(time="1D").sum()).compute()
            safran_rain_d = (adamont_sum_climate.sel(time = slice(str(year-1)+'-10-01', str(year)+'-09-30'))['RAIN'].resample(time="1D").sum()).compute()
            
            # Store daily raw data for future re-processing
            daily_temps_years = np.append(daily_temps_years, safran_tmean_d)
            daily_snow_years = np.append(daily_snow_years, safran_snow_d)
            daily_rain_years = np.append(daily_rain_years, safran_rain_d)
            daily_datetimes = np.append(daily_datetimes, safran_tmean_d.time)
        
        daily_meteo_data = {'temps':daily_temps_years, 'snow': daily_snow_years, 'rain': daily_rain_years, 'dates': daily_datetimes, 'zs': zs}
        
        # We create the folder if it's not there
        if(not os.path.exists(path_smb_function_adamont+midfolder)):
            os.makedirs(path_smb_function_adamont+midfolder)
            
        with open(path_smb_function_adamont+midfolder+'daily_temps_years_' + str(year_start) + '-' + str(year_end) + '.txt', 'wb') as dtemp_f:
                            np.save(dtemp_f, daily_temps_years)
        with open(path_smb_function_adamont+midfolder+'daily_snow_years_' + str(year_start) + '-' + str(year_end) + '.txt', 'wb') as dsnow_f:
                            np.save(dsnow_f, daily_snow_years)
        with open(path_smb_function_adamont+midfolder+'daily_rain_years_' + str(year_start) + '-' + str(year_end) + '.txt', 'wb') as drain_f:
                            np.save(drain_f, daily_rain_years)
        with open(path_smb_function_adamont+midfolder+'daily_datetimes_' + str(year_start) + '-' + str(year_end) + '.txt', 'wb') as ddates_f:
                            np.save(ddates_f, daily_datetimes)
        with open(path_smb_function_adamont+midfolder+'zs_' + str(year_start) + '-' + str(year_end) + '.txt', 'wb') as dzs_f:
                            np.save(dzs_f, zs)
        with open(path_smb_function_adamont+midfolder+'massif_number.txt', 'wb') as massif_f:
                            np.save(massif_f, massif_number.data)
        with open(path_smb_function_adamont+midfolder+'aspects.txt', 'wb') as aspects_f:
                            np.save(aspects_f, aspects.data)
                            
    return daily_meteo_data, massif_number, aspects, year_end

def get_adjusted_glacier_ADAMONT_forcings(year, year_start, glacier_mean_altitude, ADAMONT_idx, daily_meteo_data, meteo_anomalies):
    # We also need to fetch the previous year since data goes from 1st of August to 31st of July
    idx = year - year_start 
#    print("ADAMONT_idx: " + str(ADAMONT_idx))
    glacier_idx = int(ADAMONT_idx)
    
    safran_tmean_d = daily_meteo_data['temps'][idx]
    safran_snow_d = daily_meteo_data['snow'][idx]
    safran_rain_d = daily_meteo_data['rain'][idx]
    zs = daily_meteo_data['zs'][idx]
    daily_datetimes_years = daily_meteo_data['dates'][idx]
    
    # Re-scale temperature at glacier's actual altitude
    safran_tmean_d_g = copy.deepcopy(safran_tmean_d[:, glacier_idx] + ((zs[0,glacier_idx].data - glacier_alt_y)/1000.0)*6.0)
    
    # We adjust the snowfall rate at the glacier's altitude
    safran_snow_d_g = copy.deepcopy(safran_snow_d[:, glacier_idx])
    safran_rain_d_g = copy.deepcopy(safran_rain_d[:, glacier_idx])
    safran_snow_d_g.data = np.where(safran_tmean_d_g.data > 0.0, 0.0, safran_snow_d_g.data)
    safran_snow_d_g.data = np.where(((safran_tmean_d_g.data < 0.0) & (safran_snow_d_g.data == 0.0)), safran_rain_d_g.data, safran_snow_d_g.data)
    
    # Monthly data during the current hydrological year
    # Compute dask arrays prior to storage
    safran_tmean_m_g = safran_tmean_d_g.resample(time="1MS").mean().data.compute()
    safran_snow_m_g = safran_snow_d_g.resample(time="1MS").sum().data.compute()
    
    # Compute CPDD
    # Compute dask arrays prior to storage
    glacier_CPDD = np.sum(np.where(safran_tmean_d_g.data < 0, 0, safran_tmean_d_g.data)).compute()
    glacier_CPDDs_all[j]['CPDD'].append(year)
    glacier_CPDDs_all[j]['CPDD'].append(glacier_CPDD)
    
    # Compute snowfall
    # Compute dask arrays prior to storage
    glacier_winter_snow = np.sum(safran_snow_d_g.sel(time = slice(str(year-1)+'-10-01', str(year)+'-03-31')).data).compute()
    glacier_summer_snow = np.sum(safran_snow_d_g.sel(time = slice(str(year)+'-04-01', str(year)+'-07-31')).data).compute()
    
    # Seasonal anomalies
    CPDD_LocalAnomaly, winter_snow_LocalAnomaly, summer_snow_LocalAnomaly = compute_local_anomalies(glacier_CPDD, glacier_winter_snow, glacier_summer_snow,
                                                                                                    meteo_anomalies)                                                         
    
    # Monthly anomalies
    mon_temp_anomaly, mon_snow_anomaly = compute_monthly_anomalies(safran_tmean_m_g, safran_snow_m_g, meteo_anomalies['mon_temp'], meteo_anomalies['mon_snow'])
         
    season_anomalies_y = {'CPDD': CPDD_LocalAnomaly, 'winter_snow':winter_snow_LocalAnomaly, 'summer_snow': summer_snow_LocalAnomaly}
    monthly_anomalies_y = {'temps': mon_temp_anomaly, 'snow': mon_snow_anomaly}
    
#    print("\nCPDD_LocalAnomaly: " + str(CPDD_LocalAnomaly))
#    print("winter_snow_LocalAnomaly: " + str(winter_snow_LocalAnomaly))
#    print("summer_snow_LocalAnomaly: " + str(summer_snow_LocalAnomaly))
#    
#    print("\nmon_temp_anomaly: " + str(mon_temp_anomaly))
#    print("mon_snow_anomaly: " + str(mon_snow_anomaly))
    
    return season_anomalies_y, monthly_anomalies_y

# Retrieves the mean meteo values to compute the anomalies
def get_meteo_references(season_meteo_SMB, monthly_meteo_SMB, glimsID, glacierName):
    found = False
    glacier_CPDDs = season_meteo_SMB['CPDD']
    glacier_winter_snow = season_meteo_SMB['winter_snow']
    glacier_summer_snow = season_meteo_SMB['summer_snow']
    glacier_mon_temps = monthly_meteo_SMB['temp']
    glacier_mon_snow = monthly_meteo_SMB['snow']
    
    
    for cpdd, w_snow, s_snow, mon_temps, mon_snow in zip(glacier_CPDDs, glacier_winter_snow, glacier_summer_snow, glacier_mon_temps, glacier_mon_snow):
        if(cpdd['GLIMS_ID'] == glimsID):
            if((found and cpdd['Glacier'] == glacierName) or not found):
                CPDD_ref = cpdd['Mean']
                w_snow_ref = w_snow['Mean']
                s_snow_ref = s_snow['Mean'] 
                
                mon_temp_ref = mon_temps['mon_means']
                mon_snow_ref = mon_snow['mon_means']
                
                found = True
          
    return CPDD_ref, w_snow_ref, s_snow_ref, mon_temp_ref, mon_snow_ref

def store_plot(masked_ID_current_glacier, masked_DEM_current_glacier_u, masked_ID_current_glacier_u, masked_ID_previous_glacier_u, ID_difference_current_glacier,glacierName, yearly_glacier_zmean, SMB_y, year, nfigure, newpath, isNotFirst):
    if(isNotFirst):
        diff_max = 20
        diff_min = -25
        
        if(masked_ID_current_glacier.min() == masked_ID_current_glacier.max()):
            id_min = 0
            id_max = 5
        else:
            if(masked_ID_current_glacier.min() < 0):
                id_min = 0
            else:
                id_min = masked_ID_current_glacier.min()
            id_max = masked_ID_current_glacier.max()

        plt.figure(nfigure, figsize=(18,11))
        plt.subplot(1,2,1)
        plt.title("Glacier " + glacierName + "  |  Hydrological year: " + str(year-1) + "-" + str(year), fontsize=20, y=1.05)
        plt.imshow(masked_ID_current_glacier_u, cmap="Blues", vmin=id_min, vmax=id_max)
        cb1 = plt.colorbar(boundaries=np.linspace(0.0, masked_ID_current_glacier.max(), 20), orientation="horizontal")
        cb1.set_label("Ice depth (m)", fontsize=15)
        scalebar = ScaleBar(25) # 1 pixel = 25 meters
        scalebar.border_pad = 0.5
        plt.gca().add_artist(scalebar)
        plt.text(1, 1, 'Z mean: ' + str(round(yearly_glacier_zmean[-1])) + " m", size='large', horizontalalignment='left', verticalalignment='top')
        
        plt.subplot(1,2,2)
        if(SMB_y <= 0):
            if(ID_difference_current_glacier.compressed().size > 0):
                diff_ice_max = round(ID_difference_current_glacier.compressed().min(), 2)
            else:
                diff_ice_max = masked_ID_previous_glacier_u.compressed().max()
        else:
            diff_ice_max = round(ID_difference_current_glacier.compressed().max(), 2)

        plt.title("SMB: " + str(round(SMB_y, 2)) + " m w.e.  |  Max ice diff: " + str(diff_ice_max) + " m", fontsize=20, y=1.05)
        # We change the color bar depending on the SMB range of values for the glacier
        plt.imshow(ID_difference_current_glacier, cmap="RdYlBu", vmin=diff_min, vmax=diff_max)
        cb2 = plt.colorbar(boundaries=np.linspace(diff_min, diff_max, 30), ticks=[range(diff_min, diff_max, 5)], orientation="horizontal")

        cb2.set_label("Ice depth difference (m)", fontsize=15)
        scalebar = ScaleBar(36) # 1 pixel = 36 meter
        scalebar.border_pad = 0.5
        plt.gca().add_artist(scalebar)

    else:
        plt.figure(nfigure)
        plt.title("Glacier " + glacierName + "  |  Year: " + str(year) + " |  SMB: " + str(round(SMB_y)) + " m w.e.", fontsize=10, y=1.05)
        plt.imshow(masked_ID_current_glacier_u, cmap="Blues", vmin=0.0, vmax=masked_ID_current_glacier.max())
        cb3 = plt.colorbar(boundaries=np.linspace(0, masked_ID_current_glacier.max(), 20), orientation="horizontal")
        cb3.set_label("Ice depth Glacier (m) " + glacierName, fontsize=10)
        scalebar = ScaleBar(25) # 1 pixel = 25 meters
        scalebar.border_pad = 0.5
        plt.gca().add_artist(scalebar)
        plt.text(1, 0.5, 'Z mean: ' + str(round(masked_DEM_current_glacier_u.compressed().mean())) + " m", size='medium', horizontalalignment='left', verticalalignment='top')
        
#                plt.show(block=False)
    # We store the plots as images
    plt.savefig(newpath + "Glacier " + glacierName + "_" + str(year))
    plt.close()
    
    nfigure = nfigure+1

    isNotFirst = True
    
    return nfigure, isNotFirst

def store_rasters(masked_DEM_current_glacier_u, masked_ID_current_glacier_u, midfolder, glacierID, year):
    path_DEM_raster_year = path_glacier_evolution_DEM_rasters + midfolder + "DEM_Glacier_0" + str(glacierID) + "_" + str(year) + ".tif"
    if not os.path.exists(path_glacier_evolution_DEM_rasters + midfolder):
        os.makedirs(path_glacier_evolution_DEM_rasters + midfolder)
    path_ID_raster_year = path_glacier_evolution_ID_rasters + midfolder + "IceDepth_Glacier_0" + str(glacierID) + "_" + str(year) + ".tif"
    if not os.path.exists(path_glacier_evolution_ID_rasters + midfolder):
        os.makedirs(path_glacier_evolution_ID_rasters + midfolder)
    array2raster(path_DEM_raster_year, r_origin, r_pixelwidth, r_pixelheight, masked_DEM_current_glacier_u)
    array2raster(path_ID_raster_year, r_origin, r_pixelwidth, r_pixelheight, masked_ID_current_glacier_u)
    
    return path_DEM_raster_year

###########################  GLACIER EVOLUTION  ##############################################
        
def glacier_evolution(masked_DEM_current_glacier, masked_ID_current_glacier, 
                      delta_h_dh_current_glacier, delta_h_DEM_current_glacier, 
                      _DEM_sorted_current_glacier,  
                      daily_meteo_data, meteo_anomalies,
                      flowline, _raster_current_DEM, current_glacier_DEM, store_plots, 
                      glacierName, glacierID, glimsID, massif, lat, lon, aspect,
                      midfolder, pixel_area, glaciers_with_errors, glims_rabatel,
                      lasso_scaler, lasso_model, ensemble_SMB_models, 
                      year_range, ref_start, _ref_end, SAFRAN_idx, overwrite):
    
    print("Applying glacier evolution...")
       
    # We make deep copies of the original DEM and ice depth distribution to start updating them
    masked_ID_current_glacier_u = copy.deepcopy(masked_ID_current_glacier)
    masked_ID_current_glacier_u.mask = np.ma.make_mask(np.where(masked_ID_current_glacier_u > 0, 0, 1))
    masked_DEM_current_glacier_u = np.ma.array(np.ma.getdata(masked_DEM_current_glacier), mask = np.ma.make_mask(np.where(masked_ID_current_glacier_u > 0, 0, 1)))
    
    # We sort the DEM by altitude in order to have the altitudinal range ready for iteration
    DEM_sorted_current_glacier_u = np.sort(masked_DEM_current_glacier_u.compressed(), axis=None)
    DEM_sorted_current_glacier_u = np.unique(DEM_sorted_current_glacier_u[DEM_sorted_current_glacier_u > 0])
    DEM_sorted_CG_n_u = normalize_dem(DEM_sorted_current_glacier_u)
    
    nfigure = 1
    isNotFirst = False
    glacier_melted_flag = False
    glacier_melt_year = []
    
    # We get the glacier meteorological references
    year_start = year_range[0]   
    year = year_start
    
    # We shorten the name of the glacier if it's too long
    glacierName = shorten_name(glacierName)
    newpath = path_glacier_evolution_plots +  midfolder + strip_accents(massif) + '\\' + "Glacier " + strip_accents(glacierName) + "\\"
    
    path_raster_current_DEM = current_glacier_DEM
    
    if not os.path.exists(newpath) or overwrite:
        # We create a new folder in order to store the raster plots
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        current_glacierArea = pixel_area*(masked_ID_current_glacier_u.compressed().size)
        yearly_glacier_area, yearly_glacier_volume = [], []
        yearly_glacier_zmean, yearly_glacier_slope20 = [], []
        
        yearly_simulated_SMB = []
        year_range = np.asarray(year_range)
        
        mean_CPDD, mean_snow = [],[]
        
        for year in year_range:
#        for CPDD_Anomaly, winter_snow_Anomaly, summer_snow_Anomaly in zip(raw_CPDD_LocalAnomaly, raw_winter_snow_LocalAnomaly, raw_summer_snow_LocalAnomaly):
            masked_ID_previous_glacier_u = copy.deepcopy(masked_ID_current_glacier_u)
            print("\n--- Hydrological year: " + str(year-1) + "-" + str(year) + " ---\n")
            print("Glacier front: " + str(DEM_sorted_current_glacier_u[0]) + " meters")
            
            ####  RECALCULATION OF TOPOGRAPHICAL PARAMETERS  ####
            mean_glacier_alt = masked_DEM_current_glacier_u.mean()
            max_glacier_alt = masked_DEM_current_glacier_u.max()
            slope20 = get_slope20(masked_DEM_current_glacier_u, DEM_sorted_current_glacier_u, glacierName, flowline, path_raster_current_DEM, yearly_glacier_slope20)
            
            # Skip glacier if slope cannot be computed
            if(slope20 == -9):
                glaciers_with_errors.append(glacierName)
                break
            
            ####  METEOROLOGICAL FORCINGS  ####
            if(settings.projection_forcing == "SAFRAN"):
                season_anomalies_y,  monthly_anomalies_y = get_adjusted_glacier_SAFRAN_forcings(year, ref_start, 
                                                                                             masked_DEM_current_glacier_u.compressed().mean(), SAFRAN_idx, 
                                                                                             daily_meteo_data, meteo_anomalies)
            elif(settings.projection_forcing == "ADAMONT"):
                season_anomalies_y, monthly_anomalies_y = get_adjusted_glacier_ADAMONT_forcings(year, year_start, 
                                                                                             masked_DEM_current_glacier_u.compressed().mean(), SAFRAN_idx, 
                                                                                             daily_meteo_data, meteo_anomalies)
                
            ####  CREATION OF THE MODEL TEST DATASET  ####
            x_lasso, x_ann = create_input_array(season_anomalies_y, monthly_anomalies_y, mean_glacier_alt, max_glacier_alt, slope20, current_glacierArea, lat, lon, aspect)
            
            ####   We simulate the annual glacier-wide SMB  ####
            # Data is scaled as during training 
            if(settings.smb_model_type == "lasso"):
                SMB_y = lasso_model.predict(lasso_scaler.transform(x_lasso.reshape(1,-1)))
            elif(settings.smb_model_type == "ann_no_weights" or settings.smb_model_type == "ann_weights"):
                # We use an ensemble approach to compute the glacier-wide SMB
                batch_size = 34
                SMB_y, SMB_ensemble = make_ensemble_simulation(ensemble_SMB_models, x_ann, batch_size, glimsID, glims_rabatel, evolution=True)
            
            yearly_simulated_SMB.append(SMB_y)
            print("Simulated SMB: " + str(SMB_y))
            
            ####  Fs PARAMETER GENERATION  ####
            # If the glacier is smaller than 0.5 km2 we no longer apply the customized delta h function
            if(current_glacierArea < 0.5):
                delta_h_DEM_current_glacier = np.ones(50)
                delta_h_dh_current_glacier = np.ones(50)
            # We generate each year's fs constant relying on the updated glacier area
            year_fs, delta_h_dh_current_glacier = generate_fs(SMB_y, year_start, DEM_sorted_current_glacier_u, DEM_sorted_CG_n_u, delta_h_DEM_current_glacier,
                                  masked_DEM_current_glacier_u, delta_h_dh_current_glacier, masked_ID_current_glacier_u, pixel_area, current_glacierArea)
            
            ####  ANNUAL ICE THICKNESS UPDATE  ####
            for alt_band, alt_band_n in zip(DEM_sorted_current_glacier_u, DEM_sorted_CG_n_u):
                band_full_idx = np.where(masked_DEM_current_glacier_u == alt_band)
                # We choose the delta h function depending on the SMB (positive or negative)
                delta_h_idx, dh_diff = find_nearest(delta_h_DEM_current_glacier, alt_band_n)
                delta_h_i = delta_h_dh_current_glacier[delta_h_idx]
                
                # We update the glacier's DEM and Ice Depth rasters
                ID_alt_band_i = masked_ID_current_glacier_u[band_full_idx]
                ID_alt_band_i = ID_alt_band_i + year_fs*delta_h_i
                ID_alt_band_i[ID_alt_band_i < 0] = 0
                masked_ID_current_glacier_u[band_full_idx] = copy.deepcopy(ID_alt_band_i)
                np.ma.set_fill_value(masked_ID_current_glacier_u, 0.0)
                masked_ID_current_glacier_u[masked_ID_current_glacier_u <= 0] = np.ma.masked
            
            ice_idx = np.where(masked_ID_current_glacier_u > 0)
            current_glacierArea = pixel_area*(masked_ID_current_glacier_u[ice_idx]).size
            # Gather topographical data evolution
            yearly_glacier_area.append(copy.deepcopy(current_glacierArea))
            yearly_glacier_volume.append(pixel_area*(np.sum(masked_ID_current_glacier_u[ice_idx])))
            yearly_glacier_slope20.append(slope20)
            # Gather climate data evolution
            mean_CPDD.append(season_anomalies_y['CPDD'])
            mean_snow.append(season_anomalies_y['winter_snow'] + season_anomalies_y['summer_snow'])
            
            ID_difference_current_glacier = masked_ID_current_glacier_u - masked_ID_previous_glacier_u
            masked_DEM_current_glacier_u = masked_DEM_current_glacier_u + ID_difference_current_glacier
#            masked_DEM_current_glacier_u[masked_ID_current_glacier_u <= 0] = np.ma.masked
            masked_DEM_current_glacier_u = np.ma.masked_where(masked_ID_current_glacier_u.data <= 0, masked_DEM_current_glacier_u)
            
            if(len(masked_DEM_current_glacier_u.compressed()) > 0):
#                import pdb; pdb.set_trace()
                yearly_glacier_zmean.append(masked_DEM_current_glacier_u.compressed().mean())
            else:
                if(len(yearly_glacier_zmean) > 0):
                    yearly_glacier_zmean.append(yearly_glacier_zmean[-1])
                else:
                    yearly_glacier_zmean.append(mean_glacier_alt)
            
            print("Slope 20%: " + str(slope20))
            print("Area: " + str(current_glacierArea))
            print("Zmean: " + str(yearly_glacier_zmean[-1]))
            
            # We convert and store the updated DEM and ice thickness as rasters for each year
            path_raster_current_DEM = store_rasters(masked_DEM_current_glacier_u, masked_ID_current_glacier_u, midfolder, glacierID, year)
             
            ####  CREATING AND STORING GLACIER EVOLUTION PLOTS  ####
            if(store_plots):
                nfigure, isNotFirst = store_plot(masked_ID_current_glacier, masked_DEM_current_glacier_u, 
                                               masked_ID_current_glacier_u, masked_ID_previous_glacier_u, 
                                               ID_difference_current_glacier, glacierName, yearly_glacier_zmean, 
                                               SMB_y, year, nfigure, newpath, isNotFirst)
            
            # We sort the DEM by altitude in order to have the altitudinal range ready for iteration 
            # Reminder: the DEM masked array is masked based on the ID array (ID > 0)
            if(masked_ID_current_glacier_u[ice_idx].size > 0):
                masked_DEM_current_glacier_u.mask = np.ma.make_mask(np.where(masked_ID_current_glacier_u > 0, 0, 1))
                DEM_sorted_current_glacier_u = np.sort(masked_DEM_current_glacier_u.compressed(), axis=None)
                DEM_sorted_current_glacier_u = np.unique(DEM_sorted_current_glacier_u[DEM_sorted_current_glacier_u > 0])
                DEM_sorted_CG_n_u = normalize_dem(DEM_sorted_current_glacier_u)
            else:
                print("\n ------  Glacier completely melted  ------")
                DEM_sorted_current_glacier_u = np.array([0])
                DEM_sorted_CG_n_u = np.array([0])
                glacier_melted_flag = True
                glacier_melt_year.append(year)
                year = year+1
                break
                
            year = year+1  
        # End of years loop
        
        ####  KEEPING TRACK OF THE EVOLUTION OF KEY TOPOGRAPHICAL PARAMETERS  ####
        # We store the glacier evolution data projections
#        years = range(year_start, year)
        print("\nStoring data...")
        # SMB
        store_file(yearly_simulated_SMB, path_smb_simulations, midfolder, "simu_SMB", glimsID, year_start, year)
        # Area
        store_file(yearly_glacier_area, path_glacier_area, midfolder, "area", glimsID, year_start, year)
        # Volume
        store_file(yearly_glacier_volume, path_glacier_volume, midfolder, "volume", glimsID, year_start, year)
        # Z mean
        store_file(yearly_glacier_zmean, path_glacier_zmean, midfolder, "zmean", glimsID, year_start, year)
        # Slope 20%
        store_file(yearly_glacier_slope20, path_glacier_slope20, midfolder, "slope20", glimsID, year_start, year)
        # Melt year (if available)
        if(glacier_melted_flag):
            if not os.path.exists(path_glacier_melt_years + midfolder):
                os.makedirs(path_glacier_melt_years + midfolder)
            file_name_h = path_glacier_melt_years + midfolder + str(glimsID) + '_'
            file_name_t = '_melt_year.csv'
            glacier_melt_year = np.asarray(glacier_melt_year)
            automatic_file_name_save(file_name_h, file_name_t, glacier_melt_year, 'txt')
        # CPDD
        store_file(mean_CPDD, path_glacier_CPDDs, midfolder, "CPDD", glimsID, year_start, year)
        # Snowfall
        store_file(mean_snow, path_glacier_snowfall, midfolder, "snowfall", glimsID, year_start, year)
        
    else:
        print("Glacier previously processed. Skipping...")
    

    return masked_DEM_current_glacier_u, masked_ID_current_glacier_u




def main(compute, ensemble_SMB_models, overwrite_flag, counter_threshold, thickness_idx):

    ##################################################################################
    ##################		                MAIN                #####################
    #################################################################################
                
    print("\n-----------------------------------------------")
    print("             GLACIER EVOLUTION ")
    print("-----------------------------------------------\n")
    
    if(compute):
        # We close all previous plots
        plt.close('all')
        
        path_smb = workspace + 'glacier_data\\smb\\'
        path_smb_function = path_smb + 'smb_function\\'
        path_glacier_outlines_shapefile = path_glacier_2003_shapefiles + 'GLIMS_glaciers_2003_ID_massif' + '.shp' 
        path_ann = settings.path_ann
        path_safran_forcings = path_smb_function + 'SAFRAN\\'
        
#        if(settings.smb_model_type == 'ann_no_weights'):
#            path_ann_train = path_smb + 'ANN\\LSYGO\\no_weights\\'
#            path_cv_ann = path_ann_train + 'CV\\'
#        elif(settings.smb_model_type == 'ann_weights'):
#            path_ann_train = path_smb + 'ANN\\LSYGO\\weights\\'
#            path_cv_ann = path_ann_train + 'CV\\'
        
        ### We detect the forcing between SAFRAN or ADAMONT
        forcing = settings.projection_forcing
#        print("forcing: " + str(forcing))
        
        # We determine the path depending on the forcing
        path_smb_function_forcing = path_smb_function + forcing + "\\"
        
#        glims_2015 = genfromtxt(path_glims + 'GLIMS_2015_massif.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'),('Aspect_num', '<i8')])
#        glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8'), ('ID', '<f8')])
        glims_rabatel = genfromtxt(path_glims + 'GLIMS_Rabatel_30_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('slope20', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8')])        

        # Flag to determine if raster plots should be stored (time consuming)
        store_plots = True
        
        global ice_density
        ice_density = 850
        global overwrite
        overwrite = overwrite_flag
        global glacier_melted_flag
        glacier_melted_flag = False
        global glacier_melt_year
        
        ### COMMENT IN ORDER TO AVOID RE-COMPUTING FORCINGS  ###
        global overwrite_forcings
        overwrite_forcings = False
#        overwrite_forcings = overwrite_flag
        
        pixel_area = 0.000625 # km2 (25*25 m2, from Farinotti et al. 2019 rasters)
        if(settings.simulation_type == "historical"):
            year_start = 2004 
            year_end = 2015
            ref_start = 1959
            ref_end = 2015
        elif(settings.simulation_type == "future"):
#            year_start = 2015 
            year_start = 2019 
            year_end = 2099
            ref_start = 2006
            ref_end = 2099
        
        #### ONLY HISTORICAL SAFRAN DATA FOR REFS  ####
        # We load the compacted seasonal and monthly meteo forcings
        with open(path_safran_forcings+'season_meteo.txt', 'rb') as season_f:
            season_meteo = np.load(season_f,  allow_pickle=True)[()]
        with open(path_safran_forcings+'monthly_meteo.txt', 'rb') as mon_f:
            monthly_meteo = np.load(mon_f,  allow_pickle=True)[()]
            
        ###  We load the SMB models  ###
        # Deep learning
        # ANN nonlinear models ensemble preloaded separetly
        # Lasso
        # Data scaler
        with open(path_smb_function+'model_lasso_temporal.txt', 'rb') as lasso_model_f:
            lasso_model = np.load(lasso_model_f,  allow_pickle=True)
        # Lasso linear model
        with open(path_smb_function+'LOYO\\full_scaler_temporal.txt', 'rb') as lasso_scaler_f:
            lasso_scaler = np.load(lasso_scaler_f,  allow_pickle=True)[()]
        
        # We open the raster files and shapefiles:
        shapefile_glacier_outlines = ogr.Open(path_glacier_outlines_shapefile)
        layer_glaciers = shapefile_glacier_outlines.GetLayer()
        
        # We recover the list of discarded glaciers by cloud cover
        delta_h_processed_glaciers = np.asarray(genfromtxt(path_delta_h_param+"delta_h_processed_glaciers.csv", delimiter=';', dtype=np.dtype('str')))
        
        # We create the folders to store the glacier area and volume data
        if not os.path.exists(path_glacier_area):
            os.makedirs(path_glacier_area)
        if not os.path.exists(path_glacier_volume):
            os.makedirs(path_glacier_volume)
            
        if(forcing == 'ADAMONT'):
            midfolder_base = str(settings.current_ADAMONT_forcing_mean[:-11]) + "\\"
            daily_meteo_data, massif_number, aspects, year_end = get_default_ADAMONT_forcings(year_start, year_end, midfolder_base)
#            all_glacier_coordinates = get_ADAMONT_glacier_coordinates(glims_2015, massif_number, zs_years)
            print("\nCurrent RCP-GCM-RCM member: " + str(settings.current_ADAMONT_forcing_mean))
        else:
            midfolder_base = 'SAFRAN\\'
            daily_meteo_data = get_default_SAFRAN_forcings(ref_start, ref_end)
            # We retrieve all the SAFRAN glacier coordinates
            with open(path_smb_function_forcing+'all_glacier_coordinates.txt', 'rb') as coords_f:
                all_glacier_coordinates = np.load(coords_f,  allow_pickle=True)
                
        ### We modify the ice depth in order to compute the effects of the uncertainties ###
        if(thickness_idx == 1):
            thickness_folder_tail = "1.3\\"
            print("\nIce thickness *1.3 simulation \n")
        elif(thickness_idx == 2):
            thickness_folder_tail = "0.7\\"
            print("\nIce thickness *0.7 simulation \n")
        else:
            thickness_folder_tail = "1\\"
            print("\nOriginal Ice thickness simulation \n")
        midfolder = midfolder_base+thickness_folder_tail
        
        # We remove all the previous SMB and topo simulations
        empty_folder(path_smb_simulations+midfolder)
        empty_folder(path_glacier_area+midfolder)
        empty_folder(path_glacier_volume+midfolder)
        empty_folder(path_glacier_zmean+midfolder)
        empty_folder(path_glacier_slope20+midfolder)
        empty_folder(path_glacier_melt_years+midfolder)
        empty_folder(path_glacier_CPDDs+midfolder)
        empty_folder(path_glacier_snowfall+midfolder)
        empty_folder(path_glacier_evolution_ID_rasters+midfolder)
        empty_folder(path_glacier_evolution_DEM_rasters+midfolder)
        
        # We calculate the year range once we know if the ADAMONT forcings end in 2098 or 2099
        year_range = range(year_start, year_end+1)
        
        glacier_counter = 1
        glaciers_with_errors, melted_glaciers = [],[]
        
#        ensemble_SMB_models = []
        
        print("\nStarting simulations for " + str(year_start) + "-" + str(year_end) + " period")
        
        ####  ITERATING ALL THE GLACIERS  ####
        idx = 0
        for glacier in layer_glaciers:
            print("\nGlacier #" + str(glacier_counter))
            glacier_counter = glacier_counter+1
            glacierName = strip_accents(glacier.GetField("Glacier"))
            massif = glacier.GetField("Massif")
            massif = massif.encode('utf-8')
            glacierArea = glacier.GetField("Area")
            glacierID = glacier.GetField("ID")
            glimsID = glacier.GetField("GLIMS_ID")
            massif_idx = glacier.GetField('massif_SAF')
            aspect = glacier.GetField('aspect_sec')
            
            lat = glacier.GetField('y_coord')
            lon = glacier.GetField('x_coord')
            
            found_glacier = True
            
            # Use CV SMB models for the 32 French alpine glaciers dataset
            # /!\  Set to False for regional glacier evolution simulations
            smb_cv = False
            
            print("Glacier: " + str(glacierName))
            
            ####   REMOVE ONCE ALL THE ADAMONT FORCINGS ARE AVAILABLE   ####           
            if(found_glacier and glacierID != 0 and glacier_counter > counter_threshold):
#            if(True):
                glacier_delta_h = np.any(glimsID == delta_h_processed_glaciers)
                glacier_length = glacier.GetField("Length")
                print("GLIMS ID: " + str(glimsID))
                # We process only the non-discarded glaciers with a delta h function and those greater than 0.5 km2
#                if(True):
#                print('glacierID: ' + str(glacierID))
#                print("glacierArea: " + str(glacierArea))
                if(glacierID == 3651 and glacier_counter == 35): # Tré la Tête
#                if(glacierName == "d'Argentiere"):
#                if(glacierName == "d'Argentiere" or glacierName == "Mer de Glace"):
#                if(np.any(glimsID.encode('ascii') == glims_rabatel['GLIMS_ID']) and (glacierName[-1] != '2' and glacierName[-1] != '3' and glacierName[-1] != '4')):
#                if(massif == "Ubaye"):
                    print ("\n-------   Processing glacier: " + glacierName + " -------")
                    print("GLIMS ID: " + str(glimsID))
                    
                    # We crop the initial rasters to the extent of the GLIMS 2003 or 2015 database
                    path_outline_current_glacier = path_glacier_2003_shapefiles + 'individual_GLIMS_2003\\' +  'GLIMS_ID_' + str(glimsID) + '.shp'
                    current_glacier_ice_depth, current_glacier_DEM, path_glacier_DEM_2003, path_glacier_ID_2003 = crop_inital_rasters_to_GLIMS(path_glacier_ID_rasters, path_glacier_DEM_rasters, path_outline_current_glacier, 
                                                                                             glacier, glacierID, midfolder_base, year_start)
                    
                    # We fetch every time the flowlines shapefile to avoid strange issues
                    shapefile_glacier_flowlines = ogr.Open(path_glacier_flowlines_shapefile)
                    layer_flowlines = shapefile_glacier_flowlines.GetLayer()
                    
                    # We get the processed delta h function for each glacier or a linear one for small glaciers
                    if(glacier_delta_h):
                        delta_h_DEM_current_glacier = genfromtxt(path_delta_h_param + glimsID + '_DEM.csv', delimiter=';')
                        delta_h_dh_current_glacier = genfromtxt(path_delta_h_param + glimsID + '_dh.csv', delimiter=';')
                        
                        # Uncomment to add uncertainty assessement +-10%
#                        delta_h_dh_current_glacier = delta_h_dh_current_glacier*1.1
#                        delta_h_dh_current_glacier = delta_h_dh_current_glacier*0.9
                    else:
                        delta_h_DEM_current_glacier = np.ones(50)
                        delta_h_dh_current_glacier = np.ones(50)
                        
                    # We get the right flow line information
                    flowline = get_flowline(glimsID, glacier_length, layer_flowlines)
                    
                    # We get the glacier aspect in degrees
                    aspect = get_aspect_deg(aspect)
                    
                    # We clip the ice thickness raster of F19 with the glacier's outline
                    # Base raster
                    if os.path.exists(current_glacier_ice_depth):
                        raster_current_F19 = gdal.Open(current_glacier_ice_depth) 
                        ice_depth_current_glacier = raster_current_F19.ReadAsArray()
                        if(np.all(ice_depth_current_glacier == 0)):
                             print("/!\ Ice depth raster coordinates not aligned with GLIMS database for Glacier " + str(glacierName) + " with area = " + str(glacierArea) + " km2\n")
                             glaciers_with_errors.append(glacierName)
                             continue
#                        raster_F19_2003 = gdal.Open(path_glacier_ID_2003) 
#                        ice_depth_glacier_2003 = raster_F19_2003.ReadAsArray()
                        
                          ### We modify the ice depth in order to compute the effects of the uncertainties ###
                        if(thickness_idx == 1):
                            thick_comp = 1.3
                            ice_depth_current_glacier = ice_depth_current_glacier*thick_comp
                        elif(thickness_idx == 2):
                            thick_comp = 0.7
                            ice_depth_current_glacier = ice_depth_current_glacier*thick_comp
                        else:
                            thick_comp = 1
                            
                        # Filter noise 
                        ice_depth_current_glacier = np.where(ice_depth_current_glacier > 550*thick_comp, 0, ice_depth_current_glacier)
                        
                    else:
                        print("\n/!\ Ice depth raster doesn't exist for Glacier " + str(glacierName) + " with area = " + str(glacierArea) + " km2\n")
                        glaciers_with_errors.append(glacierName)
                        continue
                    
                    # We get the current raster information as global variables
                    r_projection, r_pixelwidth, r_pixelheight, r_origin = getRasterInfo(raster_current_F19)
                    
                    if os.path.exists(current_glacier_DEM):
                        raster_current_DEM = gdal.Open(current_glacier_DEM) 
                        DEM_current_glacier = np.round(raster_current_DEM.ReadAsArray())
                        
#                        raster_DEM_2003 = gdal.Open(path_glacier_DEM_2003) 
#                        DEM_glacier_2003 = np.round(raster_DEM_2003.ReadAsArray())
                        
                    else:
                        print("/!\ DEM raster doesn't exist for Glacier " + str(glacierName) + " with area = " + str(glacierArea) + " km2")
                        break
                    
                    # We get the flattened versions of the ID and the DEM of the current glacier
                    masked_ID_current_glacier = np.ma.masked_values(np.float64(ice_depth_current_glacier), ice_depth_current_glacier[0,0])
                    masked_DEM_current_glacier = np.ma.array(DEM_current_glacier, mask = masked_ID_current_glacier.mask)
                    flat_DEM_current_glacier = masked_DEM_current_glacier.compressed()
                    
                    # We sort the DEM by altitude in order to have the altitudinal range ready for iteration
                    DEM_sorted_current_glacier = np.sort(flat_DEM_current_glacier, axis=None)
                    DEM_sorted_current_glacier = np.unique(DEM_sorted_current_glacier[DEM_sorted_current_glacier > 0])
                    
                    if(forcing == 'ADAMONT'):
                        SAFRAN_idx = get_ADAMONT_idx(massif_idx, masked_DEM_current_glacier.compressed().mean(), massif_number, daily_meteo_data['zs'])
                    else:
                        SAFRAN_idx = all_glacier_coordinates[np.where(all_glacier_coordinates[:,3] == glimsID)[0]][0][1]
#                        print("SAFRAN_idx: " + str(SAFRAN_idx))
                        
                    # We get the glacier's reference meteorological values ( [()] in order to access the dictionaries)
                    CPDD_ref, w_snow_ref, s_snow_ref, mon_temp_ref, mon_snow_ref = get_meteo_references(season_meteo, monthly_meteo, glimsID, glacierName)
                    meteo_anomalies = {'CPDD': CPDD_ref, 'w_snow': w_snow_ref, 's_snow': s_snow_ref, 'mon_temp': mon_temp_ref, 'mon_snow': mon_snow_ref}
                    
                    # We compute the glacier retreat, updating the DEM and ID matrixes and storing the rasters for every year
                    
                    if(not np.all(np.isnan(masked_ID_current_glacier.compressed()))):
                        masked_DEM_current_glacier_u, masked_ID_current_glacier_u = glacier_evolution(masked_DEM_current_glacier, 
                                                                                                masked_ID_current_glacier, 
                                                                                                delta_h_dh_current_glacier,
                                                                                                delta_h_DEM_current_glacier, 
                                                                                                DEM_sorted_current_glacier, 
                                                                                                daily_meteo_data, meteo_anomalies,
                                                                                                flowline, raster_current_DEM, current_glacier_DEM,
                                                                                                store_plots, glacierName, 
                                                                                                glacierID, glimsID, massif, lat, lon, aspect,
                                                                                                midfolder, pixel_area, glaciers_with_errors, glims_rabatel,
                                                                                                lasso_scaler, lasso_model, ensemble_SMB_models, 
                                                                                                year_range, ref_start, ref_end, SAFRAN_idx, overwrite) 
                    else:
                        glacier_melted_flag = True
                        glacier_melt_year = year_start
                    
                    if(glacier_melted_flag):
                        melted_glaciers.append([glimsID, glacierName, glacier_melt_year])
                        print("\n Glacier completely melted")
                else:
                    print("\n /!\  Glacier not present in delta h dataset  ")
            else:
#                print("found_glacier: " + str(found_glacier))
#                print("glacierID: " + str(glacierID))
                glaciers_with_errors.append(glacierName)
                if(not found_glacier):
                    print("\n/!\  Glacier not found in forcings with GLIMS ID: " + str(glimsID) + "  /!\\\n")
                elif(glacierID == 0):
                    print("\n/!\  Glacier with Glacier ID = 0. No associated raster data  /!\\\n")
            idx = idx+1
            
        ### End of glaciers loop  ###
        
        glaciers_with_errors = np.asarray(glaciers_with_errors)
        melted_glaciers = np.asarray(melted_glaciers)
        
        path_glacier_w_errors_current_combination = path_glacier_w_errors + midfolder
        path_melt_years_current_combination = path_glacier_melt_years + midfolder
        
        if not os.path.exists(path_glacier_w_errors_current_combination):
            os.makedirs(path_glacier_w_errors_current_combination)
        if not os.path.exists(path_melt_years_current_combination):
            os.makedirs(path_melt_years_current_combination)
            
        try:
            if(glaciers_with_errors.size > 0):
                np.savetxt(path_glacier_w_errors_current_combination + "glaciers_w_errors_" + str(year_start)+ "_" + str(year_end) + '.csv', glaciers_with_errors, delimiter=";", fmt="%s")
            if(melted_glaciers.size > 0):
                np.savetxt(path_melt_years_current_combination + "melted_glaciers_" + str(year_start)+ "_" + str(year_end) + '.csv', melted_glaciers, delimiter=";", fmt="%s")
        except IOError:
            print("File currently opened. Please close it to proceed.")
            os.system('pause')
            # We try again
            try:
                if(glaciers_with_errors.size > 0):
                    np.savetxt(path_glacier_evolution + midfolder + "glaciers_w_errors_" + str(year_start)+ "_" + str(year_end) + '.csv', glaciers_with_errors, delimiter=";", fmt="%s")
                if(melted_glaciers.size > 0):
                    np.savetxt(path_glacier_evolution + midfolder + "melted_glaciers_" + str(year_start)+ "_" + str(year_end) + '.csv', melted_glaciers, delimiter=";", fmt="%s")
            except IOError:
                print("File still not available. Aborting simulations.")
        
    else:
        print("\nSkipping...")
        
###   End of main function  ###          
                    
