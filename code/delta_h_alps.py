# -*- coding: utf-8 -*-

"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

DELTA H PARAMETERIZATION OF FRENCH ALPINE GLACIERS

"""

## Dependencies: ##
import os
from numba import jit
from osgeo import gdal,ogr
import subprocess
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
#from scipy.optimize import curve_fit
import numpy as np
import numpy.polynomial.polynomial as poly
from numpy import genfromtxt
import pandas as pd
import unicodedata
import shutil
#import matplotlib.font_manager as font_manager
from pathlib import Path


######   FILE PATHS    #######
    
# Folders     
#workspace = 'C:\\Jordi\\PhD\\Python\\'
workspace = str(Path(os.getcwd()).parent) + '\\'
path_delta_h_param = workspace + "glacier_data\\delta_h_param\\"
path_glacier_shapefiles = workspace + 'glacier_data\\glacier_shapefiles\\2015\\'
path_glacier_rasters = workspace + 'glacier_data\\glacier_rasters\\'
path_glacier_coordinates = workspace + 'glacier_data\\glacier_coordinates\\'     
path_glacier_rasters_aligned = path_glacier_rasters + 'overall_rasters\\'   

# Rasters        
path_thickness_glaciers = path_glacier_rasters_aligned + 'DEM_2011-1979_05_CRS_aligned' + '.tif' 
path_dem_average = path_glacier_rasters_aligned + 'DEM_average_CRS_aligned' + '.tif'  

# Shapefiles
path_shapefile_glaciers = path_glacier_shapefiles + 'glaciers_2015_05' + '.shp' 


######     DECORATORS     ######
 
# Remove all accents from string
def strip_accents(s):
#   s = s.decode('latin-1').encode('utf-8')
#   s = unicode(s, "utf-8")
   return (''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')).strip()

# Clips a raster with the shape of a polygon   
def clipRaster_with_polygon(output_cropped_raster, input_raster, shapefile_mask):
    if not os.path.exists(output_cropped_raster):
        try:
            subprocess.check_output("gdalwarp --config GDALWARP_IGNORE_BAD_CUTLINE YES -q -cutline \"" + shapefile_mask 
                                               + "\" -crop_to_cutline -tr 0.000550176797434 0.000412511541874 -of GTiff \"" 
                                               + input_raster + "\" \"" + output_cropped_raster +"\"",
                                         stderr=subprocess.PIPE,
                                         shell=True)
            print("Generating new ice thickness raster... ")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        
def normalize_dh(dh, dh_ref):
    dh_n = dh.copy()
    dh_n = abs(dh_n - dh_ref.max())/abs(dh_ref.min() - dh_ref.max())
    dh_n = np.where(dh_n > 1, 1, dh_n)
    return dh_n

def normalize_dem(dem, dem_ref):
    dem_n = dem.copy()
    dem_n = (dem_ref.max() - dem)/(dem_ref.max() - dem_ref.min())
    return dem_n

def cloud_cover(dh):
    return (dh > 0).sum() > 2*(dh.size/3)

def get_wdw_size(array):
    if(array.size < 70):
       wdw = array.size/20
    elif(array.size < 40):
        wdw = array.size/10
    else:
        wdw = array.size/35
    return int(wdw)

# First input raw filter
def dh_BaseFilter(thickness, altitude):
    return ((thickness > 10) or (thickness < -130) or ((altitude > 2900) and (thickness < -40)) or ((altitude > 3400) and (thickness < -20)) or ((altitude < 2500) and (thickness > -10)) or ((altitude < 2000) and (thickness > -60)))

@jit
def dh_FlatFilter(dh, dem):
    wdw = 8*get_wdw_size(dh)
    i = 0
    for dhfi, dhti in zip(dh[0:wdw], dh[dh.size-wdw:]):
        if(((np.average(dh[dh.size-wdw:dh.size]) - dhfi) < 5) or (dhfi > -10)):
            dh = np.delete(dh, i)
            dem = np.delete(dem, i)
        if((abs(np.average(dh[0:wdw]) - dhti) < 15) or (dhti < -25)):
            dh = np.delete(dh, dh.size-wdw+i)
            dem = np.delete(dem, dh.size-wdw+i)
        else:
            i=i+1
    return dh, dem

@jit
def apply_BaseFilter(dem, deltah, multimask):
    i = 0
    for altitude, thickness in zip(dem.flatten(), deltah.flatten()):
        if((multimask[i] == False) and dh_BaseFilter(thickness, altitude)):
            multimask[i] = True
        i=i+1
    return multimask

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    diff = np.amin(np.abs(array-value))
    return idx, diff

@jit
def generate_envelopes(dh, dem_dh, poly_fit, dem_poly):
     max_cloud = []
     min_cloud = []
     # We need to project the input raster values of dh to the polygonal fit's DEM axis ones
     for dh_alt, dh_i in zip(dem_dh, dh):
         nearest_idx, diff = find_nearest(dem_poly, dh_alt)
         poly_nearest = poly_fit[nearest_idx]
         if(dh_i >= poly_nearest):
             max_cloud.append(dh_i)
             min_cloud.append(float("NaN"))
         else:
             min_cloud.append(dh_i)
             max_cloud.append(float("NaN"))
         
     envelope_max = interpolate_envelope(np.asarray(max_cloud))
     envelope_min = interpolate_envelope(np.asarray(min_cloud))
     return envelope_max, envelope_min   
    
def interpolate_envelope(envelope):
    '''
    interpolate to fill NaN values
    '''
    good = np.where(np.isfinite(envelope))
    bad = np.where(np.isnan(envelope))
    interpolated = np.interp(bad[0], good[0], envelope[good])
    envelope[bad] = interpolated
    return envelope

def dezerofy(array, it):
    if(array[it] == 0 and array[it+1] != 0):
        return array[it+1]
    else:
        return array[it]


# Fills the front and tail of the convolved envelope in order to replace the artifacts
def fill_convolved_FrontTail(envelope_max, envelope_min, smoothed_max_envelope_1, smoothed_min_envelope_1, dh, poly_fit, window, n_ave):
    if(np.average(poly_fit[(poly_fit.size - int(window.size/2)):poly_fit.size]) < -10):
            k = 0.3
    else:
            k = 1.5
    if(np.count_nonzero(dh[(dh.size-n_ave):dh.size]) < (2*n_ave)/3):
        max_tail = np.zeros(int(window.size/2))
    else:
        max_tail = abs(k*poly_fit[(poly_fit.size - int(window.size/2)):poly_fit.size]) + abs(3.0*dezerofy(envelope_max, smoothed_max_envelope_1.size-1))
        
    max_front = poly_fit[0:(int(window.size/2)-1)] + 0.0*envelope_max[0]
    
    min_tail = 0.5*poly_fit[(poly_fit.size - int(window.size/2)):poly_fit.size] + k*dezerofy(envelope_min, smoothed_max_envelope_1.size-1)
    min_front = poly_fit[0:(int((window.size -1)/2))] + 0.6*envelope_min[0]
    
    smoothed_max_envelope_1 = np.append(smoothed_max_envelope_1, max_tail)
    smoothed_max_envelope_1 = np.insert(smoothed_max_envelope_1, 0, max_front)
    smoothed_min_envelope_1 = np.append(smoothed_min_envelope_1, min_tail)
    smoothed_min_envelope_1 = np.insert(smoothed_min_envelope_1, 0, min_front)
    
    return smoothed_max_envelope_1, smoothed_min_envelope_1


def smooth_envelope(envelope_max, envelope_min, poly_fit, dh):
    n_ave = get_wdw_size(envelope_max)
    if(n_ave%2 != 0):
        n_ave = n_ave+1
    window = np.ones(n_ave)/n_ave 
    # We force the window size to be of even size
    
    smoothed_max_envelope_1 = np.convolve(envelope_max, window, mode='valid')
    smoothed_min_envelope_1 = np.convolve(envelope_min, window, mode='valid')
    
    # We fill the missing values at the beginning and the end of the array with a trend
    smoothed_max_envelope_1, smoothed_min_envelope_1 = fill_convolved_FrontTail(envelope_max, 
                                                                                envelope_min, 
                                                                                smoothed_max_envelope_1, 
                                                                                smoothed_min_envelope_1,
                                                                                dh,
                                                                                poly_fit,
                                                                                window,
                                                                                n_ave)
    return smoothed_max_envelope_1, smoothed_min_envelope_1

def needs_zero_topping(dh, wdw):
        return (dh[(dh.size-wdw):dh.size] > -2).sum() > wdw/3


def stat_topping(dh, poly_fit, wdw):
    # We recalculate the processing window
    if(needs_zero_topping(dh, wdw)):
            poly_fit[(poly_fit.size-wdw):poly_fit.size] = np.zeros(wdw)
    zero_idx = np.where(poly_fit >= -0.5)[0]
    if(zero_idx.size > 1):
        poly_fit[zero_idx[0]:poly_fit.size] = np.zeros(poly_fit.size-zero_idx[0])
    # We prevent the fit to descend again at high altitudes (tail)
    poly_fit[np.argmax(poly_fit):poly_fit.size] = np.where(poly_fit[np.argmax(poly_fit):poly_fit.size] < poly_fit.max(), 
            poly_fit.max(), 
            poly_fit[np.argmax(poly_fit):poly_fit.size])
    # And now we prevent the fit to go up again at low altitudes (front)
    poly_fit[0:np.argmin(poly_fit)] = np.where(poly_fit[0:np.argmin(poly_fit)] > poly_fit.min(), 
            poly_fit.min(), 
            poly_fit[0:np.argmin(poly_fit)])
    
    return poly_fit

 # Determines the specific weights for the polynomial fit
def set_weights(dh, dem, wdw, extraw):
    topping_threshold = -0.5
    weights = np.ones(dem.size)
    
    idx,diff = find_nearest(dh[0:wdw], -140)
    weights[idx] = extraw
    if(needs_zero_topping(dh, wdw)):
        zero_idx = np.where(dh >= topping_threshold)[0]
        weights[zero_idx[zero_idx > (2*dh.size)/3]] = extraw 
    else:
        idx,diff = find_nearest(dh[dh.size-wdw:dh.size], 0)
       
        weights[idx+dh.size-wdw] = extraw
    
    return weights

# Returns the indexes of the values within the envelopes 
def dh_EnvelopeFitFilterIdxs(current_glacier, envelope_max, envelope_min):
    filt_idx = np.where((current_glacier <= envelope_max) & (current_glacier >= envelope_min))
    return filt_idx

def compute_envelopes(dem, dh):
    # We calculate the polynomial fit in order to set a trend for the scattered points
    if(dh.size < 60):
       wdw = int(dh.size/10)
    elif(dh.size < 40):
        wdw = int(dh.size/5)
    else:
        wdw = int(dh.size/25)
    weights = set_weights(dh, dem, wdw, 2)
    p1 = poly.Polynomial.fit(dem, dh, 2, w=weights)
    
    # We generate an envelope for the polyonmial fit based on the interpolation of max and min values
    dem_poly_fit_1 = np.asarray(*p1.linspace(n=dem.size)[:1]).flatten()
    poly_fit_1 = np.asarray(*p1.linspace(n=dh.size)[1:]).flatten()
    envelope_max, envelope_min = generate_envelopes(dh, 
                                                    dem, 
                                                    poly_fit_1,
                                                    dem_poly_fit_1)
    
    
    
    # Let's smooth the fit envelope. Set convolution window n_ave based on the glacier array's length
    smoothed_max_envelope, smoothed_min_envelope = smooth_envelope(envelope_max, 
                                                                       envelope_min, 
                                                                       poly_fit_1,
                                                                       dh)
    
    return smoothed_max_envelope, smoothed_min_envelope, poly_fit_1, dem_poly_fit_1    


# Determines the polynomial order for the fit
def get_poly_order(dh_filtered):
    if(dh_filtered.size > 900):
        deg_pol = 6
        ending = "th"
    elif(dh_filtered.size > 400):
        deg_pol = 4
        ending = "th"
    elif(dh_filtered.size > 150):
        deg_pol = 3
        ending = "rd"
    else:
        deg_pol = 2
        ending = "nd"
        
    return deg_pol, ending


def get_poly_fit(dem, dh):
    deg_pol, ending = get_poly_order(dh)
    if(dh.size < 60):
       wdw = int(dh.size/10)
    elif(dh.size < 40):
        wdw = int(dh.size/5)
    else:
        wdw = int(dh.size/30)
        
    weights = set_weights(dh, dem, wdw, 10)
    
    p2 = poly.Polynomial.fit(dem, dh, deg_pol, w=weights)
    dem_poly_fit = np.asarray(*p2.linspace(n=dem.size)[:1]).flatten()
    polynomial_fit_2 = np.asarray(*p2.linspace(n=dh.size)[1:]).flatten()
    
    # We process the high-altitude values in order reduce the mass loss
    polynomial_fit_2 = stat_topping(dh, polynomial_fit_2, wdw)
    
    return dem_poly_fit, polynomial_fit_2, deg_pol, ending

def perform_stat_filtering(dem, dh, max_envelope, min_envelope):
    # We proceed to the second wave of filtering
    statFilt_idx = dh_EnvelopeFitFilterIdxs(dh, max_envelope, min_envelope)
    dh_statFilt = dh[statFilt_idx]
    dem_statFilt = dem[statFilt_idx]
    
    dem_poly_fit, polynomial_fit_2, deg_pol, ending = get_poly_fit(dem_statFilt, dh_statFilt)
    
    return dem_statFilt, dh_statFilt, polynomial_fit_2, dem_poly_fit, deg_pol, ending

def crop_dh_range(dh, dh_ref):
    cropped_dh_filtered = np.where(dh < dh_ref.min(), dh_ref.min(), dh)
    cropped_dh_filtered = np.where(cropped_dh_filtered > 0, 0, cropped_dh_filtered)
    return cropped_dh_filtered

def export_to_csv(glimsID, dem, dh):
    poly_dem_path = path_delta_h_param + glimsID + "_DEM.csv"
    poly_fit_path = path_delta_h_param + glimsID + "_dh.csv"
    np.savetxt(poly_dem_path, dem, delimiter=";", fmt="%.7f")
    np.savetxt(poly_fit_path, dh, delimiter=";", fmt="%.7f")
    
def copyfile (src, dest):
    shutil.copyfile (src, dest)
    
def main(compute, overwrite):

    ################################################################################
    ###############		                MAIN               	#####################
    ################################################################################                      
    
    # Delta h parametrization implementation
    # Required input files: 
    # - Averaged DEM between the 1979 and 2011 DEMs
    # - Delta h raster layer with the (filtered) DEM 1979-2011 
    # - Individual glacier shapefiles
            
    
    #######    Delta h parametrization for all the selected glaciers   #######
    
    print("\n-----------------------------------------------")
    print("          DELTA H PARAMETERIZATION ")
    print("-----------------------------------------------\n")
    
    if(compute):
        # We open the raster files and shapefiles:
        shapefile_glaciers = ogr.Open(path_shapefile_glaciers)
        layer_glaciers = shapefile_glaciers.GetLayer()
        
        # Let's iterate through each glacier to compute its delta h parametrization
        nfigure = 1
        discarded_glaciers, delta_h_processed_glaciers = [],[]
        glacierLat, glacierLon, glacierNames = [],[],[]
        
        for glacier in layer_glaciers:
            glacierName = strip_accents(glacier.GetField("Glacier"))
            glimsID = glacier.GetField("GLIMS_ID")
    #        massif = glacier.GetField("Massif")
            
            delta_h_function_exists = os.path.exists(path_delta_h_param+glacierName+'_dh.csv')
            
            if(overwrite or (not delta_h_function_exists)):
                print ("\n---   Processing glacier: " + glacierName + " ---")
                
                # Current glacier paths declarations
                current_glacier_shapefile_path = path_glacier_shapefiles + "GLIMS_ID_" + glimsID + ".shp"
                current_glacier_cropped_dh = path_glacier_rasters + "dh_2015\\dh_Glacier_" + glacierName + ".tif"
                current_glacier_cropped_DEM = path_glacier_rasters + "DEM_2015\\DEM_Glacier_" + glacierName + ".tif"
                
                # Let's crop the ice thickness difference and the averaged 1979-2011 DEM
                # Base raster
                clipRaster_with_polygon(current_glacier_cropped_dh, path_thickness_glaciers, current_glacier_shapefile_path)
                
                # Averaged 1979-2011 DEM
                clipRaster_with_polygon(current_glacier_cropped_DEM, path_dem_average, current_glacier_shapefile_path)
                
                # Now we open and read as arrays the newly cropped rasters for the current glacier
                current_glacier_raster_DEM = gdal.Open(current_glacier_cropped_DEM)   
                current_glacier_raster_dh = gdal.Open(current_glacier_cropped_dh)
                current_glacier_array_DEM = current_glacier_raster_DEM.ReadAsArray()
                current_glacier_array_dh = current_glacier_raster_dh.ReadAsArray()
                        
                # We flatten, round-up and mask the NoData values by compressing the masked array
                nodata_dh_current_glacier = current_glacier_raster_dh.GetRasterBand(1).GetNoDataValue()
                nodata_DEM_current_glacier = current_glacier_raster_DEM.GetRasterBand(1).GetNoDataValue()
                masked_dh_currentGlacier = np.ma.masked_values(current_glacier_array_dh, nodata_dh_current_glacier)
                masked_DEM_currentGlacier = np.ma.masked_values(current_glacier_array_DEM, nodata_DEM_current_glacier)
                
                # We perform a quick check to detect if the cloud cover is too important to be filtered
                if(cloud_cover(masked_dh_currentGlacier.compressed())):
                    print("\n Glacier discarded due to cloud cover")
                    discarded_glaciers.append(glimsID)
                    continue
                else:
                    delta_h_processed_glaciers.append(glimsID)
                
                # We store the glacier's coordinates in order to later find the temperature locations
                glacierLat.append(glacier.GetField("y_coord"))
                glacierLon.append(glacier.GetField("x_coord"))
                glacierNames.append(glacierName) 
                
                # Let's compute the hypsometry of the glacier
                hypsometry, bins = np.histogram(masked_DEM_currentGlacier.compressed(), bins="fd")
                
                # We create a new mask merging the DEM and ice thickness ones in order to keep only the pixels which are valid
                # in both rasters
                multimask = ((masked_dh_currentGlacier.mask == True) | (masked_DEM_currentGlacier.mask == True)).flatten()
                DEM_currentGlacier_nodata = np.ma.array(current_glacier_array_DEM, mask = multimask, fill_value = nodata_DEM_current_glacier)
                deltah_currentGlacier_nodata = np.ma.array(current_glacier_array_dh, mask = multimask, fill_value = nodata_dh_current_glacier)
        
                # We update the mask in order to filter the ice thickness difference values which are 
                multimask = apply_BaseFilter(DEM_currentGlacier_nodata, deltah_currentGlacier_nodata, multimask)
                
                # We apply the mask again with the filtered pixels
                DEM_currentGlacier_nodata_flat = np.ma.array(current_glacier_array_DEM, mask = multimask, fill_value = nodata_DEM_current_glacier).compressed()
                deltah_currentGlacier_nodata_flat = np.ma.array(current_glacier_array_dh, mask = multimask, fill_value = nodata_dh_current_glacier).compressed()
                
                # We create a DataFrame with the DEM and delta h information of the glacier and its normalized version
                current_glacier = pd.DataFrame(np.column_stack([deltah_currentGlacier_nodata_flat, DEM_currentGlacier_nodata_flat]), columns=['dh','DEM'])
                
                # Group by DEM (altitude)
                current_glacier_grouped = current_glacier.groupby("DEM")
                
                # Mean dh for each altitude
                current_glacier_mean = current_glacier_grouped.mean()
                
                # Separating values (dh's) and indexes (DEMs)
                dem_filtered_current_glacier = current_glacier_mean.index.get_values()
                dh_filtered_current_glacier = current_glacier_mean.values.flatten()
                
                dh_filtered_current_glacier, dem_filtered_current_glacier = dh_FlatFilter(dh_filtered_current_glacier, 
                                                                                          dem_filtered_current_glacier)
                
                # We calculate again a polynomial fit of a higher order in order to get a more accurate fit
                max_envelope, min_envelope, poly_fit_1, dem_poly_fit_1 = compute_envelopes(dem_filtered_current_glacier, 
                                                                                           dh_filtered_current_glacier)
                
                dem_filtered, dh_filtered, poly_fit_2, dem_poly_fit_2, deg_pol, ending = perform_stat_filtering(dem_filtered_current_glacier, 
                                                                                           dh_filtered_current_glacier, 
                                                                                           max_envelope, 
                                                                                          min_envelope)
                # We normalize all the output variables
                dem_filtered_n = normalize_dem(dem_filtered, dem_poly_fit_2)
                dh_filtered_n = normalize_dh(crop_dh_range(dh_filtered, poly_fit_2), poly_fit_2)
                poly_fit_2_n = normalize_dh(poly_fit_2, poly_fit_2)
                dem_poly_fit_2_n = normalize_dem(dem_poly_fit_2, dem_poly_fit_2)
                
                filt_ratio = 100.0*(float(dh_filtered.size)/float(masked_dh_currentGlacier.compressed().size))
                print("Raster pixels used: " + "%.2f" % filt_ratio + "%")
                
                
                
                #####   SAVE DELTA H PARAMETERIZATION IN CSV FILES    ######
                export_to_csv(glimsID, dem_poly_fit_2_n, poly_fit_2_n)
                
                
                #####        PLOTS        #####
                
                # Plot the polynomial interpolation in meters
                plt.figure(nfigure, figsize=(10, 20))
                plt.subplot(211)
        
                plt.ylabel('Normalized thickness change', fontsize=24)
                plt.xlabel('Normalized altitude', fontsize=24)
                plt.title("Glacier: " + glacierName, fontsize=28)
                plt.tick_params(labelsize=20)
                
                plt.plot(dem_filtered_n, dh_filtered_n, 'o', color='skyblue', markersize=3, label='Filtered DEM raster values')
                plt.plot(dem_poly_fit_2_n, poly_fit_2_n, label=(str(deg_pol) + str(ending) + ' order polynomial fit'), linewidth=4)
                
                plt.legend(fontsize='xx-large')
                plt.gca().invert_yaxis()
                
                plt.subplot(212)
        
                plt.ylabel('Surface area')
                plt.xlabel('Altitude (m)')
                plt.title("Glacier: " + glacierName)
                
                plt.bar(bins[:-1] + np.diff(bins) / 2, hypsometry, np.diff(bins), label='Hypsometry')
                
                plt.legend()
                plt.gca().invert_xaxis()
                nfigure+=1
                
                ## End for loop ##
                
        
        print("\nDiscarded glaciers due to cloud cover: ")
        for glacier in discarded_glaciers:
            print("\n - " + glacier)
        
        # We store the discarded glaciers in a CSV file
        np.savetxt(path_delta_h_param+"discarded_glaciers.csv", discarded_glaciers, delimiter=";", fmt="%40s")
        
        print("\nProcessing glaciers with cloud cover...")
        # We now process the discarded glaciers with standard delta h function
        for glacier in discarded_glaciers:
            # We use Glacier Ouest des Sellettes delta h function as a reference 
            copyfile(path_delta_h_param + 'G006204E44866N' + '_DEM.csv', path_delta_h_param + str(glacier) + '_DEM.csv')
            copyfile(path_delta_h_param + 'G006204E44866N' + '_dh.csv', path_delta_h_param + str(glacier) + '_dh.csv')

            delta_h_processed_glaciers.append(glimsID)
        
        np.savetxt(path_delta_h_param+"delta_h_processed_glaciers.csv", delta_h_processed_glaciers, delimiter=";", fmt="%50s")
        
            
        #####   SAVE GLACIER COORDINATES IN A CSV FILE    ######
        
        glacierCoordinates = pd.DataFrame(np.column_stack([glacierNames, glacierLon, glacierLat]), columns=["Name", "Lon", "Lat"])
        glacierCoordinates.to_csv(path_glacier_coordinates + "glacierCoordinates.csv", sep=";", float_format="%f2.10", index=False)
        
        if(len(delta_h_processed_glaciers) > 0):
            print("\nGathering all plots in a pdf file... ")
            try:
                pdf_plots = workspace + "output_plots.pdf"
                pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_plots)
                for fig in range(1, nfigure): 
                    pdf.savefig( fig )
                pdf.close()
                plt.close('all')
                subprocess.Popen(pdf_plots, shell=True)
                
            except IOError:
                print("Could not open pdf. File already opened. Please close the pdf file.")
                os.system('pause')
                # We try again
                try:
                    subprocess.Popen(pdf_plots, shell=True)
                except IOError:
                    print("File still not available")
                    pass
    else:
        print("\nSkipping...")
            
###   End of main function   ###
