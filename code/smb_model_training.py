# -*- coding: utf-8 -*-

"""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

GLACIER SMB MACHINE LEARNING MODEL(S) TRAINING

"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import numpy.polynomial.polynomial as poly
#from numba import jit
import math
import time
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from keras import backend as K
from keras.models import load_model
import statsmodels.api as sm
import itertools
#import pandas as pd
import pickle
import copy
import os
import settings
from scipy.stats import gaussian_kde
from warnings import filterwarnings
from pathlib import Path

filterwarnings('ignore')

######   FILE PATHS    #######
    
# Folders     
workspace_path = Path(os.getcwd()).parent 
workspace = str(Path(os.getcwd()).parent) + '\\'
root = str(workspace_path.parent) + '\\'
path_smb = workspace + 'glacier_data\\smb\\'
path_smb_function = path_smb + 'smb_function\\'
path_smb_function_SPAZM = path_smb_function + 'SPAZM\\'
path_smb_function_ADAMONT = path_smb_function + 'ADAMONT\\'
path_smb_simulations = path_smb + 'smb_simulations\\'
path_glims = workspace + 'glacier_data\\GLIMS\\' 
path_glacier_coordinates = workspace + 'glacier_data\\glacier_coordinates\\' 

####    FUNCTIONS     #####

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def interpolate_extended_glims_variable(variable_name, glims_rabatel, glims_2003, glims_1985, glims_1967):
    # In case there are multiple results, we choose the one with the most similar area
    
    interpolated_var = []
    for glims_glacier in glims_rabatel:
        idx_2003 = np.where(glims_2003['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
        if(idx_2003.size > 1):
            idx_aux = find_nearest(glims_2003[idx_2003]['Area'], glims_glacier['Area'])
            idx_2003 = idx_2003[idx_aux]
        idx_1985 = np.where(glims_1985['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
        if(idx_1985.size > 1):
            idx_aux = find_nearest(glims_1985[idx_2003]['Area'], glims_glacier['Area'])
            idx_1985 = idx_1985[idx_aux]
        
        idx_1959 = np.where(glims_1967['GLIMS_ID'] == glims_glacier['GLIMS_ID'])[0]
        if(idx_1959.size > 1):
            idx_aux = find_nearest(glims_1967[idx_2003]['Area'], glims_glacier['Area'])
            idx_1959 = idx_1959[idx_aux]
        
        var_1959 = glims_1967[idx_1959][variable_name]
        var_2015 = glims_glacier[variable_name]
        var_2003 = glims_2003[idx_2003][variable_name]
        var_1985 = glims_1985[idx_1985][variable_name]
        
        if(not math.isnan(var_2015)):
            interp_1959_1984 = np.linspace(var_1959, var_1985, num=(1984-1959))
            interp_1984_2003 = np.linspace(var_1985, var_2003, num=(2003-1984))
            interp_2003_2015 = np.linspace(var_2003, var_2015, num=(2015-2003)+1)
            interp_1959_2003 = np.append(interp_1959_1984, interp_1984_2003)
            interp_1959_2015 = np.append(interp_1959_2003, interp_2003_2015)
        else:
            interp_1959_2015 = [var_1985]
        interpolated_var.append(interp_1959_2015)
            
    return np.asarray(interpolated_var)

def all_combinations(any_list):
    return itertools.chain.from_iterable(
        itertools.combinations(any_list, i + 1)
        for i in range(2, 5))
    

# Creates the spatiotemporal matrix to train the SMB model with climatic data at each glacier's centroid
def create_spatiotemporal_matrix(SMB_raw, season_raw_meteo_anomalies_SMB, mon_temp_anomalies, mon_snow_anomalies, glims_rabatel, glacier_mean_altitude, glacier_area, spatiotemporal_flag, first_year):
    print("Creating spatiotemporal matrix...")
    x_reg, y_reg = [],[]
    max_altitudes = glims_rabatel['MAX_Pixel']
    slopes20 = glims_rabatel['slope20_evo']
#    slopes20 = glims_rabatel['slope20']
    lons = glims_rabatel['x_coord']
    lats = glims_rabatel['y_coord']
    
    aspects = np.cos(glims_rabatel['Aspect_num'])
    
    best_models = genfromtxt(path_smb + 'chosen_models_3_5.csv', delimiter=';', skip_header=1, dtype=None) 
    best_models = best_models[:19]
    
#    print("best_models['f1']: " + str(best_models['f1']))
    
    x_reg_full_array, x_reg_array, y_reg_array = [],[],[]
    
    #### CODE FOR STEP-WISE CALCULATION  ###
#    x_reg_combinations_idxs = all_combinations(range(0, 31))
#    count = 0
#    i = 0
#    
#    combinations_df = pd.DataFrame([], columns = ['combination_idxs', 'r2_adj', 'VIF', 'p-value'])
#    for x_reg_idxs in x_reg_combinations_idxs:
#        x_reg_idxs = list(x_reg_idxs)
#        print("Current idx combination: " + str(x_reg_idxs))
    ##########################################
    
    ###  NORMAL FLOW    #######
    
    for model in best_models:
        x_reg_idxs = eval(model['f1'])
        x_reg_idxs = list(x_reg_idxs)
        
#        print(x_reg_idxs)
        
    #################################
    
        count_glacier = 0
        x_reg_full, x_reg_nn, x_reg, y_reg = [],[],[],[]
        
        for SMB_glacier, CPDD_glacier, winter_snow_glacier, summer_snow_glacier, mon_temp_glacier, mon_snow_glacier, mean_alt, slope20, max_alt, area, lon, lat, aspect in zip(SMB_raw, season_raw_meteo_anomalies_SMB['CPDD'], season_raw_meteo_anomalies_SMB['winter_snow'], season_raw_meteo_anomalies_SMB['summer_snow'], mon_temp_anomalies, mon_snow_anomalies, glacier_mean_altitude, slopes20, max_altitudes, glacier_area, lons, lats, aspects):
            year = 1959 
            
            for SMB_y, cpdd_y, w_snow_y, s_snow_y,  mon_temp_y, mon_snow_y, mean_alt_y, area_y in zip(SMB_glacier, CPDD_glacier[first_year:], winter_snow_glacier[first_year:], summer_snow_glacier[first_year:], mon_temp_glacier[first_year:], mon_snow_glacier[first_year:], mean_alt, area):
                
                mon_temp_y = mon_temp_y[:12]
                mon_snow_y = mon_snow_y[:12]
                
                # We get the current iteration combination
                input_variables_array = np.array([cpdd_y, w_snow_y, s_snow_y, mean_alt_y, max_alt, slope20, area_y, lon, lat, aspect, mean_alt_y*cpdd_y, slope20*cpdd_y, max_alt*cpdd_y, area_y*cpdd_y, lat*cpdd_y, lon*cpdd_y, aspect*cpdd_y, mean_alt_y*w_snow_y, slope20*w_snow_y, max_alt*w_snow_y, area_y*w_snow_y, lat*w_snow_y, lon*w_snow_y, aspect*w_snow_y, mean_alt_y*s_snow_y, slope20*s_snow_y, max_alt*s_snow_y, area_y*s_snow_y, lat*s_snow_y, lon*s_snow_y, aspect*s_snow_y])
                if(spatiotemporal_flag == 'spatial'): 
                    input_full_array = np.append(input_variables_array, mon_temp_y)
                    input_full_array = np.append(input_full_array, mon_snow_y)
                    x_reg_full.append(copy.deepcopy(input_full_array.tolist()))
                elif(spatiotemporal_flag == 'temporal'):
                    x_reg_full.append(copy.deepcopy(input_variables_array.tolist()))
                x_reg.append(copy.deepcopy(input_variables_array[x_reg_idxs].tolist()))
                
                # We create a separate smaller matrix for the Neural Network algorithm
                input_features_nn_array = np.array([cpdd_y, w_snow_y, s_snow_y, mean_alt_y, max_alt, slope20, area_y, lon, lat, aspect])
                input_features_nn_array = np.append(input_features_nn_array, mon_temp_y)
                input_features_nn_array = np.append(input_features_nn_array, mon_snow_y)
                
                x_reg_nn.append(input_features_nn_array.tolist())
                
                year = year+1
                
                y_reg.append(SMB_y)
                
            count_glacier = count_glacier+1
            
        x_reg_full = np.asarray(x_reg_full)
        x_reg_nn = np.asarray(x_reg_nn)
        x_reg = np.asarray(x_reg)
        y_reg = np.asarray(y_reg) 
        
        with open(root + 'X_nn_extended.txt', 'wb') as x_f:
            np.save(x_f, x_reg_nn)
        with open(root + 'y_extended.txt', 'wb') as y_f:
            np.save(y_f, SMB_raw)
        
        finite_idxs = np.isfinite(y_reg)
        x_reg_full = x_reg_full[finite_idxs,:]
        x_reg_nn = x_reg_nn[finite_idxs,:]
        x_reg = x_reg[finite_idxs,:]
        y_reg = y_reg[finite_idxs]
        
        #######  CODE FOR STEP-WISE CALCULATION OF BEST PERFORMING VARIABLE COMBINATIONS FOR THE SMB MODEL  ##########
        
#        vif_sm = [variance_inflation_factor(x_reg, i) for i in range(x_reg.shape[1])]
#        
#        model = sm.OLS(y_reg, sm.add_constant(x_reg, prepend=False))
#        model_fit = model.fit()
#        
#        ## We fill the results dataframe
#        combinations_df.loc[-1] = [x_reg_idxs, model_fit._results.rsquared_adj, vif_sm, model_fit._results.f_pvalue]
#        combinations_df.index = combinations_df.index +1
#        
#        count = count +1
#        if(count%150 == 0):
#            print("Autosaving...")
#            # We store the results in a csv
#            try:
#                combinations_df.to_csv(workspace + 'model_combinations.csv', sep=';')
#            except IOError:
#                print("Could not save csv file. File already opened. Please close the file.")
#                os.system('pause')
#                # We try again
#                try:
#                    combinations_df.to_csv(workspace + 'model_combinations.csv', sep=';')
#                except IOError:
#                    print("File still not available")
#                    pass
        ###################################################
        
        x_reg_full_array.append(copy.deepcopy(x_reg_full))
        x_reg_array.append(copy.deepcopy(x_reg))
        y_reg_array.append(copy.deepcopy(y_reg))
        
    

    return x_reg_full_array, x_reg_nn, x_reg_array, y_reg_array

def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(alphas_, criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(alpha_, color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('alpha')
    plt.ylabel('criterion')

def lasso_CV_model_selection(X, y, fig_idx, year_groups, plot):
    
    # #############################################################################
    # LassoLarsIC: least angle regression with BIC/AIC criterion
    
    model_bic = LassoLarsIC(criterion='bic')
    t1 = time.time()
    model_bic.fit(X, y)
    t_bic = time.time() - t1
#    alpha_bic_ = model_bic.alpha_
    
    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(X, y)
#    alpha_aic_ = model_aic.alpha_
    
    if(plot):
        plt.figure(fig_idx)
        plot_ic_criterion(model_aic, 'AIC', 'b')
        plot_ic_criterion(model_bic, 'BIC', 'r')
        plt.legend()
        plt.title('Information-criterion for model selection (training time %.3fs)'
                  % t_bic)
        fig_idx = fig_idx+1
    
    print("\n LassoLarsBIC score: " + str(model_bic.score(X,y)))
    print("\n LassoLarsAIC score: " + str(model_aic.score(X,y)))
    
    # #############################################################################
    # LassoCV: coordinate descent
    
    # Compute paths
    print("Computing regularization path using the coordinate descent lasso...")
    t1 = time.time()
    model_CV = LassoCV(cv=30).fit(X, y)
    t_lasso_cv = time.time() - t1
    
    # Display results
    if(plot):
        m_log_alphas = model_CV.alphas_
        plt.figure(fig_idx)
    #        ymin, ymax = 2300, 3800
        plt.plot(m_log_alphas, model_CV.mse_path_, ':')
        plt.plot(m_log_alphas, model_CV.mse_path_.mean(axis=-1), 'k',
                 label='Average across the folds', linewidth=2)
        plt.axvline(model_CV.alpha_, linestyle='--', color='k',
                    label='alpha: CV estimate')
        
        plt.legend()
        
        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.title('Mean square error on each fold: coordinate descent '
                  '(train time: %.2fs)' % t_lasso_cv)
        plt.axis('tight')
    #        plt.ylim(ymin, ymax)
        
        fig_idx = fig_idx+1
        
    
    print("\n LassoCV score: " + str(model_CV.score(X,y)))
    
    # #############################################################################
    # LassoLarsCV: least angle regression
    
    # Compute paths
    print("Computing regularization path using the Lars lasso...")
    t1 = time.time()
    model_lassolars = LassoLarsCV(cv=30).fit(X, y)
    t_lasso_lars_cv = time.time() - t1
    
    # Display results
    if(plot):
        m_log_alphas = -np.log10(model_lassolars.cv_alphas_)
        
        plt.figure(fig_idx)
        plt.plot(m_log_alphas, model_lassolars.mse_path_, ':')
        plt.plot(m_log_alphas, model_lassolars.mse_path_.mean(axis=-1), 'k',
                 label='Average across the folds', linewidth=2)
        plt.axvline(-np.log10(model_lassolars.alpha_), linestyle='--', color='k',
                    label='alpha CV')
        plt.legend()
        
        plt.xlabel('-log(alpha)')
        plt.ylabel('Mean square error')
        plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
                  % t_lasso_lars_cv)
        plt.axis('tight')
    #        plt.ylim(ymin, ymax)
        
        plt.show()
        fig_idx = fig_idx+1
    
    print("\n LassoLarsCV score: " + str(model_lassolars.score(X,y)))
    
    return model_CV
    

def generate_SMB_models(SMB_raw, season_raw_meteo_anomalies_SMB, mon_temp_anomalies, mon_snow_anomalies, spatiotemporal_flag, first_year, glims_rabatel, glims_2015, glims_2003, glims_1985, glims_1967, fig_idx):
    
        
    # We create the full matrix for the spatiotemporal multiple linear regression
    glacier_mean_altitude = interpolate_extended_glims_variable('MEAN_Pixel', glims_rabatel, glims_2003, glims_1985, glims_1967)
    glacier_area = interpolate_extended_glims_variable('Area', glims_rabatel, glims_2003, glims_1985, glims_1967)
    glims_IDs = glims_rabatel['GLIMS_ID']
    
    x_reg_full_array, x_reg_nn, x_reg_array, y_reg_array = create_spatiotemporal_matrix(SMB_raw, season_raw_meteo_anomalies_SMB, mon_temp_anomalies, mon_snow_anomalies, glims_rabatel, glacier_mean_altitude, glacier_area, spatiotemporal_flag, first_year)
    
    norm_scaler_array, logo_scaler_array = [],[]
    best_models, logo_models = [],[]
    
    # OLS cross-validation
    ## Leave-one-group-out cross validation ###
    logo = LeaveOneGroupOut()
    groups = []
    group_n = 1
    for glacier in SMB_raw:
        current_group = np.repeat(group_n, np.count_nonzero(~np.isnan(glacier)))
        if(len(current_group) > 32):
            current_group[:-33] = 0
        groups = np.concatenate((groups, current_group), axis=None)
        
        group_n = group_n+1
        
    # Single-year folds
    year_groups = []
    current_group = 1
    
    for glacier in SMB_raw:
        for year in range(1, SMB_raw.shape[1]+1):
            year_groups.append(year)
    year_groups = np.asarray(year_groups)  
    
    finite_mask = np.isfinite(SMB_raw.flatten())
    year_groups = year_groups[finite_mask] - 25

    # Remove negative fold indexes and set them to 0 (not used in CV)
    year_groups = np.where(year_groups < 0, 0, year_groups)
    
    loyo = LeaveOneGroupOut()
    
    # We iterate through the top subset models
    SMB_ols_all = []
    for x_reg, y_reg in zip(x_reg_array, y_reg_array):
        SMB_ols_model, SMB_ols_ref = [],[]
        norm_scaler = StandardScaler()
        scaled_x_reg = sm.add_constant(x_reg)
        
        # Original SINGLE model 
        model = sm.OLS(y_reg, scaled_x_reg)
        model_fit = model.fit()
        best_models.append(model)
        norm_scaler_array.append(norm_scaler)
        
        if(spatiotemporal_flag == 'spatial'):
            splits = logo.split(scaled_x_reg, groups=groups)
        elif(spatiotemporal_flag == 'temporal'):
            splits = loyo.split(scaled_x_reg, groups=year_groups)
        
        xreg_logo_models = []
        xreg_logo_scaler = []
        for (train_idx, test_idx), glimsID, SMB_glacier in zip(splits, glims_IDs, SMB_raw):
            logo_scaler = StandardScaler()
            finite_mask = np.isfinite(SMB_glacier)

            model = sm.OLS(y_reg[train_idx], scaled_x_reg[train_idx])
            model_fit = model.fit()
            
            SMB_test_predict = model_fit.predict(scaled_x_reg[test_idx])
            SMB_ols_model = np.concatenate((SMB_ols_model, SMB_test_predict), axis=None)
            SMB_ols_ref = np.concatenate((SMB_ols_ref, y_reg[test_idx]), axis=None)
            
            xreg_logo_models.append(model)
            xreg_logo_scaler.append(logo_scaler)
        
        SMB_ols_model = np.asarray(SMB_ols_model)
        SMB_ols_all.append(SMB_ols_model)
        xreg_logo_models = np.asarray(xreg_logo_models)
        logo_models.append(xreg_logo_models)
        xreg_logo_scaler = np.asarray(xreg_logo_scaler)
        logo_scaler_array.append(xreg_logo_scaler)
        
    SMB_ols_all = np.asarray(SMB_ols_all)
    SMB_ols = np.mean(SMB_ols_all, axis=0)
    SMB_ols_ref = np.asarray(SMB_ols_ref)
    
    y = np.asarray(y_reg_array)[0,:]
    print("\nOLS RMSE: " + str(math.sqrt(mean_squared_error(SMB_ols_ref, SMB_ols))))
    print("OLS r2: " + str(r2_score(SMB_ols_ref, SMB_ols)))
    
    
    #############################################

     ### Lasso CV
    x_reg_full_array = np.asarray(x_reg_full_array)
    x_reg_full_array = x_reg_full_array[0,:,:]
    
    # Scale training data
    full_scaler = StandardScaler()
    X_full_scaled = full_scaler.fit_transform(x_reg_full_array)
    y = np.asarray(y_reg_array)
    y = y[0,:]
    
    # Activate in order to test different Lasso versions using CV 
    model_CV = lasso_CV_model_selection(X_full_scaled, y, fig_idx, year_groups, plot=False)
    
    #############   CHECKING RESULTS VIA CROSS-VALIDATION   ###########################
    print("\nStarting cross-validation...")
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    if(spatiotemporal_flag == 'spatial'):
        logo = LeaveOneGroupOut()
        lasso_splits = logo.split(X_full_scaled, groups=groups)
        ann_splits = logo.split(x_reg_nn, groups=groups)
        if(settings.smb_model_type == 'ann_no_weights'):
            path_cv_ann = workspace + 'glacier_data\\smb\\ANN\\LOGO\\no_weights\\CV\\'
        elif(settings.smb_model_type == 'ann_weights'):
            path_cv_ann = workspace + 'glacier_data\\smb\\ANN\\LOGO\\weights\\CV\\'
        
    elif(spatiotemporal_flag == 'temporal'):
        loyo = LeaveOneGroupOut()
        lasso_splits = loyo.split(X_full_scaled, groups=year_groups)
        ann_splits = loyo.split(x_reg_nn, groups=year_groups)
        if(settings.smb_model_type == 'ann_no_weights'):
            path_cv_ann = workspace + 'glacier_data\\smb\\ANN\\LOYO\\no_weights\\CV\\'
        elif(settings.smb_model_type == 'ann_weights'):
            path_cv_ann = workspace + 'glacier_data\\smb\\ANN\\LOYO\\weights\\CV\\'
    
    lasso_cv_models = []
    SMB_lasso_all , SMB_ann_all = [],[]
    y_ref_lasso, y_ref_ann = [],[]
    ref_weights_ann = []
    rmse_ann, mae_ann, bias_ann = [],[],[]
    mae_lasso, bias_lasso = [],[]
    fold_idx = 0
    for (train_idx, test_idx), (train_ann_idx, test_ann_idx), glimsID, SMB_glacier in zip(lasso_splits, ann_splits, glims_IDs, SMB_raw):
 
        if(fold_idx > 0): # We filter the first fold for LOYO with the years outside 1984-2014
            print("\nFold " + str(fold_idx))
            print("test_idx: " + str(test_idx))
            
            # Lasso
            scaled_X_train = X_full_scaled[train_idx]
            scaled_X_test = X_full_scaled[test_idx]
            
            if(spatiotemporal_flag == 'spatial'):
                logo_fold = LeaveOneGroupOut()
                lasso_fold_splits = logo_fold.split(scaled_X_train, groups=groups[groups != fold_idx])
            elif(spatiotemporal_flag == 'temporal'):
                loyo_fold = LeaveOneGroupOut()
                year_groups_u = year_groups[year_groups != fold_idx]
#                lasso_fold_splits = loyo_fold.split(scaled_X_train, groups=year_groups_u)
                
#            logo_model_lasso = LassoLarsIC(criterion='bic').fit(scaled_X_train, y[train_idx])
            cv_model_lasso = LassoCV(cv=150, selection='random').fit(scaled_X_train, y[train_idx])
#            cv_model_lasso = LassoCV(cv=lasso_fold_splits, selection='random').fit(scaled_X_train, y[train_idx])
            lasso_cv_models.append(cv_model_lasso)
            
            SMB_lasso = cv_model_lasso.predict(scaled_X_test)
            SMB_lasso_all = np.concatenate((SMB_lasso_all, SMB_lasso), axis=None)
            
            y_ref_lasso = np.concatenate((y_ref_lasso, y[test_idx]), axis=None)
            
            lasso_mae_fold = np.abs(SMB_lasso - y[test_idx]).mean()
            mae_lasso.append(lasso_mae_fold)
            lasso_bias_fold = SMB_lasso.mean() - y[test_idx].mean()
            bias_lasso.append(lasso_bias_fold)
            
            print("\nLasso RMSE (m.w.e): " + str((math.sqrt(mean_squared_error(y[test_idx], SMB_lasso)))))
            print("Lasso r2 (fold): " + str(r2_score(y[test_idx], SMB_lasso)))
            
            # ANN
            # We retrie the CV ANN model
            cv_ann_model = load_model(path_cv_ann + 'glacier_' + str(fold_idx) + '_model.h5', custom_objects={"r2_keras": r2_keras, "root_mean_squared_error": root_mean_squared_error})
            
#            import pdb; pdb.set_trace()
            
            SMB_nn = cv_ann_model.predict(x_reg_nn[test_ann_idx], batch_size = 34)
            SMB_ann_all = np.concatenate((SMB_ann_all, SMB_nn), axis=None)
            
            y_ref_ann = np.concatenate((y_ref_ann, y[test_ann_idx]), axis=None)
            ref_weights_ann = np.concatenate((ref_weights_ann, sample_weights[test_ann_idx]), axis=None)
            
            SMB_nn = SMB_nn.flatten()
            ann_mae_fold = np.abs(SMB_nn - y[test_ann_idx]).mean()
            
            mae_ann.append(ann_mae_fold)
            ann_bias_fold = SMB_nn.mean() - y[test_ann_idx].mean()
            bias_ann.append(ann_bias_fold)
            
            print("\nANN RMSE (m.w.e): " + str((math.sqrt(mean_squared_error(y[test_idx], SMB_nn)))))
            print("ANN r2 (fold): " + str(r2_score(y[test_idx], SMB_nn)))
            
            if(settings.smb_model_type == 'ann_weights'):
                rmse_ann.append(math.sqrt(mean_squared_error(y[test_ann_idx], SMB_nn, sample_weights[test_ann_idx])))
            
            elif(settings.smb_model_type == 'ann_no_weights'):
                rmse_ann.append(math.sqrt(mean_squared_error(y[test_ann_idx], SMB_nn)))
            
        fold_idx = fold_idx+1
    
    y_ref_lasso = np.asarray(y_ref_lasso)
    y_ref_ann = np.asarray(y_ref_ann)
    ref_weights_ann = np.asarray(ref_weights_ann)
    
    print("\nLasso RMSE: " + str(math.sqrt(mean_squared_error(y_ref_lasso, SMB_lasso_all))))
    print("Lasso r2: " + str(r2_score(y_ref_lasso, SMB_lasso_all)))
    
    rmse_ann = np.asarray(rmse_ann)
    mae_ann = np.asarray(mae_ann)
    
    print("MAE ANN: " + str(mae_ann.mean()))
    print("RMSE ANN: " + str(rmse_ann))
    
    if(settings.smb_model_type == 'ann_weights'):
        print("\nANN RMSE: " + str(math.sqrt(mean_squared_error(y_ref_ann, SMB_ann_all, ref_weights_ann))))
        print("ANN r2: " + str(r2_score(y_ref_ann, SMB_ann_all, ref_weights_ann)))
        
    elif(settings.smb_model_type == 'ann_no_weights'):
        print("\nANN RMSE: " + str(math.sqrt(mean_squared_error(y_ref_ann, SMB_ann_all))))
        print("ANN r2: " + str(r2_score(y_ref_ann, SMB_ann_all)))
    
        ####  ANN   ##############
    # Calculate the point density
    xy = np.vstack([y_ref_ann,SMB_ann_all])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ann_plt, z = y_ref_ann[idx], SMB_ann_all[idx], z[idx]
    
    plt.figure(fig_idx, figsize=(6,6))
    if(settings.smb_model_type == 'ann_no_weights'):
        plt.title("Deep learning without weights", fontsize=17)
    elif(settings.smb_model_type == 'ann_weights'):
        plt.title("Deep learning with weights", fontsize=17)
    plt.ylabel('SMB modeled with ANN (m.w.e)', fontsize=16)
    plt.xlabel('Reference SMB data (m.w.e)', fontsize=16)
    sc = plt.scatter(y_plt, ann_plt, c=z, s=50)
    plt.clim(0,0.4)
    plt.tick_params(labelsize=14)
    plt.colorbar(sc)
    lineStart = y_ref_ann.min() 
    lineEnd = y_ref_ann.max()  
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.show()
    fig_idx = fig_idx+1
    
    plt.figure(fig_idx)
    density_nn = gaussian_kde(SMB_ann_all)
    density = gaussian_kde(y)
    xs = np.linspace(-4,3,200)
    density.covariance_factor = lambda : .25
    density_nn.covariance_factor = lambda : .25
    density._compute_covariance()
    density_nn._compute_covariance()
    plt.figure(fig_idx)
    plt.title("Neural network SMB density distribution")
    plt.plot(xs,density(xs), label='Ground truth')
    plt.plot(xs,density_nn(xs), label='NN simulation')
    plt.legend()
    plt.show()
    
    fig_idx = fig_idx+1
    
    
    # Error distribution
    plt.figure(fig_idx)
    SMB_ann_errors = np.abs(y_ref_ann - SMB_ann_all)
    pf = poly.Polynomial.fit(y_ref_ann, SMB_ann_errors, 5)
    x_pf = np.asarray(*pf.linspace(n=y_ref_ann.size)[:1]).flatten()
    y_pf = np.asarray(*pf.linspace(n=SMB_ann_errors.size)[1:]).flatten()
#    bins = np.linspace(-4, 3, 70)
#    digitized = np.digitize(y_re)
    plt.scatter(y_ref_ann, SMB_ann_errors, s=30, alpha=0.2, c='red')
    plt.plot(x_pf, y_pf, c='darkred', linewidth=4)
#    plt.title("Glacier-wide SMB error distribution")
    plt.ylabel('Error (m.w.e)', fontsize=16)
    plt.xlabel('Deep learning glacier-wide SMB (m.w.e)', fontsize=16)
    plt.ylim(0, 2.7)
#    plt.legend()
    plt.show()
    
    ####################################
    
    #####  LASSO   ##############
    # Calculate the point density
    xy = np.vstack([y_ref_lasso,SMB_lasso_all])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, lasso_plt, z = y_ref_lasso[idx], SMB_lasso_all[idx], z[idx]
    
    plt.figure(fig_idx, figsize=(6,6))
    plt.title("Lasso", fontsize=20)
    plt.ylabel('SMB modeled with Lasso (m.w.e)', fontsize=16)
    plt.xlabel('Reference SMB data (m.w.e)', fontsize=16)
    sc = plt.scatter(y_plt, lasso_plt, c=z, s=50, cmap='plasma')
    plt.colorbar(sc)
    plt.tick_params(labelsize=14)
    lineStart = y_ref_lasso.min() 
    lineEnd = y_ref_lasso.max()  
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.clim(0,0.4)
    plt.show()
    fig_idx = fig_idx+1
    
    plt.figure(fig_idx)
    density_nn = gaussian_kde(SMB_lasso_all)
    density = gaussian_kde(y)
    xs = np.linspace(-4,3,200)
    density.covariance_factor = lambda : .25
    density_nn.covariance_factor = lambda : .25
    density._compute_covariance()
    density_nn._compute_covariance()
    plt.figure(fig_idx)
    plt.title("Lasso SMB density distribution")
    plt.plot(xs,density(xs), label='Ground truth')
    plt.plot(xs,density_nn(xs), label='Lasso simulation')
    plt.legend()
    plt.show()
    
    lasso_cv_models = np.asarray(lasso_cv_models)
    
     #####  OLS   ##############
    # Calculate the point density
    xy = np.vstack([SMB_ols_ref,SMB_ols])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ols_plt, z = SMB_ols_ref[idx], SMB_ols[idx], z[idx]
    
    plt.figure(fig_idx, figsize=(6,6))
    plt.title("OLS", fontsize=20)
    plt.ylabel('SMB modeled with OLS (m.w.e)', fontsize=16)
    plt.xlabel('Reference SMB data (m.w.e)', fontsize=16)
    sc = plt.scatter(y_plt, ols_plt, c=z, s=50, cmap='plasma')
    plt.colorbar(sc)
    plt.tick_params(labelsize=14)
    lineStart = SMB_ols_ref.min() 
    lineEnd = SMB_ols_ref.max()  
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'black')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.clim(0,0.4)
    plt.show()
    fig_idx = fig_idx+1
    
    plt.figure(fig_idx)
    density_ols = gaussian_kde(SMB_ols)
    density = gaussian_kde(y)
    xs = np.linspace(-4,3,200)
    density.covariance_factor = lambda : .25
    density_ols.covariance_factor = lambda : .25
    density._compute_covariance()
    density_ols._compute_covariance()
    plt.figure(fig_idx)
    plt.title("OLS SMB density distribution")
    plt.plot(xs,density(xs), label='Ground truth')
    plt.plot(xs,density_ols(xs), label='OLS simulation')
    plt.legend()
    plt.show()
    
    # Save LassoLars model
    with open(path_smb+'smb_function\\model_lasso_' + str(spatiotemporal_flag) + '.txt', 'wb') as model_lasso_gbl_f:
        pickle.dump(model_CV, model_lasso_gbl_f)
        
    # Save data scaler
    with open(path_smb+'smb_function\\full_scaler_' + str(spatiotemporal_flag) + '.txt', 'wb') as full_scaler_f:
        np.save(full_scaler_f, full_scaler)
    with open(path_smb+'smb_function\\norm_scaler_array_' + str(spatiotemporal_flag) + '.txt', 'wb') as norm_scaler_f:
        np.save(norm_scaler_f, norm_scaler_array)
    
    return model_fit._results.rsquared, best_models, logo_models, logo_scaler_array, lasso_cv_models

def get_raw_mon_data(local_mon_temp_anomalies_SMB, key):
    mon_temp_anomalies = []
    for glacier_temp in local_mon_temp_anomalies_SMB:
        mon_temp_anomalies.append(glacier_temp[key])

    mon_temp_anomalies = np.asarray(mon_temp_anomalies)
    return mon_temp_anomalies



def main(compute):

    ################################################################################
    ##################		                MAIN               	#####################
    ################################################################################  
    
    print("\n-----------------------------------------------")
    print("          SMB MODEL TRAINING")
    print("-----------------------------------------------\n")
        
    if(compute):
        #####  READ CLIMATIC FORCING TO BE USED  #########
        
        forcing = settings.historical_forcing
        
        ######################################################
            
        # We read the SMB from the csv file
        SMB_raw_full = genfromtxt(path_smb + 'SMB_raw_temporal.csv', delimiter=';', dtype=float)
        
        #### GLIMS data for 1985, 2003 and 2015
        glims_2015 = genfromtxt(path_glims + 'GLIMS_2015.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50')])
        glims_2003 = genfromtxt(path_glims + 'GLIMS_2003.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<i8'), ('Aspect_num', '<i8')])
        glims_1985 = genfromtxt(path_glims + 'GLIMS_1985.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50')])
        glims_1967 = genfromtxt(path_glims + 'GLIMS_1967.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('GLIMS_ID', '<a50')])

        
        ####  GLIMS data for the 30 glaciers with remote sensing SMB data (Rabatel et al. 2016)   ####
        glims_rabatel = genfromtxt(path_glims + 'GLIMS_Rabatel_30_2015.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8'), ('slope20', '<f8'), ('GLIMS_ID', '<a50'), ('Massif_SAFRAN', '<f8'), ('Aspect_num', '<f8'), ('slope20_evo', '<f8'),])        
        path_smb_function_forcing = path_smb_function + forcing + '\\'
        
        # We open all the files with the data to be modelled
        # Glaciers with SMB data (Rabatel et al. 2016)
        # We load the compacted seasonal and monthly meteo forcings
        with open(path_smb_function_forcing+'season_meteo_anomalies_SMB.txt', 'rb') as season_a_f:
            season_meteo_anomalies_SMB = np.load(season_a_f)[()]
        with open(path_smb_function_forcing+'season_raw_meteo_anomalies_SMB.txt', 'rb') as season_raw_a_f:
            season_raw_meteo_anomalies_SMB = np.load(season_raw_a_f)[()]
        with open(path_smb_function_forcing+'monthly_meteo_anomalies_SMB.txt', 'rb') as mon_a_f:
            monthly_meteo_anomalies_SMB = np.load(mon_a_f)[()]
            
        # We get the raw forcings
        mon_temp_anomalies =  get_raw_mon_data(monthly_meteo_anomalies_SMB['temp'], 'mon_temp')
        mon_snow_anomalies =  get_raw_mon_data(monthly_meteo_anomalies_SMB['snow'], 'mon_snow')
        
        year_start = int(season_meteo_anomalies_SMB['CPDD'][0]['CPDD'][0])
#        year_end = int(season_meteo_anomalies_SMB['CPDD'][0]['CPDD'][-2])  
        
        first_SMB_year = 1959
#        first_SMB_year = 1984
        first_year = first_SMB_year - year_start
        
        
        ###########   PLOTS   ########################################################
        
        fig_idx = 1
        
        # We generate the cross-validation machine learning SMB models
        # Temporal models for the 1959-2014 - onwards period
        print("\nTemporal dimension training/validation")
        r2_period_gbl_loyo, model_gbl_loyo, loyo_gbl_models, loyo_scaler_array, lasso_loyo_models = generate_SMB_models(SMB_raw_full, season_raw_meteo_anomalies_SMB, 
                                                                                                                                  mon_temp_anomalies, mon_snow_anomalies, 'temporal', first_year, 
                                                                                                                                  glims_rabatel, glims_2015, glims_2003, glims_1985, glims_1967, fig_idx)
        
        # Spatial models for the 1984-2014 period
        print("\nSpatial dimension training/validation")
        r2_period_gbl_logo, model_gbl_logo, logo_gbl_models, logo_scaler_array, lasso_logo_models = generate_SMB_models(SMB_raw_full, season_raw_meteo_anomalies_SMB, 
                                                                                                                                  mon_temp_anomalies, mon_snow_anomalies, 'spatial', first_year,
                                                                                                                                  glims_rabatel, glims_2015, glims_2003, glims_1985, glims_1967, fig_idx)
        
        # We store all the data in files        
        # LOYO
        midfolder = 'smb_function\\LOYO\\'
        if not os.path.exists(path_smb + midfolder):
            os.makedirs(path_smb + midfolder)
            
        with open(path_smb + midfolder + 'lasso_loyo_models.txt', 'wb') as lasso_loyos_f:
                pickle.dump(lasso_loyo_models, lasso_loyos_f)     
                
        # LOGO
        midfolder = 'smb_function\\LOGO\\'
        if not os.path.exists(path_smb + midfolder):
            os.makedirs(path_smb + midfolder)
            
        # We store the model_fits
        with open(path_smb + midfolder + 'model_gbl_logo.txt', 'wb') as model_gbl_f:
            np.save(model_gbl_f, model_gbl_logo)
            
        ## We store all the Leave-one-group-out model combinations for proper cross validation in smb_projection.py
        with open(path_smb + midfolder + 'logo_gbl_models.txt', 'wb') as logos_f:
                    np.save(logos_f, logo_gbl_models)
        with open(path_smb + midfolder + 'logo_scaler_array.txt', 'wb') as logo_scaler_f:
                    np.save(logo_scaler_f, logo_scaler_array)
        with open(path_smb + midfolder + 'lasso_logo_models.txt', 'wb') as lasso_logos_f:
                    np.save(lasso_logos_f, lasso_logo_models)
                    
    else:
        print("Skipping...")
    ### End of main function  ###