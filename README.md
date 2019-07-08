# ALPGM (ALpine Parameterized Glacier Model) v1.0

![ALPGM](https://www.dropbox.com/s/8zycrf67lloppr5/algpm_logo2.png?raw=1)

[![DOI](https://zenodo.org/badge/195388796.svg)](https://zenodo.org/badge/latestdoi/195388796)

#### Author 
<p><b>Jordi Bolíbar</b></p>
<p>jordi.bolibar@univ-grenoble-alpes.fr</p>
<p>Institut des Géosciences de l'Environnement (Université Grenoble Alpes)</p>

## Overview
<p>
    ALPGM is a fully parameterized glacier evolution model based on machine learning. Glacier-wide surface mass balance (SMB) is simulated using a deep artificial neural network (deep learning) or Lasso. 
    Glacier dynamics are parameterized using glacier-specific delta-h functions (Huss et al. 2008). The model has been implemented with a dataset of French alpine glaciers, using climate forcings
    for past (SAFRAN, Durand et al. 1993) and future (ADAMONT, Verfaillie et al. 2018) periods.
</p>

<p>
    The machine learning SMB modelling approach is built on widely used Python libraries (Keras, Scikit-learn and Statsmodels). 
</p>

## Workflow
<p>
    ALPGM's workflow can be controlled via the alpgm_interface.py file. In this file, different settings can be configured, and each step can be run or skipped with a boolean flag. 
    The default workflow runs as it follows:
</p>

<p>
    <b>(1)</b> First of all, the meteorological forcings are pre-processed (safran_forcings.py / adamont_forcings.py) in order to extract the necessary data closest to each glacier’s centroid. The meteorological features are stored in intermediate files in order 
    to reduce computation times for future runs, automatically skipping  this preprocessing step when the files are already generated. 
    <br><br>
    <b>(2)</b> The SMB machine learning module retrieves the pre-processed meteorological features and assembles the spatio-temporal training dataset, comprised by both climatic and topographical data. An algorithm is 
    chosen for the SMB model, which can be loaded from a previous training or it can be trained again with the training dataset (smb_model_training.py). These model(s) are stored in intermediate files, allowing to skip this step for future runs.
    <br><br>
    <b>(3)</b> The performances of these SMB models can be evaluated performing a leave-one-glacier-out (LOGO) cross-validation (smb_validation.py). This step can be skipped when using already established models. Basic statistical performance 
    metrics are given for each glacier and model, as well as plots with the simulated cumulative glacier-wide SMBs compared to their reference values with uncertainties for each of the glaciers from the training dataset.
    <br><br>
    <b>(4)</b> The Glacier Geometry Update module starts with the generation of the glacier specific parameterized functions, using the difference of the two pre-selected digital elevation model (DEM) rasters covering the 
    whole study area for two separate dates, as well as the glacier contours (delta_h_alps.py). These parameterized functions are then stored in individual files to be used in the final simulations.
    <br><br>
    <b>(5)</b> Once all the previous steps have been run and the glacier-wide SMB models as well as the parameterized functions for all the glaciers are obtained, the final simulations are launched (glacier_evolution.py). 
    For each glacier, the initial ice thickness raster and the parameterized function are retrieved. The meteorological data at the glaciers’ centroid is re-computed with an annual time step based on each glacier’s evolving topographical 
    characteristics. These forcings are used to simulate the annual glacier-wide SMB using the machine learning model. Once an annual glacier-wide SMB value is obtained, the changes in geometry are computed using the 
    parameterized function, thus updating the glacier’s DEM and ice thickness rasters. If all the ice thickness raster pixels of a glacier become zero, the glacier is considered as disappeared and is removed from the 
    simulation pipeline. For each year, multiple results are stored in data files as well as the raster DEM and ice thickness values for each glacier.
</p>

## SMB machine learning model(s)

<p>
    ALPGM simulates glacier-wide SMBs using topographical and climate data at the glacier. This repository comes with some pre-trained SMB models, but they can be retrained again at will with new data. 
    Retraining is important when working with a different region (outside the European Alps in this case), or when expanding the training dataset in order to improve the model's performance.
    <br><br>
    Two main models can be chosen for the SMB simulations:
    <br><br>
    <b>Deep Artificial Neural Network</b>: A deep ANN, also know as deep learning, is a nonlinear statistical model capable of finding extremely complex patters in data. This approach represents an improvement with 
    respect to classical linear methods, such as multiple linear regression. They allow the use of sample weights in order to balance the training dataset to give equal importance to all the sample values. This can help
    the ANN to better simulate extreme SMB values, thus increasing the explained variance of the model (r<sup>2</sup>) at the cost of sacrificing the overall accuracy of the model (RMSE). In order to use it for simulations, 
    choose the "ann_weights" or "ann_no_weights" models in alpgm_interface.py
    <br><br>
    <b>Lasso</b>: The Lasso (Least absolute shrinkage and selection operator) (Tibshirani, 1996), is a shrinkage method which attempts to overcome the shortcomings of the simpler step-wise and all-possible regressions. 
	In these two classical approaches, predictors are discarded in a discrete way, giving subsets of variables which have the lowest prediction error. However, due to its discrete selection, these different subsets can exhibit high variance, 
	which does not reduce the prediction error of the full model. The Lasso performs a more continuous regularization by shrinking some coefficients and setting others to zero, thus producing more interpretable models (Hastie et al., 2009). 
	Because of its properties, it strikes a balance between subset selection (like all-possible regressions) and Ridge regression (Hoerl and Kennard, 1970)
</p>

## Included data
<p>
    All the data needed to run the French alpine glaciers case study simulations is available in this repository: the topographical and SMB data for the glaciers,
	the glacier-specific delta-h parameterized functions, and the initial glacier ice thickness for the all the glaciers in the region (Farinotti et al. 2019). 
	With the exception of the SAFRAN (Durand et al. 2009) climate data preprocessed files, which can be [downloaded here](https://www.dropbox.com/s/2kisbxk2ajaunh2/SAFRAN_meteo_data.rar?raw=1) separately due to their size.
	
</p>
