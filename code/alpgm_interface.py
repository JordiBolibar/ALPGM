# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@author: Jordi Bolibar
Institut des Géosciences de l'Environnement (Université Grenoble Alpes)
jordi.bolibar@univ-grenoble-alpes.fr

* ALPINE PARAMETERIZED GLACIER MODEL INTERFACE *


Model workflow:
    
ALPGM:
    - delta h: delta_h_alps.py ----------------------------------------------------------------------> | ---> glacier_evolution.py
    - smb:  - safran_forcings.py --| ----> smb_model_training.py --> smb_validation.py -----------------------------> |
            - adamont_forcings.py ------------------------------------------------------------------------------------|

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import settings
import imp

################################################################################
##################		        ALPGM WORKFLOW             #####################
################################################################################ 


print("\n-----------------------------------------------")
print("              Launching ALPGM...")
print("-----------------------------------------------")

#########    SETTINGS      ##############################################################################
# projection == True -> Projections with ADAMONT for the 21st century
# projection == False -> Historical simulations for the 1984 - 2015 period with SAFRAN
historical_forcing, projection_forcing, simulation_type = settings.simulation_settings(projection = True)

### Global variables  ###
# Set the glacier index to start the simulations
counter_threshold = 0

imp.reload(settings)
#######   Choose a SMB model:   #########################################
# 'ann_no_weights': Deep Artificial Neural Network without sample weights
# 'ann_weights': Deep Artificial Neural Network with sample weights
# 'lasso': Lasso linear model
settings.init(historical_forcing, projection_forcing, simulation_type, smb_model = 'ann_no_weights')

##########    WORKFLOW     ################################################################################

##########    SMB SIMULATION   ######################################
# SMB machine learning models generation

settings.train_smb_model(historical_forcing, compute_forcing = False, # Compute historical climate forcings
                                                 train_model = False) # Re-train SMB machine learning models

##########     DELTA H FUNCTIONS GENERATION   #######################

settings.glacier_parameterized_functions(compute = False,
                                       overwrite = True)

##########     SMB PROJECTION + GLACIER GEOMETRY EVOLUTION    #######
settings.glacier_simulation(simulation_type, counter_threshold,
                                           validate_SMB = False, # SMB model(s) validation or reconstruction
                            compute_projection_forcings = False, # Compute projection climate forcings
                                      compute_evolution = True, # Compute glacier evolution
                                             reconstruct = True, # Reconstruct glacier-wide SMB timeseries
                                               overwrite = True)


#############################################################################################################