# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 15:46:00 2018

@author: bolibarj
"""

## Dependencies: ##
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import numpy.polynomial.polynomial as poly
from scipy import stats
from scipy.optimize import curve_fit

######   FILE PATHS    #######
    
# Folders     
workspace = 'C:\\Jordi\\PhD\\Python\\'
#path_obs = 'C:\\Jordi\\PhD\\Data\\Obs\\'
path_smb = workspace + 'ALPGM\\glacier_data\\smb\\'
path_glacier_coordinates = workspace + 'ALPGM\\glacier_data\\glacier_coordinates\\'
path_smb_simulations = path_smb + 'smb_simulations\\'

####  GLIMS data for the 30 glaciers with remote sensing SMB data (Rabatel et al. 2016)   ####
glims_glaciers = genfromtxt(path_glacier_coordinates + 'GLIMS_Rabatel_30.csv', delimiter=';', skip_header=1,  dtype=[('Area', '<f8'), ('Perimeter', '<f8'), ('Glacier', '<a50'), ('Annee', '<i8'), ('Massif', '<a50'), ('MEAN_Pixel', '<f8'), ('MIN_Pixel', '<f8'), ('MAX_Pixel', '<f8'), ('MEDIAN_Pixel', '<f8'), ('Length', '<f8'), ('Aspect', '<a50'), ('x_coord', '<f8'), ('y_coord', '<f8')])

####  SMB data for the 30 glaciers  #####
smb_glaciers_tag = genfromtxt(path_smb + 'SMB.csv', delimiter=';', dtype=[('Glacier_name', '<a50'), ('1984', '<f8'), ('1985', '<f8'), ('1986', '<f8'), ('1987', '<f8'), ('1988', '<f8'), ('1989', '<f8'), ('1990', '<f8'), ('1991', '<f8'), ('1992', '<f8'), ('1993', '<f8'), ('1994', '<f8'), ('1995', '<f8'), ('1996', '<f8'), ('1997', '<f8'), ('1998', '<f8'), ('1999', '<f8'), ('2000', '<f8'), ('2001', '<f8'), ('2002', '<f8'), ('2003', '<f8'), ('2004', '<f8'), ('2005', '<f8'), ('2006', '<f8'), ('2007', '<f8'), ('2008', '<f8'), ('2009', '<f8'), ('2010', '<f8'), ('2011', '<f8'), ('2012', '<f8'), ('2013', '<f8'), ('2014', '<f8')]) 
smb_glaciers = genfromtxt(path_smb + 'SMB.csv', delimiter=';') 

#### Slope on the lowermost 20% area of the glaciers #####
slope20_glaciers = genfromtxt(path_glacier_coordinates + 'glacier_low20_slope.csv', delimiter=';')

######   FUNCTIONS    #########

def fn1(x, a, b):
    return x*a + b

def fn3(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[2]

def compute_t_statistics(ref_data, simu_data):
    ## Calculate the Standard Deviation
    #Calculate the variance to get the standard deviation
#    N = ref_data.size
#    
#    print("ref_data: " + str(ref_data))
#    print("simu_data: " + str(simu_data))
#    
#    #For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
#    var_ref_smb = ref_data.var(ddof=1)
#    var_simu_smb = simu_data.var(ddof=1)
#    
#    #Std Deviation
#    s = np.sqrt((var_ref_smb + var_simu_smb)/2)
#    
#    # We compute the t-statistics
#    t = (ref_data.mean() - simu_data.mean())/(s*np.sqrt(2/N))
#    
#    # Compare with the critical t-value
#    # Degrees of freedom
#    df = 2*N - 2
#    
#    # p-value after comparison with the t
#    p = 1 - stats.t.cdf(t, df=df)
    
    t2, p2 = stats.ttest_rel(ref_data, simu_data, nan_policy='omit')
    
    # Print results
#    print("t = " + str(t))
#    print("p = " + str(p))
    
#    print("t = " + str(t2))
#    print("p = " + str(p2))
    
    return t2, p2

#######   MAIN   ##########

# We compute the average SMB for each glacier and we look for each SMB historical simulation
avg_smb = []
nfigure = 1
for ref_glacier, glacier_tag in zip(smb_glaciers, smb_glaciers_tag):
    glacier_name = glacier_tag[0]
    # We compute the glacier SMB average
    avg_smb.append(np.nanmean(ref_glacier[1:]))
    # We get the simulated historical glacier SMB
    simu_glacier = genfromtxt(path_smb_simulations + str(glacier_name) + '_simu_SMB_1984_2014.csv', delimiter=';')
    
#    print("t statistics for Glacier " + str(glacier_name))
    t, p = compute_t_statistics(ref_glacier[1:], simu_glacier[:,1])
    
#    plt.figure(nfigure)
#    plt.title("Glacier " + str(glacier_name) + " t = " + str(t) + " / p = " + str(p))
#    plt.ylabel('Glacier-wide SMB')
#    plt.xlabel('Year')
#    plt.plot(range(1984, 2014+1), simu_glacier[:,1], markersize=3, label='Simulated SMB')
#    plt.plot(range(1984, 2014+1), ref_glacier[1:], markersize=3, label='Reference remote sensing SMB')
#    plt.legend()
#    nfigure = nfigure+1
                
avg_smb = np.asarray(avg_smb)

# We get the indexes by massif
idx_ecrins = np.append(np.where(glims_glaciers['Massif'] == "Grandes Rousses"), np.where(glims_glaciers['Massif'] == "Ecrins"))
idx_vanoise1 = np.append(np.where(glims_glaciers['Massif'] == "Vanoise"), np.where(glims_glaciers['Massif'] == "Haute Maurienne"))
idx_vanoise = np.append(idx_vanoise1, np.where(glims_glaciers['Massif'] == "Haute Valle de l'Isere"))
idx_montblanc = np.where(glims_glaciers['Massif'] == "Mont Blanc")

#### Multiple linear regression for SMB adjustment based on mean altitude and mean slope of the lowermost 20% area
x_altadj = np.array([glims_glaciers['MEAN_Pixel'], slope20_glaciers[:,1], glims_glaciers['MAX_Pixel']])
y_altadj = np.array(avg_smb)
popt_altadj, pcov_altadj = curve_fit(fn3, x_altadj, y_altadj)

simu_altadj_smb = fn3(x_altadj, popt_altadj[0], popt_altadj[1], popt_altadj[2], popt_altadj[3])
r_multiple_altadj = np.corrcoef(y_altadj, simu_altadj_smb)
r2_multiple_altadj = r_multiple_altadj[1][0]**2

plt.figure(nfigure)
plt.scatter(simu_altadj_smb, avg_smb)
plt.title("SMB reference vs SMB altitude adjustment (r2 = " + str(r2_multiple_altadj))
plt.xlabel("SMB altitude adjustment")
plt.ylabel("SMB reference (1984-2014)")
axes = plt.gca()
#plt.legend(loc=2)
plt.show()
nfigure = nfigure+1



#######        PLOTS         #############
#nfigure = 1

# Median glacier altitude vs SMB
plt.figure(nfigure)
x_median = glims_glaciers['MEDIAN_Pixel']
t_median, p_median = stats.ttest_rel(x_median, avg_smb, nan_policy='omit')
plt.scatter(x_median[idx_ecrins], avg_smb[idx_ecrins], marker="2", s=3, label='Ecrins')
plt.scatter(x_median[idx_vanoise], avg_smb[idx_vanoise], marker="s", s=3, label='Vanoise')
plt.scatter(x_median[idx_montblanc], avg_smb[idx_montblanc], marker="*", s=3, label='Mont Blanc')

popt_median, pcov_median = curve_fit(fn1, x_median, avg_smb)
perr_median = np.sqrt(np.diag(pcov_median))
nstd = 1 # 5 sigma intervals
popt_up_median = popt_median + nstd*perr_median
popt_dw_median = popt_median - nstd*perr_median
x_median_sort = np.sort(x_median)
fit_median = fn1(x_median_sort, *popt_median)
fit_up_median = fn1(x_median_sort, *popt_up_median)
fit_dw_median = fn1(x_median_sort, *popt_dw_median)

#errorbar(glims_glaciers['MEDIAN_Pixel'], avg_sm)
plt.plot(x_median_sort, fit_median, 'r', lw=2, label="Best fit curve")
#plt.plot(glims_glaciers['MEDIAN_Pixel'], fit_up_median, 'b', lw=2, label="Best fit curve")
#plt.plot(glims_glaciers['MEDIAN_Pixel'], fit_dw_median, 'g', lw=2, label="Best fit curve")
#plt.fill_between(x_median_sort, fit_up_median, fit_dw_median, alpha=.25, label="5-sigma interval")

p_ga = poly.Polynomial.fit(x_median, avg_smb, 1)
x_p_ga = np.asarray(*p_ga.linspace(n=x_median.size)[:1]).flatten()
y_p_ga = np.asarray(*p_ga.linspace(n=avg_smb.size)[1:]).flatten()
#plt.plot(x_p_ga, y_p_ga)

ga_idx = np.argsort(x_median)
r_ga = np.corrcoef(avg_smb[ga_idx], y_p_ga)
r2_ga = r_ga[1][0]**2

plt.title("SMB vs median glacier altitude (r2 = " + str(r2_ga) + " / p = " + str('{:.3g}'.format(p_median)) + ")")
plt.xlabel("Glacier median altitude (m)")
plt.ylabel("Average SMB (1984-2014)")
axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()

nfigure = nfigure+1

# Mean glacier altitude vs SMB
plt.figure(nfigure)
x_mean = glims_glaciers['MEAN_Pixel']
t_mean, p_mean = stats.ttest_rel(x_mean, avg_smb, nan_policy='omit')
plt.scatter(x_mean[idx_ecrins], avg_smb[idx_ecrins], marker="2", s=3, label='Ecrins')
plt.scatter(x_mean[idx_vanoise], avg_smb[idx_vanoise], marker="s", s=3, label='Vanoise')
plt.scatter(x_mean[idx_montblanc], avg_smb[idx_montblanc], marker="*", s=3, label='Mont Blanc')
p_ga2 = poly.Polynomial.fit(x_mean, avg_smb, 1)
x_p_ga2 = np.asarray(*p_ga2.linspace(n=x_mean.size)[:1]).flatten()
y_p_ga2 = np.asarray(*p_ga2.linspace(n=avg_smb.size)[1:]).flatten()
plt.plot(x_p_ga2, y_p_ga2)

ga2_idx = np.argsort(x_mean)
r_ga2 = np.corrcoef(avg_smb[ga2_idx], y_p_ga2)
r2_ga2 = r_ga2[1][0]**2

popt_mean, pcov_mean = curve_fit(fn1, x_mean, avg_smb)
perr_mean = np.sqrt(np.diag(pcov_mean))
nstd = 1 # 5 sigma intervals
popt_up_mean = popt_mean + nstd*perr_mean
popt_dw_mean = popt_mean - nstd*perr_mean
x_mean_sort = np.sort(x_mean)
fit_mean = fn1(x_mean_sort, *popt_mean)
fit_up_mean = fn1(x_mean_sort, *popt_up_mean)
fit_dw_mean = fn1(x_mean_sort, *popt_dw_mean)

#errorbar(x_mean, avg_sm)
plt.plot(x_mean_sort, fit_mean, 'r', lw=2, label="Best fit curve")
#plt.plot(x_mean, fit_up_mean, 'b', lw=2, label="Best fit curve")
#plt.plot(x_mean, fit_dw_mean, 'g', lw=2, label="Best fit curve")
#plt.fill_between(x_mean_sort, fit_up_mean, fit_dw_mean, alpha=.25, label="5-sigma interval")

plt.title("SMB vs mean glacier altitude (r2 = " + str(r2_ga2) + " / p = " + str('{:.3g}'.format(p_mean)) + ")")
plt.xlabel("Glacier mean altitude (m)")
plt.ylabel("Average SMB (1984-2014)")
#axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()
nfigure = nfigure+1

## Mean glacier slope vs SMB
plt.figure(nfigure)

x_slope = slope20_glaciers[:,1]
t_slope, p_slope = stats.ttest_rel(x_slope, avg_smb, nan_policy='omit')
plt.scatter(x_slope[idx_ecrins], avg_smb[idx_ecrins], marker="2", s=3, label='Ecrins')
plt.scatter(x_slope[idx_vanoise], avg_smb[idx_vanoise], marker="s", s=3, label='Vanoise')
plt.scatter(x_slope[idx_montblanc], avg_smb[idx_montblanc], marker="*", s=3, label='Mont Blanc')
p_ga3 = poly.Polynomial.fit(x_slope, avg_smb, 1)
x_p_ga3 = np.asarray(*p_ga3.linspace(n=x_slope.size)[:1]).flatten()
y_p_ga3 = np.asarray(*p_ga3.linspace(n=avg_smb.size)[1:]).flatten()
plt.plot(x_p_ga3, y_p_ga3)

ga3_idx = np.argsort(x_slope)
r_ga3 = np.corrcoef(avg_smb[ga3_idx], y_p_ga3)
r2_ga3 = r_ga3[1][0]**2

popt_slope, pcov_slope = curve_fit(fn1, x_slope, avg_smb)
perr_slope = np.sqrt(np.diag(pcov_slope))
nstd = 1 # 5 sigma intervals
popt_up_slope = popt_slope + nstd*perr_slope
popt_dw_slope = popt_slope - nstd*perr_slope
x_slope_sort = np.sort(x_slope)
fit_slope = fn1(x_slope_sort, *popt_slope)
fit_up_slope = fn1(x_slope_sort, *popt_up_slope)
fit_dw_slope = fn1(x_slope_sort, *popt_dw_slope)

#errorbar(x_mean, avg_sm)
plt.plot(x_slope_sort, fit_slope, 'r', lw=2, label="Best fit curve")
#plt.plot(x_mean, fit_up_mean, 'b', lw=2, label="Best fit curve")
#plt.plot(x_mean, fit_dw_mean, 'g', lw=2, label="Best fit curve")
#plt.fill_between(x_mean_sort, fit_up_mean, fit_dw_mean, alpha=.25, label="5-sigma interval")

plt.title("SMB vs mean glacier slope (r2 = " + str(r2_ga3) + " / p = " + str('{:.3g}'.format(p_slope)) + ")")
plt.xlabel("Glacier mean slope (deg)")
plt.ylabel("Average SMB (1984-2014)")
#axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()
nfigure = nfigure+1

# Max glacier altitude vs SMB
plt.figure(nfigure)
plt.scatter(glims_glaciers['MAX_Pixel'], avg_smb, s=3)
p_mga = poly.Polynomial.fit(glims_glaciers['MAX_Pixel'], avg_smb, 1)
x_p_mga = np.asarray(*p_mga.linspace(n=glims_glaciers['MAX_Pixel'].size)[:1]).flatten()
y_p_mga = np.asarray(*p_mga.linspace(n=avg_smb.size)[1:]).flatten()
plt.plot(x_p_mga, y_p_mga)

mga_idx = np.argsort(glims_glaciers['MAX_Pixel'])
r_mga = np.corrcoef(avg_smb[mga_idx], y_p_mga)
r2_mga = r_mga[1][0]**2

plt.title("SMB vs max glacier altitude (r2 = " + str(r2_mga) + ")")
plt.xlabel("Glacier max altitude (m)")
plt.ylabel("Average SMB (1984-2014)")
axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()
nfigure = nfigure+1

# Min glacier altitude vs SMB
plt.figure(nfigure)
plt.scatter(glims_glaciers['MIN_Pixel'], avg_smb, s=3)
p_mnga = poly.Polynomial.fit(glims_glaciers['MIN_Pixel'], avg_smb, 1)
x_p_mnga = np.asarray(*p_mnga.linspace(n=glims_glaciers['MIN_Pixel'].size)[:1]).flatten()
y_p_mnga = np.asarray(*p_mnga.linspace(n=avg_smb.size)[1:]).flatten()
plt.plot(x_p_mnga, y_p_mnga)

mnga_idx = np.argsort(glims_glaciers['MIN_Pixel'])
r_mnga = np.corrcoef(avg_smb[mnga_idx], y_p_mnga)
r2_mnga = r_mnga[1][0]**2


plt.title("SMB vs min glacier altitude (r2 = " + str(r2_mnga) + ")")
plt.xlabel("Glacier min altitude (m)")
plt.ylabel("Average SMB (1984-2014)")
axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()
nfigure = nfigure+1

# Glacier surface area vs SMB
plt.figure(nfigure)
plt.scatter(glims_glaciers['Area'], avg_smb, s=3)
p_gs = poly.Polynomial.fit(glims_glaciers['Area'], avg_smb, 1)
x_p_gs = np.asarray(*p_gs.linspace(n=glims_glaciers['Area'].size)[:1]).flatten()
y_p_gs = np.asarray(*p_gs.linspace(n=avg_smb.size)[1:]).flatten()
plt.plot(x_p_gs, y_p_gs)

gs_idx = np.argsort(glims_glaciers['Area'])
r_gs = np.corrcoef(avg_smb[gs_idx], y_p_gs)
r2_gs = r_gs[1][0]**2

plt.title("SMB vs Glacier surface area (r2 = " + str(r2_gs) + ")")
plt.xlabel("Glacier surface area (km2)")
plt.ylabel("Average SMB (1984-2014)")
axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()
nfigure = nfigure+1

# Glacier latitude vs SMB
plt.figure(nfigure)
plt.scatter(glims_glaciers['y_coord'], avg_smb, s=3)
p_gl = poly.Polynomial.fit(glims_glaciers['y_coord'], avg_smb, 1)
x_p_gl = np.asarray(*p_gl.linspace(n=glims_glaciers['y_coord'].size)[:1]).flatten()
y_p_gl = np.asarray(*p_gl.linspace(n=avg_smb.size)[1:]).flatten()
plt.plot(x_p_gl, y_p_gl)

gl_idx = np.argsort(glims_glaciers['y_coord'])
r_gl = np.corrcoef(avg_smb[gl_idx], y_p_gl)
r2_gl = r_gl[1][0]**2

plt.title("SMB vs Glacier latitude (r2 = " + str(r2_gl) + ")")
plt.xlabel("Glacier latitude")
plt.ylabel("Average SMB (1984-2014)")
axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()
nfigure = nfigure+1

# Glacier length vs SMB
plt.figure(nfigure)
plt.scatter(glims_glaciers['Length'], avg_smb, s=3)
p_gle = poly.Polynomial.fit(glims_glaciers['Length'], avg_smb, 1)
x_p_gle = np.asarray(*p_gle.linspace(n=glims_glaciers['Length'].size)[:1]).flatten()
y_p_gle = np.asarray(*p_gle.linspace(n=avg_smb.size)[1:]).flatten()
plt.plot(x_p_gle, y_p_gle)

gle_idx = np.argsort(glims_glaciers['Length'])
r_gle = np.corrcoef(avg_smb[gle_idx], y_p_gle)
r2_gle = r_gle[1][0]**2

plt.title("SMB vs Glacier length (r2 = " + str(r2_gle) + ")")
plt.xlabel("Glacier length (m)")
plt.ylabel("Average SMB (1984-2014)")
axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()
nfigure = nfigure+1

# Glacier altitude span vs SMB
plt.figure(nfigure)
alt_diff = glims_glaciers['MAX_Pixel'] - glims_glaciers['MIN_Pixel']
plt.scatter(alt_diff, avg_smb, s=3)
p_gas = poly.Polynomial.fit(alt_diff, avg_smb, 1)
x_p_gas = np.asarray(*p_gas.linspace(n=alt_diff.size)[:1]).flatten()
y_p_gas = np.asarray(*p_gas.linspace(n=avg_smb.size)[1:]).flatten()
plt.plot(x_p_gas, y_p_gas)

gas_idx = np.argsort(alt_diff)
r_gas = np.corrcoef(avg_smb[gas_idx], y_p_gas)
r2_gas = r_gas[1][0]**2

plt.title("SMB vs Glacier altitude span (r2 = " + str(r2_gas) + ")")
plt.xlabel("Glacier altitude span (m)")
plt.ylabel("Average SMB (1984-2014)")
axes = plt.gca()
#axes.set_xlim([0, 500])
#axes.set_ylim([0, 500])
plt.legend(loc=2)
plt.show()
nfigure = nfigure+1


######################################################################

# We store the optimal parameters for the SMB altitude adjustment
with open(path_smb+'smb_function\\popt_SMB_adjustment.txt', 'wb') as popt_f:
    np.save(popt_f, popt_altadj)
    