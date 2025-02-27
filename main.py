#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:26:24 2025

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt
import gaussian_fitting as ewald

X = np.linspace(-10, 10, 1000)
A_true = 5       # Amplitude
mu_true = 2      # Mean
sigma_true = 2   # Standard deviation

Y = A_true * np.exp(-((X - mu_true) ** 2) / (2 * sigma_true ** 2))
noise = np.random.normal(0, 0.2, X.shape)
Y = Y + noise


plt.plot(X,Y)


gaussian = ewald.fit_gaussian(X, Y, 
               region_start=-2,  
               region_stop=7,
               scatter_corr=False)  # Disable scatter correction

# Extract parameters
mu = gaussian.mu
sigma = gaussian.sigma
fwhm = 2.355 * sigma

# Extract uncertainties from covariance matrix
mu_uncertainty = np.sqrt(gaussian.covar_matrix[1, 1])  # Variance of mu
sigma_uncertainty = np.sqrt(gaussian.covar_matrix[2, 2])  # Variance of sigma
fwhm_uncertainty = 2.355 * sigma_uncertainty






# Store results in a dictionary
gaussian_params = {
    "mu": f'{mu} +/- {mu_uncertainty}',
    "fwhm": f'{fwhm} +/- {fwhm_uncertainty}'}

# Print the results
print(gaussian_params)