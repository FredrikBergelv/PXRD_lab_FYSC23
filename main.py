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



def gaussian(x, y, region_start, region_stop):
    # Perform Gaussian fitting
    gaussian_fit = ewald.fit_gaussian(x, y, 
                                      region_start=region_start,  
                                      region_stop=region_stop,
                                      scatter_corr=False)  # Disable scatter correction
    
    # Extract parameters
    mu = gaussian_fit.mu
    A = gaussian_fit.A  # Corrected amplitude assignment
    sigma = gaussian_fit.sigma
    fwhm = 2.355 * sigma  # Calculate FWHM
    
    # Extract uncertainties from covariance matrix
    mu_uncertainty = np.sqrt(gaussian_fit.covar_matrix[1, 1])  # Variance of mu
    sigma_uncertainty = np.sqrt(gaussian_fit.covar_matrix[2, 2])  # Variance of sigma
    fwhm_uncertainty = 2.355 * sigma_uncertainty  # Propagate uncertainty for FWHM
    
    # Store results in a dictionary
    gaussian_params = {
        "mu": mu,
        "mu_uncertainty": mu_uncertainty,
        "fwhm": fwhm,
        "fwhm_uncertainty": fwhm_uncertainty
    }
    
    # Select region for plotting
    mask = (x >= region_start) & (x <= region_stop)
    xplot = x[mask]*1.1
    yplot = y[mask]*1.1
    
    # Plot data and Gaussian fit
    plt.figure(figsize=(5, 5))
    plt.plot(xplot, yplot, label='Data')

    # Compute Gaussian curve
    gaus = A * np.exp(-((xplot - mu) ** 2) / (2 * sigma ** 2))
    plt.plot(xplot, gaus, label='Gaussian Fit', color='red')

    plt.grid()
    plt.legend()
    plt.show()
    
    # Print the results
    print(f'mu = {mu:.4f} +/- {mu_uncertainty:.4f}, fwhm = {fwhm:.4f} +/- {fwhm_uncertainty:.4f}')
    
    return gaussian_params  # Return the dictionary
    
    
    
gaussian(X, Y, region_start=-2, region_stop=7)
    