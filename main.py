#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:26:24 2025

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt
import gaussian_fitting as ewald

sample1 = 'samples/Sample1.txt'
sample2 = 'samples/Sample1.txt'


data = np.loadtxt(sample1)

# Extract The data
theta = data[:, 0]/2  # First column
intensity = data[:, 1]  # Second column


plt.figure(figsize=(10, 6))  # Optional: set figure size
plt.plot(theta, intensity, label='Intensity vs Theta')

plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.xlabel('Theta')  # Label for x-axis
plt.ylabel('Intensity (log scale)')  # Label for y-axis
plt.title('Intensity vs Theta in Log Scale')  # Title of the plot
plt.grid(True)  # Show grid
plt.legend()  # Show legend
plt.show()  # Display the plot


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
    
    
    
    