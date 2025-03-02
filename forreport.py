#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:02:40 2025

@author: fredrik
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import sin


Table = False

# Define the Gaussian function
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

samples = {"sample1": 'samples/Sample1.txt',
           "sample2": 'samples/Sample2.txt'}

names = 'sample1', 'sample2'

for name in names:
    
    data = np.loadtxt(samples[name])
    
    # Extract the data
    theta = (data[:, 0] / 2) * np.pi / 180  # Convert to radians
    intensity = data[:, 1]  # Second column for intensity
    
    plt.figure(figsize=(9, 6))  # Optional: set figure size
    plt.plot(theta, intensity, label=f'data {name}')
    
    # Adjust y-limits based on sample name
    if name == 'sample1':
        plt.ylim(1, 60) 
    if name == 'sample2':
        plt.ylim(1, 13) 
    
    plt.xlabel(r'$\theta$ (radians)')  # Label for x-axis using LaTeX
    plt.ylabel(r'Intensity')  # Label for y-axis
    plt.title(rf'Spectrum for {name}')  # Title of the plot using LaTeX
    plt.grid(True)  # Show grid
    
    # Define peaks to fit
    if name == 'sample1':
        peaks = [(0.1448, 0.1642, 44.27), (0.1695, 0.1817, 18.39), (0.24423, 0.25800, 12.39), (0.2892, 0.3016, 13.29), (0.3051, 0.3119, 4.82)]
    elif name == 'sample2':
        peaks = [(0.1747, 0.1802, 14.56), (0.2485, 0.2546, 4.45), (0.3059, 0.3129, 4.52)]
    
    if Table:
        print(f'\\begin{{enumerate}}')  # Start the LaTeX enumerate environment
        print(f'\\item For {name}:')
        print(f'\\begin{{enumerate}}')  # Start a new enumerate for fitting results
        
    # Loop through each peak for fitting
    for i, (start, stop, amplitude) in enumerate(peaks):
        # Select the region of interest
        mask = (theta >= start) & (theta <= stop)
        x_data = theta[mask]
        y_data = intensity[mask]
    
        # Check if x_data and y_data are empty
        if len(x_data) == 0 or len(y_data) == 0:
            print(f"  No data found for peak in range {start:.2f} - {stop:.2f}. Skipping this peak.")
            continue  # Skip to the next peak
    
        # Initial guess for the parameters: [amplitude, mean, sigma]
        initial_guess = [amplitude, np.mean(x_data), np.std(x_data)]
    
        # Perform the Gaussian fit
        try:
            popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
            fit_amplitude, fit_mean, fit_sigma = popt
    
            # Calculate FWHM and its uncertainty
            fwhm = 2 * np.sqrt(2 * np.log(2)) * fit_sigma
            fwhm_uncertainty = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(pcov[2, 2])  # Error in sigma
    
            # Plot the fit
            fit_y = gaussian(x_data, *popt)
            plt.plot(x_data, fit_y, label=f'Gaussian Fit {i + 1}')
            
            if name == 'sample2':
                # Add arrow and label for the peak
                plt.annotate(f'Peak {i + 1}', 
                             xy=(fit_mean, fit_amplitude), 
                             xytext=(fit_mean + 0.01, fit_amplitude + 1.2), 
                             arrowprops=dict(facecolor='black', arrowstyle='->'))
                
            if name == 'sample1':
                # Add arrow and label for the peak
                plt.annotate(f'Peak {i + 1}', 
                             xy=(fit_mean, fit_amplitude), 
                             xytext=(fit_mean + 0.01, fit_amplitude + 12), 
                             arrowprops=dict(facecolor='black', arrowstyle='->'))
    
            sf = 4  # significant figures
            
            # Print the fitting parameters in LaTeX format using \SI{}
            if Table:
                print(f'\\item Peak {i + 1}: Amplitude = \\SI{{{fit_amplitude:.2f}}}{{units}} $\\pm$ \\SI{{{np.sqrt(pcov[0, 0]):.2f}}}{{units}}, '
                  f'Mean = \\SI{{{fit_mean:.5f}}}{{radians}} $\\pm$ \\SI{{{np.sqrt(pcov[1, 1]):.5f}}}{{radians}}, '
                  f'Sigma = \\SI{{{fit_sigma:.5f}}}{{radians}} $\\pm$ \\SI{{{np.sqrt(pcov[2, 2]):.5f}}}{{radians}}, '
                  f'FWHM = \\SI{{{fwhm:.5f}}}{{radians}} $\\pm$ \\SI{{{fwhm_uncertainty:.5f}}}{{radians}}')
    
        except RuntimeError:
            print(f"  Peak {i + 1}: Could not fit peak in range {start:.4f} - {stop:.4f}")
    
    if Table: 
        print(f'\\end{{enumerate}}')  # End the LaTeX enumerate environment for fitting results
        print(f'\\end{{enumerate}}')  # End the main LaTeX enumerate environment
    plt.legend()  # Show legend including Gaussian fits
    plt.savefig(f'PXRD_report/Figures/gaussian_{name}.pdf')
    plt.show()  # Display the plot

if Table:
    plt.close('all')

##############################################################################

# Constants
ev = 1.60217663e-19  # Electronvolt to Joules conversion
h = 6.62607015e-34   # Planck's constant
c = 299792458        # Speed of light in m/s

# Energy and wavelength of X-ray
E_xray = 17.45e3 * ev  # Energy in Joules
lamb_xray = h * c / E_xray  # Wavelength in meters

lattice_constants = {
    "Si": 5.43102,
    "CdTe": 6.482,
    "KCl": 6.29,
    "Ag": 4.079,
    "Au": 4.065,
    "W": 3.155,
    "Fe": 2.856,
    "Mo": 3.142,
}

# Sample 1 mean angles with uncertainties
sample1_peak1 = ufloat(0.15362, 0.00010)
sample1_peak2 = ufloat(0.17731, 0.00019)
sample1_peak3 = ufloat(0.25180, 0.00018)
sample1_peak4 = ufloat(0.29569, 0.00015)
sample1_peak5 = ufloat(0.30900, 0.00015)

# Create a list of all sample 1 mean angles
sample1 = [sample1_peak1, sample1_peak2, sample1_peak3, sample1_peak4, sample1_peak5]

# Sample 2 mean angles with uncertainties
sample2_peak1 = ufloat(0.17746, 0.00012)
sample2_peak2 = ufloat(0.25186, 0.00017)
sample2_peak3 = ufloat(0.30942, 0.00014)

sample2 = [sample2_peak1, sample2_peak2, sample2_peak3]

def bragg(n, lamb, theta):
    # Calculate d-spacing while ensuring theta is treated properly
    d = n * lamb / (2 * sin(theta))  # Use sin from uncertainties.umath
    return d

n = 1  # Order of reflection

plane_111 = np.sqrt(3)

plane_110 = np.sqrt(2)


# Calculate d for Sample 1
for i, theta in enumerate(sample1):
    d = bragg(n, lamb_xray, theta)  # Calculate d-spacingplane_111
    a = d*plane_111
    a = (a * 1e10)  # Convert from meters to angstroms
    if np.isclose(a.nominal_value, lattice_constants['Au'], atol=0.05): 
        print(f'Sample 1: peak {i+1} and plane (111) a = {a} Å, where a_Au = {lattice_constants['Au']}')  # Print result


# Calculate d for Sample 1
for i, theta in enumerate(sample1):
    d = bragg(n, lamb_xray, theta)  # Calculate d-spacingplane_111
    a = d*plane_110
    a = (a * 1e10)  # Convert from meters to angstroms
    if np.isclose(a.nominal_value, lattice_constants['Fe'], atol=0.05): 
        print(f'Sample 2: peak {i+1} and plane (110) a = {a} Å, where a_Fe = {lattice_constants['Fe']}')  # Print result


