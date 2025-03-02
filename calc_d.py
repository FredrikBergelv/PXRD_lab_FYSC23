#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:16:21 2025

@author: fredrik
"""

import numpy as np
from uncertainties import ufloat
from uncertainties.umath import sin

# Constants
ev = 1.60217663e-19  # Electronvolt to Joules conversion
h = 6.62607015e-34   # Planck's constant
c = 299792458        # Speed of light in m/s

# Energy and wavelength of X-ray
E_xray = 17.45e3 * ev  # Energy in Joules
lamb_xray = h * c / E_xray  # Wavelength in meters

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

plane_111 = np.sqrt(3), 

plane_110 = np.sqrt(2)

plane_113 = np.sqrt(1**2+1**2+3**2)

plane_133 = np.sqrt(1**2+3**2+3**2)

plane_200 = np.sqrt(2**2)

plane_220 = np.sqrt(2**2+2**2)


plane_222 = np.sqrt(2**2+2**2+2**2)


plane = plane_220

#%%
# Calculate d for Sample 1
print('Sample 1:')
for i, theta in enumerate(sample1):
    d = bragg(n, lamb_xray, theta)  # Calculate d-spacingplane_111
    a = d*plane
    aish = (a * 1e10)  # Convert from meters to angstroms
    print(f'Peak {i+1}: d = {aish} Å')  # Print result


#%%
# Calculate d for Sample 2
print('Sample 2:')
for i, theta in enumerate(sample2):
    d = bragg(n, lamb_xray, theta)  # Calculate d-spacing
    a = d*plane
    aish = (a * 1e10)  # Convert from meters to angstroms
    print(f'Peak {i+1}: d = {aish} Å')  # Print result


