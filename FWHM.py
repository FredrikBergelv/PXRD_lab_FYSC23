#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 14:50:45 2025

@author: fredrik
"""

import numpy as np
from uncertainties import ufloat
from uncertainties.umath import cos

# Constants
ev = 1.60217663e-19  # Electronvolt to Joules conversion
h = 6.62607015e-34   # Planck's constant
c = 299792458        # Speed of light in m/s
k = 0.94             # Scherre's constant

# Energy and wavelength of X-ray
E_xray = 17.45e3 * ev  # Energy in Joules
lamb_xray = h * c / E_xray  # Wavelength in meters

def Scherre(k, lamb, FWHM, angle):
    return k*lamb / (FWHM*cos(angle))

# Sample 1 mean angles with uncertainties
sample1_peak1 = ufloat(0.15362, 0.00010)
sample1_peak2 = ufloat(0.17731, 0.00019)
sample1_peak3 = ufloat(0.25180, 0.00018)
sample1_peak4 = ufloat(0.29569, 0.00015)
sample1_peak5 = ufloat(0.30900, 0.00015)

# Create a list of all sample 1 mean angles
peaks1 = [sample1_peak1, sample1_peak2, sample1_peak3, sample1_peak4, sample1_peak5]

sample1_FWHM1 = ufloat(0.00388, 0.00023)
sample1_FWHM2 = ufloat(0.00535, 0.00046)
sample1_FWHM3 = ufloat(0.00550, 0.00043)
sample1_FWHM4 = ufloat(0.00530, 0.00035)
sample1_FWHM5 = ufloat(0.00616, 0.00049)

# Create a list of all sample 1 mean FWHM
peak1 = [[sample1_peak1, sample1_FWHM1], 
         [sample1_peak2, sample1_FWHM2], 
         [sample1_peak3, sample1_FWHM3], 
         [sample1_peak4, sample1_FWHM4], 
         [sample1_peak5, sample1_FWHM5]]

# Sample 2 mean angles with uncertainties
sample2_peak1 = ufloat(0.17746, 0.00012)
sample2_peak2 = ufloat(0.25186, 0.00017)
sample2_peak3 = ufloat(0.30942, 0.00014)


sample2_FWHM1 = ufloat(0.00404, 0.00034)
sample2_FWHM2 = ufloat(0.00895, 0.00092)
sample2_FWHM3 = ufloat(0.00768, 0.00056)

peak2 = [[sample2_peak1, sample2_FWHM1], 
         [sample2_peak2, sample2_FWHM2], 
         [sample2_peak3, sample2_FWHM3]]

#%%

t1_values = []

print('Sample 1:')
for angle, FWHM in peak1:
    t = Scherre(k, lamb_xray, FWHM, angle)
    t1_values.append(t)
    print(t)
    
print(f'mean particle size is {10**9*np.mean(t1_values)} nm')

#%%%

t2_values = []

print('Sample 2:')
for angle, FWHM in peak2:
    t = Scherre(k, lamb_xray, FWHM, angle)
    t2_values.append(t)
    print(t)
    
print(f'mean particle size is {10**9*np.mean(t2_values)} nm')
