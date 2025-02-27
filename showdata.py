#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:26:24 2025

@author: fredrik
"""

import numpy as np
import matplotlib.pyplot as plt
import gaussian_fitting as ewald

samples = {"sample1": 'samples/Sample1.txt',
           "sample2":'samples/Sample2.txt'}




name = 'sample2'

data = np.loadtxt(samples[name])

# Extract The data
theta = (data[:, 0]/2) * np.pi/180   # First in radians
intensity = data[:, 1]  # Second in intesity


plt.figure(figsize=(7, 4))  # Optional: set figure size
plt.plot(theta, intensity, label=f'data {name}')

plt.title('Intensity vs $\Theta$')
if name == 'sample1':
    plt.ylim(1, 60) 
if name == 'sample2':
    plt.ylim(1, 13) 
plt.xlabel(r'$\theta$ (radians)')  # Label for x-axis using LaTeX
plt.ylabel(r'Intensity')  # Label for y-axis
plt.title(r'Intensity vs $\theta$ in Log Scale')  # Title of the plot using LaTeX
plt.grid(True)  # Show grid
plt.legend()  # Show legend
plt.savefig(f'PXRD_report/Figures/{name}.pdf')
plt.show()  # Display the plot


if name == 'sample1':
    peak1 = 0.1448, 0.1642
    peak2 = 0.1695, 0.1817
    peak3 = 0.2892, 0.3016
    peak4 = 0.3051, 0.3119
    peaks = [peak1, peak2, peak3, peak4]
    
if name == 'sample2':
    peak1 = 0.1722, 0.1803
    peak2 = 0.2485, 0.2546
    peak3 = 0.3059, 0.2129
    peaks = [peak1, peak2, peak3]

    
for (start, stop) in peaks:
    gaussian(theta, intensity, start, stop)
    
        
    