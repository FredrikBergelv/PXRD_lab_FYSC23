import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
            print(f'\\item Peak {i + 1}: Amplitude = \\SI{{{fit_amplitude:.2f}}}{{units}} $\\pm$ \\SI{{{np.sqrt(pcov[0, 0]):.2f}}}{{units}}, '
                  f'Mean = \\SI{{{fit_mean:.5f}}}{{radians}} $\\pm$ \\SI{{{np.sqrt(pcov[1, 1]):.5f}}}{{radians}}, '
                  f'Sigma = \\SI{{{fit_sigma:.5f}}}{{radians}} $\\pm$ \\SI{{{np.sqrt(pcov[2, 2]):.5f}}}{{radians}}, '
                  f'FWHM = \\SI{{{fwhm:.5f}}}{{radians}} $\\pm$ \\SI{{{fwhm_uncertainty:.5f}}}{{radians}}')
    
        except RuntimeError:
            print(f"  Peak {i + 1}: Could not fit peak in range {start:.4f} - {stop:.4f}")
    
    print(f'\\end{{enumerate}}')  # End the LaTeX enumerate environment for fitting results
    print(f'\\end{{enumerate}}')  # End the main LaTeX enumerate environment
    plt.legend()  # Show legend including Gaussian fits
    plt.savefig(f'PXRD_report/Figures/gaussian_{name}.pdf')
    plt.show()  # Display the plot

plt.close('all')
