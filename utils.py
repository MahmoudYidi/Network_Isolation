import spectral
from spectral import imshow, get_rgb
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import pandas as pd
import dask.array as da
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from skimage import exposure


def load_envi_hsi_by_wavelength(filepath, start_wavelength, end_wavelength):
    """
    Load an ENVI hyperspectral image (HSI) from a specific range of wavelengths.
    
    Parameters:
        filepath (str): Path to the ENVI HSI file.
        start_wavelength (float): Starting wavelength (in nanometers).
        end_wavelength (float): Ending wavelength (in nanometers).
    
    Returns:
        numpy.ndarray: HSI data cube containing the specified range of wavelengths.
    """
    # Open the HSI file
    hsi_data = spectral.envi.open(filepath)
    
    # Get wavelength information
    wavelengths = hsi_data.bands.centers
    
    # Find bands within the specified range of wavelengths
    selected_bands = [i for i, w in enumerate(wavelengths) if start_wavelength <= w <= end_wavelength]
    
    if not selected_bands:
        raise ValueError("No bands found within the specified range of wavelengths.")
    
    # Read the selected bands
    hsi_subset = hsi_data.read_bands(selected_bands)
    
    selected_wavelengths = [wavelengths[i] for i in selected_bands]

    return hsi_subset, selected_wavelengths

def load_envi_RGB_wave(filepath, wavelengths):
    """
    Load an ENVI hyperspectral image (HSI) for specific wavelengths.
    
    Parameters:
        filepath (str): Path to the ENVI HSI file.
        wavelengths (list of float): List of wavelengths (in nanometers) to load.
    
    Returns:
        numpy.ndarray: HSI data cube containing the specified wavelengths.
    """
    # Open the HSI file
    hsi_data = spectral.envi.open(filepath)
    
    # Get wavelength information
    all_wavelengths = hsi_data.bands.centers
    
    # Find bands corresponding to the selected wavelengths
    selected_bands = [i for i, w in enumerate(all_wavelengths) if w in wavelengths]
    
    if not selected_bands:
        raise ValueError("No bands found for the specified wavelengths.")
    
    # Read the selected bands
    hsi_subset = hsi_data.read_bands(selected_bands)
    
    return hsi_subset


def load_envi_hsi_2D(filepath, target_wavelength):
    """
    Load an ENVI hyperspectral image (HSI) at a specific wavelength.
    
    Parameters:
        filepath (str): Path to the ENVI HSI file.
        target_wavelength (float): Target wavelength (in nanometers).
    
    Returns:
        numpy.ndarray: HSI 2D image at the specified wavelength.
    """
    # Open the HSI file
    hsi_data = spectral.envi.open(filepath)
    
    # Get wavelength information
    wavelengths = np.array(hsi_data.bands.centers)
    
    # Find the index of the band closest to the target wavelength
    closest_band_index = np.argmin(np.abs(wavelengths - target_wavelength))
    
    # Read the selected band
    hsi_image = hsi_data.read_band(closest_band_index)
    
    return hsi_image

def data_correction(raw_data, dark_data, white_data):
    """"
    Performing Calibration of HSI data
     
    """
    
    corrected_data = np.divide(
            np.subtract(raw_data, dark_data),
            np.subtract(white_data, dark_data))
    return corrected_data

def get_band_index(wavelengths, target_wavelength):
    """
    Get the index position of a particular wavelength in the list of wavelengths.
    
    Parameters:
        wavelengths (list): List of wavelengths.
        target_wavelength (float): Wavelength to search for.
    
    Returns:
        int: Index position of the target wavelength in the list of wavelengths, or -1 if not found.
    """
    try:
        index = wavelengths.index(target_wavelength)
        return index
    except ValueError:
        return -1


def get_right_clicks(image):
    """
    Get right-click events and store the pixel locations of the clicks.

    Args:
    - image: The input image (numpy array).

    Returns:
    - List of pixel locations of the right-click events.
    """

    clicks = []  # Initialize clicks list to store pixel locations

    def mouse_callback(event, x, y, flags, params):
        """
        Callback function to handle mouse events.
        """
        if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
            clicks.append((x, y))  # Store the pixel location of the click

    # Create a window and set the mouse callback function
    cv2.namedWindow('Click the points')
    cv2.setMouseCallback('Click the points', mouse_callback)

    # Display the image and wait for right-click events
    while True:
        cv2.imshow('Click the points', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit capturing clicks
            break

    # Destroy the window and return the list of clicks
    cv2.destroyAllWindows()
    return clicks




def extract_rois(image, clicked_points, box_size):
    """
    Extract ROIs (Regions of Interest) from an image based on clicked points
    and assign a bounding box of specific size for each click.

    Args:
    - image: The input image (numpy array).
    - clicked_points: List of tuples containing (x, y) coordinates of clicked points.
    - box_size: Tuple containing (width, height) of the bounding box.

    Returns:
    - List of extracted ROIs.
    """

    rois = []

    # Iterate over clicked points
    for point in clicked_points:
        x, y = point

        # Calculate bounding box coordinates
        x1 = max(0, x - box_size[0] // 2)
        y1 = max(0, y - box_size[1] // 2)
        x2 = min(image.shape[1], x1 + box_size[0])
        y2 = min(image.shape[0], y1 + box_size[1])

        # Extract ROI
        roi = image[y1:y2, x1:x2]

        rois.append(roi)

    return rois

def calculate_average_reflectance(roi):
    """
    Calculate the average reflectance for an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - The average reflectance for the ROI.
    """
    # Assuming each channel represents reflectance at a different wavelength,
    # calculate the average pixel value across all channels
    average_reflectance = np.mean(roi, axis=(0, 1))
    return average_reflectance


def plot_reflectance_data(rois, wavelengths, output_filename='reflectance_plot.png'):
    """
    Plot each ROI's average reflectance versus the wavelength along with the reflectance data for reference.

    Args:
    - rois: List of ROIs.
    - wavelengths: Array of wavelengths.
    - reflectance_data: Reflectance data for reference.
    - output_filename: Output filename to save the plot as an image file. Default is 'reflectance_plot.png'.
    """

    # Calculate average reflectance for each ROI
    average_reflectances = [calculate_average_reflectance(roi) for roi in rois]

    # Plot each ROI's average reflectance versus the wavelength
    for i, average_reflectance in enumerate(average_reflectances):
        plt.plot(wavelengths, average_reflectance, label=f'ROI {i+1}')



    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Average Reflectance vs. Wavelength')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(output_filename)

    # Display a message indicating where the plot is saved
    print(f"Plot saved as '{output_filename}'")

def calculate_mean(roi):
    """
    Calculate the mean of an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - The mean of the ROI.
    """
    return np.mean(roi)

def calculate_standard_deviation(roi):
    """
    Calculate the standard deviation of an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - The standard deviation of the ROI.
    """
    return np.std(roi)

def calculate_variance(roi):
    """
    Calculate the variance of an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - The variance of the ROI.
    """
    return np.var(roi)

def calculate_statistics_for_rois(rois):
    """
    Calculate mean, standard deviation, and variance for each ROI.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).

    Returns:
    - List of tuples (mean, standard deviation, variance) for each ROI.
    """
    statistics = []
    for roi in rois:
        mean = calculate_mean(roi)
        std_deviation = calculate_standard_deviation(roi)
        variance = calculate_variance(roi)
        statistics.append((mean, std_deviation, variance))
    
    
    return statistics

def plot_statistics_for_rois(rois, output_filename='statistics_plot.png'):
    """
    Plot and save the statistics (mean, standard deviation, variance) for each ROI.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).
    - output_filename: Output filename to save the plot as an image file. Default is 'statistics_plot.png'.
    """
    # Calculate statistics for each ROI
    roi_statistics = []
    for roi in rois:
        mean = calculate_mean(roi)
        std_deviation = calculate_standard_deviation(roi)
        variance = calculate_variance(roi)
        roi_statistics.append((mean, std_deviation, variance))

    # Extract statistics for plotting
    means = [stat[0] for stat in roi_statistics]
    std_deviations = [stat[1] for stat in roi_statistics]
    variances = [stat[2] for stat in roi_statistics]

    # Plotting
    labels = [f"ROI {i+1}" for i in range(len(rois))]
    x = np.arange(len(rois))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, means, width, label='Mean')
    ax.bar(x, std_deviations, width, label='Standard Deviation')
    ax.bar(x + width, variances, width, label='Variance')

    ax.set_ylabel('Values')
    ax.set_title('Statistics Comparison Across ROIs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(output_filename)

    # Display a message indicating where the plot is saved
    print(f"Plot saved as '{output_filename}'")
    
def envi_loading(raw,dark,white,start_wl, end_wl):
    start_wl=float(start_wl)
    end_wl=float(end_wl)
    hsi_data_raw, bandss = load_envi_hsi_by_wavelength(raw, start_wl, end_wl)
    print("Done loading raw")
    hsi_data_white, _ = load_envi_hsi_by_wavelength(white, start_wl, end_wl)
    print("Done loading white")
    hsi_data_dark, _ = load_envi_hsi_by_wavelength(dark, start_wl, end_wl)
    print("Done loading dark")
    
    return hsi_data_raw, hsi_data_dark, hsi_data_white, bandss

def calculate_mean_per(roi):
    """
    Calculate the mean of each band in an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - A list containing the mean of each band in the ROI.
    """
    if len(roi.shape) != 3:
        raise ValueError("ROI must be a 3-dimensional numpy array")

    num_bands = roi.shape[2]
    means = []
    for i in range(num_bands):
        band_mean = np.mean(roi[:,:,i])
        means.append(band_mean)
    return means

def calculate_standard_deviation_per(roi):
    """
    Calculate the standard deviation of each band in an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - A list containing the standard deviation of each band in the ROI.
    """
    if len(roi.shape) != 3:
        raise ValueError("ROI must be a 3-dimensional numpy array")

    num_bands = roi.shape[2]
    std_deviations = []
    for i in range(num_bands):
        band_std_deviation = np.std(roi[:,:,i])
        std_deviations.append(band_std_deviation)
    return std_deviations

def calculate_variance_per(roi):
    """
    Calculate the variance of each band in an ROI.

    Args:
    - roi: The ROI (numpy array).

    Returns:
    - A list containing the variance of each band in the ROI.
    """
    if len(roi.shape) != 3:
        raise ValueError("ROI must be a 3-dimensional numpy array")

    num_bands = roi.shape[2]
    variances = []
    for i in range(num_bands):
        band_variance = np.var(roi[:,:,i])
        variances.append(band_variance)
    return variances

def calculate_statistics_for_rois_per(rois):
    """
    Calculate mean, standard deviation, and variance for each band of each ROI.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).

    Returns:
    - List of lists, where each inner list contains tuples (mean, standard deviation, variance) for each band of an ROI.
    """
    statistics = []
    for roi in rois:
        roi_stats = []
        for band in range(roi.shape[2]):
            mean = calculate_mean(roi[:, :, band])
            std_deviation = calculate_standard_deviation(roi[:, :, band])
            variance = calculate_variance(roi[:, :, band])
            roi_stats.append((mean, std_deviation, variance))
        statistics.append(roi_stats)
    return statistics

def plot_statistics_for_rois_per(rois, band_wavelengths, roi_index=None, output_filename='statistics_plot_per_band.png'):
    """
    Plot and save the statistics (mean, standard deviation, variance) for each band of each ROI.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).
    - band_wavelengths: List of wavelengths corresponding to each band.
    - roi_index: Index of the ROI to plot. If None, all ROIs will be plotted. Default is None.
    - output_filename: Output filename to save the plot as an image file. Default is 'statistics_plot_per_band.png'.
    """
    # Calculate statistics for selected ROI or all ROIs
    if roi_index is not None:
        rois_to_plot = [rois[roi_index]]
        roi_labels = [f"ROI {roi_index+1}"]
    else:
        rois_to_plot = rois
        roi_labels = [f"ROI {i+1}" for i in range(len(rois))]

    roi_statistics = calculate_statistics_for_rois_per(rois_to_plot)

    # Extract statistics for plotting
    num_bands = rois[0].shape[2]  # Assuming all ROIs have the same number of bands
    labels = [f"{roi_label} Band {j+1}\n({band_wavelengths[j]} nm)" for roi_label in roi_labels for j in range(num_bands)]
    means = [stat[0] for roi_stats in roi_statistics for stat in roi_stats]
    std_deviations = [stat[1] for roi_stats in roi_statistics for stat in roi_stats]
    variances = [stat[2] for roi_stats in roi_statistics for stat in roi_stats]

    # Plotting
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.bar(x - width, means, width, label='Mean')
    ax.bar(x, std_deviations, width, label='Standard Deviation')
    ax.bar(x + width, variances, width, label='Variance')

    ax.set_ylabel('Values')
    if roi_index is not None:
        ax.set_title(f'Statistics Comparison for {roi_labels[0]} Across Bands')
    else:
        ax.set_title('Statistics Comparison Across ROIs and Bands')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(output_filename)

    # Display a message indicating where the plot is saved
    print(f"Plot saved as '{output_filename}'")
    
def plot_statistics_metric(rois, band_wavelengths, roi_index=None):
    """
    Plot and save scatter plots for each metric (mean, standard deviation, variance) for each band of the ROI.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).
    - band_wavelengths: List of wavelengths corresponding to each band.
    - roi_index: Index of the ROI to plot. If None, all ROIs will be plotted. Default is None.
    """
    # Define colors for each ROI
    num_rois = len(rois)
    roi_colors = plt.cm.get_cmap('tab10', num_rois)

    # Calculate statistics for selected ROI or all ROIs
    if roi_index is not None:
        rois_to_plot = [rois[roi_index]]
        roi_labels = [f"ROI {roi_index+1}"]
    else:
        rois_to_plot = rois
        roi_labels = [f"ROI {i+1}" for i in range(len(rois))]

    roi_statistics = calculate_statistics_for_rois_per(rois_to_plot)

    # Extract statistics for plotting
    num_bands = rois[0].shape[2]  # Assuming all ROIs have the same number of bands

    for metric_index, metric_name in enumerate(["Mean", "Standard Deviation", "Variance"]):
        plt.figure(figsize=(10, 6))
        for roi_index, (roi_stats, roi_label) in enumerate(zip(roi_statistics, roi_labels)):
            metric_values = np.array([stat[metric_index] for stat in roi_stats])
            color = roi_colors(roi_index)
            plt.scatter(band_wavelengths, metric_values, label=roi_label, color=color)

        plt.xlabel('Wavelength (nm)')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Comparison Across Bands')

        # Display legend outside the plot area
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Save or show the plot
        if roi_index is not None:
            plt.savefig(f'{metric_name.lower()}_scatter_roi_{roi_index+1}.png', bbox_inches='tight')
        else:
            plt.savefig(f'{metric_name.lower()}_scatter_all_rois.png', bbox_inches='tight')
        plt.close()

def calculate_statistics_between_selected_rois(rois, roi_indices):
    """
    Calculate mean, standard deviation, and variance between selected ROIs across each wavelength for each band.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).
    - roi_indices: List of indices specifying the ROIs to include in the calculation.

    Returns:
    - Tuple containing mean, standard deviation, and variance across selected ROIs for each wavelength for each band.
    """
    # Convert selected ROIs into a single numpy array
    selected_rois = np.array([rois[i] for i in roi_indices])

    # Initialize lists to store statistics for each band
    means = []
    std_deviations = []
    variances = []

    # Iterate over each band
    for band in range(selected_rois.shape[2]):
        # Calculate mean, standard deviation, and variance across selected ROIs for the current band
        mean_between_selected_rois = np.mean(selected_rois[:, :, band])
        std_deviation_between_selected_rois = np.std(selected_rois[:, :, band])
        variance_between_selected_rois = np.var(selected_rois[:, :, band])

        # Append the statistics for the current band to the respective lists
        means.append(mean_between_selected_rois)
        std_deviations.append(std_deviation_between_selected_rois)
        variances.append(variance_between_selected_rois)

    return means, std_deviations, variances

def plot_statistics_between_selected_rois_scatter(statistics_sets, band_wavelengths, statistic_name):
    """
    Plot and save scatter plots for the given statistics between selected ROIs across each wavelength.

    Args:
    - statistics_sets: List of tuples, where each tuple contains the statistics for each pair of ROIs.
    - band_wavelengths: List of wavelengths corresponding to each band.
    - statistic_name: Name of the statistic being plotted (e.g., 'Mean', 'Standard Deviation', 'Variance').
    """
    num_bands = len(band_wavelengths)

    plt.figure(figsize=(10, 6))

    for i, statistics in enumerate(statistics_sets):
        for band in range(num_bands):
            plt.scatter(band_wavelengths[band], statistics[band], label=f'Pair {i+1}', alpha=0.7)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel(statistic_name)
    plt.title(f'{statistic_name} Comparison Between Selected ROIs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{statistic_name.lower()}_between_selected_rois_scatter.png')
    plt.close()

def compute_mean_and_variance_between_selected_rois(rois, roi_indices):
    """
    Compute the mean and variance between selected ROIs for each wavelength.

    Args:
    - rois: List of ROIs (each ROI is a numpy array with shape (40, 40, 195)).
    - roi_indices: List of indices specifying the ROIs to compare.

    Returns:
    - Tuple containing arrays containing the mean and variance between the selected ROIs for each wavelength.
    """
    # Convert the list of ROIs into a numpy array
    rois_array = np.array(rois)

    # Extract the pixel values for the selected ROIs
    selected_rois = rois_array[roi_indices]

    # Initialize empty arrays to store mean and variance values for each wavelength
    num_wavelengths = selected_rois[0].shape[2]
    mean_values = np.zeros(num_wavelengths)
    variance_values = np.zeros(num_wavelengths)

    # Iterate over each wavelength
    for i in range(num_wavelengths):
        # Extract pixel values for the current wavelength from both selected ROIs
        pixels_roi_1 = selected_rois[0][:, :, i]
        pixels_roi_2 = selected_rois[1][:, :, i]

        # Compute the mean between the selected ROIs for the current wavelength
        mean_values[i] = np.mean(pixels_roi_1 - pixels_roi_2)

        # Compute the variance between the selected ROIs for the current wavelength
        variance_values[i] = np.var(pixels_roi_1 - pixels_roi_2)

    return mean_values, variance_values

def plot_2_mean_and_variance(mean_values_1, mean_values_2, variance_values_1, variance_values_2, wavelengths):
    """
    Plot mean and variance values for two pairs of ROIs on separate graphs and save both.

    Args:
    - mean_values_1: Array containing the mean between the first pair of ROIs for each wavelength.
    - mean_values_2: Array containing the mean between the second pair of ROIs for each wavelength.
    - variance_values_1: Array containing the variance between the first pair of ROIs for each wavelength.
    - variance_values_2: Array containing the variance between the second pair of ROIs for each wavelength.
    - wavelengths: List of wavelengths corresponding to each band.
    """
    # Plot mean values
    plt.figure(figsize=(10, 6))
    plt.scatter(wavelengths, mean_values_1, label='Mean 1')
    plt.scatter(wavelengths, mean_values_2, label='Mean 2')
    plt.xlabel('Wavelength')
    plt.ylabel('Mean Value')
    plt.title('Mean Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mean_scatter_plot.png')
    plt.close()

    # Plot variance values
    plt.figure(figsize=(10, 6))
    plt.scatter(wavelengths, variance_values_1, label='Variance 1')
    plt.scatter(wavelengths, variance_values_2, label='Variance 2')
    plt.xlabel('Wavelength')
    plt.ylabel('Variance Value')
    plt.title('Variance Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('variance_scatter_plot.png')
    plt.close()


def plot_2_mean_and_cos(mean_values_1, mean_values_2, variance_values_1, variance_values_2, wavelengths):
    """
    Plot mean and variance values for two pairs of ROIs on separate graphs and save both.

    Args:
    - mean_values_1: Array containing the mean between the first pair of ROIs for each wavelength.
    - mean_values_2: Array containing the mean between the second pair of ROIs for each wavelength.
    - variance_values_1: Array containing the variance between the first pair of ROIs for each wavelength.
    - variance_values_2: Array containing the variance between the second pair of ROIs for each wavelength.
    - wavelengths: List of wavelengths corresponding to each band.
    """
    # Plot mean values
    plt.figure(figsize=(10, 6))
    plt.scatter(wavelengths, mean_values_1, label='Mean 1')
    plt.scatter(wavelengths, mean_values_2, label='Mean 2')
    plt.xlabel('Wavelength')
    plt.ylabel('Mean Value')
    plt.title('Mean Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('mean_scatter_plot.png')
    plt.close()

    # Plot variance values
    plt.figure(figsize=(10, 6))
    plt.scatter(wavelengths, variance_values_1, label='Variance 1')
    plt.scatter(wavelengths, variance_values_2, label='Variance 2')
    plt.xlabel('Wavelength')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity Scatter Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cos_scatter_plot.png')
    plt.close()
   
def compute_mse_cos(rois, roi_indices):
    """
    Compute mean squared error (MSE) and cosine similarity per wavelength between selected ROIs.

    Args:
    - rois: List of ROIs (each ROI is a numpy array with shape (40, 40, 195)).
    - roi_indices: List of indices specifying the ROIs to compare.

    Returns:
    - Two lists containing mean squared error and cosine similarity per wavelength.
    """
    # Select ROIs based on indices
    roi1 = rois[roi_indices[0]]
    roi2 = rois[roi_indices[1]]

    # Get the number of wavelengths
    num_wavelengths = roi1.shape[2]

    # Initialize lists to store metrics per wavelength
    mse_per_wavelength = []
    cos_similarity_per_wavelength = []

    # Iterate over each wavelength
    for i in range(num_wavelengths):
        # Extract pixel values for the current wavelength from both selected ROIs
        pixels_roi1 = roi1[:, :, i].reshape(-1)
        pixels_roi2 = roi2[:, :, i].reshape(-1)

        # Compute mean squared error
        mse_per_wavelength.append(mean_squared_error(pixels_roi1, pixels_roi2))

        # Compute cosine similarity
        cos_similarity_per_wavelength.append(1 - cosine(pixels_roi1, pixels_roi2))

    return mse_per_wavelength, cos_similarity_per_wavelength

def view_image_at_wavelength(image_data, wavelength_index, save_path):
    """
    View the image at a specific wavelength using imshow and save it as an image file.

    Args:
    - image_data: NumPy array containing hyperspectral image data with shape (height, width, num_wavelengths).
    - wavelength_index: Index of the wavelength to view.
    - save_path: File path to save the image.

    Returns:
    - None. Saves the image file.
    """
    # Extract data for the specified wavelength
    image_at_wavelength = image_data[:, :, wavelength_index]
    # Enhance contrast using histogram equalization
   #equalized_image = exposure.equalize_hist(image_at_wavelength)

    # Display the image using imshow
    plt.figure(figsize=(8, 6))
    plt.imshow(image_at_wavelength, cmap='gray')
    #plt.imshow(equalized_image, cmap='gray')
    plt.title(f"Image at Wavelength {wavelength_index}")
    plt.colorbar(label='Intensity')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')

    # Save the image as an image file
    plt.savefig(save_path)
    plt.close()
    
def view_image_at_wavelength_invert(image_data, wavelength_index, save_path):
    """
    View the image at a specific wavelength using imshow and save it as an image file.

    Args:
    - image_data: NumPy array containing hyperspectral image data with shape (height, width, num_wavelengths).
    - wavelength_index: Index of the wavelength to view.
    - save_path: File path to save the image.

    Returns:
    - None. Saves the image file.
    """
    # Extract data for the specified wavelength
    image_at_wavelength = image_data[:, :, wavelength_index]

    # Invert the intensity
    max_intensity = np.max(image_at_wavelength)
    inverted_image = max_intensity - image_at_wavelength

    # Display the image using imshow
    plt.figure(figsize=(8, 6))
    plt.imshow(inverted_image, cmap='gray')
    plt.title(f"Inverted Image at Wavelength {wavelength_index}")
    plt.colorbar(label='Intensity')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')

    # Save the image as an image file
    plt.savefig(save_path)
    plt.close()

def plot_3_cosine_similarities(cos_1, cos_2, cos_3, metric_name, wavelengths):
    """
    Plot variance values for three pairs of ROIs on the same graph and save.

    Args:
    - variance_values_1: Array containing the variance between the first pair of ROIs for each wavelength.
    - variance_values_2: Array containing the variance between the second pair of ROIs for each wavelength.
    - variance_values_3: Array containing the variance between the third pair of ROIs for each wavelength.
    - wavelengths: List of wavelengths corresponding to each band.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, cos_1, label='ROI 1')
    plt.plot(wavelengths, cos_2, label='ROI 2')
    plt.plot(wavelengths, cos_3, label='ROI 3')
    plt.xlabel('Wavelength')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison Between ROIs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{metric_name.lower()}_comparison.png')
    plt.close()
    

def plot_4_cosine_similarities(cosine_values_1, cosine_values_2, cosine_values_3, cosine_values_4, wavelengths):
    """
    Plot cosine similarity values for four pairs of ROIs on the same graph and save.

    Args:
    - cosine_values_1: Array containing the cosine similarity between the first pair of ROIs for each wavelength.
    - cosine_values_2: Array containing the cosine similarity between the second pair of ROIs for each wavelength.
    - cosine_values_3: Array containing the cosine similarity between the third pair of ROIs for each wavelength.
    - cosine_values_4: Array containing the cosine similarity between the fourth pair of ROIs for each wavelength.
    - wavelengths: List of wavelengths corresponding to each band.
    """
    # Plot cosine similarity values for all pairs of ROIs
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, cosine_values_1, label='Cosine 1')
    plt.plot(wavelengths, cosine_values_2, label='Cosine 2')
    plt.plot(wavelengths, cosine_values_3, label='Cosine 3')
    plt.plot(wavelengths, cosine_values_4, label='Cosine 4')
    plt.xlabel('Wavelength')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity Line Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cosine_similarity_line_plot.png')
    plt.close()
    
############################ Metric Stacks ###################################################
def compute_similarity_metrics_between_signatures(rois, roi_index1, roi_index2):
    """
    Compute multiple similarity metrics between two spectral signatures.

    Args:
    - signature1: Numpy array representing the spectral signature of the first ROI.
    - signature2: Numpy array representing the spectral signature of the second ROI.

    Returns:
    - Lists containing computed similarity metrics for each wavelength.
    """
    signature1 = rois[roi_index1]
    signature2 = rois[roi_index2]
    
    # Initialize lists to store similarity metrics for each wavelength
    mse_list = []
    rmse_list = []
    ssi_list = []
    cross_corr_list = []

    # Iterate over each wavelength
    for i in range(signature1.shape[2]):
        # Extract spectral values for the current wavelength
        spectral_values1 = signature1[:, :, i]
        spectral_values2 = signature2[:, :, i]

        # Compute Mean Squared Error (MSE)
        mse = np.mean((spectral_values1 - spectral_values2) ** 2)
        mse_list.append(mse)

        # Compute Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)
        
        data_range = spectral_values1.max() - spectral_values1.min()

        # Compute Structural Similarity Index (SSI)
        ssi = ssim(spectral_values1, spectral_values2, data_range=data_range)
        ssi_list.append(ssi)

        # Compute Cross-correlation
        cross_corr, _ = pearsonr(spectral_values1.flatten(), spectral_values2.flatten())
        cross_corr_list.append(cross_corr)

    return mse_list, rmse_list, ssi_list, cross_corr_list

def plot_two_similarity_metrics_vs_wavelength(metric_list_1, metric_list_2, metric_name, wavelengths):
    """
    Plot two sets of similarity metrics vs. wavelengths on the same graph and save.

    Args:
    - metric_list_1: List containing the first set of similarity metric values.
    - metric_list_2: List containing the second set of similarity metric values.
    - metric_name: Name of the similarity metric (e.g., 'MSE', 'RMSE', etc.).
    - wavelengths: List of wavelengths corresponding to each band.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, metric_list_1, label='ROI 1')
    plt.plot(wavelengths, metric_list_2, label='ROI 2')
    plt.xlabel('Wavelength')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison Between ROIs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{metric_name.lower()}_comparison.png')
    plt.close()

def cos_and_euc(rois, roi_indices):
    """
    Compute cosine similarity and Euclidean distance between selected ROIs for each wavelength.

    Args:
    - rois: List of ROIs (each ROI is a numpy array with shape (40, 40, 195)).
    - roi_indices: List of indices specifying the ROIs to compare.

    Returns:
    - Tuple containing the cosine similarity and Euclidean distance between the selected ROIs for each wavelength.
    """
    # Convert the list of ROIs into a numpy array
    rois_array = np.array(rois)

    # Extract the pixel values for the selected ROIs
    selected_rois = rois_array[roi_indices]

    # Initialize empty arrays to store similarity metrics for each wavelength
    num_wavelengths = selected_rois[0].shape[2]
    cosine_similarity = np.zeros(num_wavelengths)
    euclidean_distance = np.zeros(num_wavelengths)

    # Iterate over each wavelength
    for i in range(num_wavelengths):
        # Extract pixel values for the current wavelength from both selected ROIs
        pixels_roi_1 = selected_rois[0][:, :, i].flatten()
        pixels_roi_2 = selected_rois[1][:, :, i].flatten()

        # Compute cosine similarity between the selected ROIs for the current wavelength
        dot_product = np.dot(pixels_roi_1, pixels_roi_2)
        magnitude_roi_1 = np.linalg.norm(pixels_roi_1)
        magnitude_roi_2 = np.linalg.norm(pixels_roi_2)
        cosine_similarity[i] = dot_product / (magnitude_roi_1 * magnitude_roi_2)

        # Compute Euclidean distance between the selected ROIs for the current wavelength
        euclidean_distance[i] = np.linalg.norm(pixels_roi_1 - pixels_roi_2)

    return cosine_similarity, euclidean_distance

def compute_SAM(rois, roi_indices):
    """
    Compute the Spectral Angle Mapper (SAM) similarity score and angle between selected ROIs for each wavelength.

    Args:
    - rois: List of ROIs (each ROI is a numpy array with shape (40, 40, 195)).
    - roi_indices: List of indices specifying the ROIs to compare.

    Returns:
    - Tuple containing arrays of SAM similarity scores and angles between the selected ROIs for each wavelength.
    """
    # Convert the list of ROIs into a numpy array
    rois_array = np.array(rois)

    # Extract the pixel values for the selected ROIs
    selected_rois = rois_array[roi_indices]

    # Initialize empty arrays to store SAM similarity scores and angles for each wavelength
    num_wavelengths = selected_rois[0].shape[2]
    sam_scores = np.zeros(num_wavelengths)
    angles = np.zeros(num_wavelengths)

    # Iterate over each wavelength
    for i in range(num_wavelengths):
        # Extract pixel values for the current wavelength from both selected ROIs
        pixels_roi_1 = selected_rois[0][:, :, i].flatten()
        pixels_roi_2 = selected_rois[1][:, :, i].flatten()

        # Normalize the spectral vectors to unit length
        spectral_vector_1 = pixels_roi_1 / np.linalg.norm(pixels_roi_1)
        spectral_vector_2 = pixels_roi_2 / np.linalg.norm(pixels_roi_2)

        # Compute the dot product between the normalized spectral vectors
        dot_product = np.dot(spectral_vector_1, spectral_vector_2)

        # Compute the angle between the vectors using the dot product
        angle = np.arccos(dot_product)

        # Convert the angle to degrees
        angle_degrees = np.degrees(angle)

        # Convert the angle to a similarity score using the cosine function
        similarity_score = np.cos(angle)

        # Store the similarity score and angle for the current wavelength
        sam_scores[i] = similarity_score
        angles[i] = angle_degrees

    return sam_scores, angles

def z_score_normalize(data):
    mean = np.mean(data)
    std_deviation = np.std(data)
    normalized_data = (data - mean) / std_deviation
    return normalized_data

def standardize_per_wavelength(spectral_data):
    """
    Standardize spectral data per wavelength.

    Args:
    - spectral_data: Spectral data as a 2D numpy array (rows = samples, columns = wavelengths).

    Returns:
    - Standardized spectral data.
    """
    # Initialize an empty array to store standardized data
    standardized_data = np.zeros_like(spectral_data)

    # Iterate over each wavelength
    for i in range(spectral_data.shape[1]):
        # Extract values for the current wavelength
        values = spectral_data[:, i]

        # Calculate mean and standard deviation for the current wavelength
        mean = np.mean(values)
        std_dev = np.std(values)

        # Standardize values for the current wavelength
        standardized_values = (values - mean) / std_dev

        # Store standardized values in the result array
        standardized_data[:, i] = standardized_values

    return standardized_data

def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

def min_max_scaling_all(data_list):
    # Concatenate all arrays into a single array
    combined_data = np.concatenate(data_list)
    # Find the minimum and maximum values from the combined array
    min_val = np.min(combined_data)
    max_val = np.max(combined_data)
    # Scale each array based on the global minimum and maximum values
    scaled_data_list = [(data - min_val) / (max_val - min_val) for data in data_list]
    return scaled_data_list

def min_max_normalization_per_wavelength(data):
    """
    Perform min-max normalization per wavelength for spectral data.

    Args:
    - data: 3D numpy array representing spectral data with shape (num_samples, num_pixels_per_sample, num_wavelengths).

    Returns:
    - Normalized spectral data with the same shape as the input data.
    """
    # Initialize an array to store the normalized data
    normalized_data = np.zeros_like(data)

    # Iterate over each wavelength
    for i in range(data.shape[2]):
        # Extract spectral data for the current wavelength
        spectral_data_wavelength = data[:, :, i]

        # Calculate the minimum and maximum values for the current wavelength
        min_value = np.min(spectral_data_wavelength)
        max_value = np.max(spectral_data_wavelength)

        # Perform min-max normalization for the current wavelength
        normalized_data[:, :, i] = (spectral_data_wavelength - min_value) / (max_value - min_value)

    return normalized_data

def compute_SAM_all(rois, roi_indices):
    """
    Compute the Spectral Angle Mapper (SAM) similarity score and angle between selected ROIs for the entire dataset.

    Args:
    - rois: List of ROIs (each ROI is a numpy array with shape (40, 40, 195)).
    - roi_indices: List of indices specifying the ROIs to compare.

    Returns:
    - Tuple containing the SAM similarity score and angle between the selected ROIs.
    """
    # Convert the list of ROIs into a numpy array
    rois_array = np.array(rois)

    # Extract the pixel values for the selected ROIs
    selected_rois = rois_array[roi_indices]

    # Flatten the pixel values for both ROIs
    pixels_roi_1 = selected_rois[0].flatten()
    pixels_roi_2 = selected_rois[1].flatten()

    # Normalize the spectral vectors to unit length
    spectral_vector_1 = pixels_roi_1 / np.linalg.norm(pixels_roi_1)
    spectral_vector_2 = pixels_roi_2 / np.linalg.norm(pixels_roi_2)

    # Compute the dot product between the normalized spectral vectors
    dot_product = np.dot(spectral_vector_1, spectral_vector_2)

    # Compute the angle between the vectors using the dot product
    angle = np.arccos(dot_product)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle)

    # Convert the angle to a similarity score using the cosine function
    similarity_score = np.cos(angle)

    return similarity_score, angle_degrees

def plot_SAM_all(sam_scores, ROI, filename):
    """
    Plot the SAM similarity scores and angles.

    Args:
    - sam_scores: Array containing the SAM similarity scores.
    - angles: Array containing the angles between the spectral vectors.

    Returns:
    - None
    """
    # Plot SAM similarity scores
    plt.figure(figsize=(10, 6))
    plt.scatter(ROI[0], sam_scores[0], label='ROI 1', color='blue')
    plt.scatter(ROI[1],sam_scores[1], label='ROI 2', color='green')
    plt.scatter(ROI[2],sam_scores[2], label='ROI 3', color='red')
    plt.xlabel('Index')
    plt.ylabel('Similarity Score')
    plt.title('SAM Similarity Scores')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    
    
def subdivide_roi(roi, num_subdivisions):
    """
    Subdivide a single ROI into equal parts as determined by the user.

    Args:
    - roi: The ROI to subdivide (numpy array).
    - num_subdivisions: Number of subdivisions to create for the ROI.

    Returns:
    - List of subdivided ROIs.
    """
    subdivided_rois = []

    # Determine the size of each subdivision
    sub_width = roi.shape[1] // num_subdivisions
    sub_height = roi.shape[0] // num_subdivisions

    # Subdivide the ROI
    for i in range(num_subdivisions):
        for j in range(num_subdivisions):
            # Calculate bounding box coordinates for the current subdivision
            x1 = i * sub_width
            y1 = j * sub_height
            x2 = (i + 1) * sub_width
            y2 = (j + 1) * sub_height

            # Extract the current subdivision from the ROI
            subdivision = roi[y1:y2, x1:x2]

            # Append the subdivision to the list
            subdivided_rois.append(subdivision)

    return subdivided_rois

def select_and_subdivide_roi(rois, roi_index, num_subdivisions):
    """
    Select a specific ROI from the list of ROIs and subdivide it into equal parts.

    Args:
    - rois: List of ROIs (each ROI is a numpy array).
    - roi_index: Index of the ROI to subdivide.
    - num_subdivisions: Number of subdivisions to create for the ROI.

    Returns:
    - List of subdivided ROIs.
    """
    selected_roi = rois[roi_index]
    subdivided_roi = subdivide_roi(selected_roi, num_subdivisions)
    return subdivided_roi

def flat_field_correction(hsi_cube, flat_field_image):
    # Normalize flat field image
    normalized_flat_field = flat_field_image / np.mean(flat_field_image)
    
    # Apply flat field correction
    corrected_hsi_cube = hsi_cube / normalized_flat_field[:,:,np.newaxis]
    
    return corrected_hsi_cube

def histogram_equalization_hsi_cube(hsi_cube):
    # Initialize an empty array to store the equalized HSI cube
    equalized_hsi_cube = np.zeros_like(hsi_cube)

    # Calculate the minimum and maximum values from the input cube
    min_value = np.min(hsi_cube)
    max_value = np.max(hsi_cube)

    # Apply histogram equalization to each spectral band
    for i in range(hsi_cube.shape[2]):  # Iterate over spectral bands
        # Extract the spectral band
        spectral_band = hsi_cube[:,:,i]

        # Convert the spectral band to 8-bit unsigned integer format
        spectral_band_uint8 = cv2.normalize(spectral_band, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Perform histogram equalization
        equalized_spectral_band_uint8 = cv2.equalizeHist(spectral_band_uint8)

        # Rescale pixel values to the original data range
        equalized_spectral_band = cv2.normalize(equalized_spectral_band_uint8, None, min_value, max_value, cv2.NORM_MINMAX)

        # Store the equalized and scaled spectral band in the equalized HSI cube
        equalized_hsi_cube[:,:,i] = equalized_spectral_band.astype(hsi_cube.dtype)

    return equalized_hsi_cube

def gradient_based_correction(hsi_cube):
    # Initialize an empty array to store the corrected HSI cube
    corrected_hsi_cube = np.zeros_like(hsi_cube, dtype=np.float32)

    # Iterate over spectral bands (wavelengths)
    for i in range(hsi_cube.shape[2]):
        # Extract the spectral band
        spectral_band = hsi_cube[:,:,i]

        # Convert the spectral band to grayscale
        grayscale_image = spectral_band.astype(np.uint8)

        # Compute gradient magnitude using Sobel filter
        gradient_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # Avoid division by zero or invalid values
        gradient_magnitude_nonzero = np.where(gradient_magnitude != 0, gradient_magnitude, 1e-6)

        # Apply correction by dividing each pixel value by the corresponding gradient magnitude
        corrected_spectral_band = spectral_band / gradient_magnitude_nonzero

        # Store the corrected spectral band in the corrected HSI cube
        corrected_hsi_cube[:,:,i] = corrected_spectral_band

    return corrected_hsi_cube


