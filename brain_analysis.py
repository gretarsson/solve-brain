import numpy as np
from scipy.signal import hilbert
from scipy.signal import butter, sosfilt, sosfreqz
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# -----------------------------------------
# Here we include functions for analyzing
# synthetic and experimental data used for
# whole-brain modeling
# -----------------------------------------

# ---------------------------------------------------
# Compute phase-lag index (PLI) of signal
#
#   input:
#           signal - array-like (signals, time)
# ---------------------------------------------------
def PLI(signal):
    """
    Computes the Phase-Lag Index (PLI) of a given signal.

    Parameters:
    signal (numpy.ndarray): The input signal (array-like).

    Returns:
    numpy.ndarray: The functional matrix representing PLI values between pairs of nodes.
    """
    # find phases of signal (over time) using the Hilbert transform
    hil = hilbert(signal)
    phases = np.angle(hil)

    # initialize functional matrix (lower triangular)
    N, T = signal.shape  
    F = np.zeros((N,N))

    # compute MPCs for each node pair
    for c in range(N):
        for r in range(c+1,N):
            diff_phase = phases[r,:] - phases[c,:]

            pli_i = np.sum(np.sign(np.sign(diff_phase))) / T
            pli_i = abs(pli_i)

            F[r,c] = pli_i
            F[c,r] = pli_i
    
    # we're done
    return F

def PLI_from_complex(signal):
    """
    Computes the Phase-Lag Index (PLI) of a given signal.

    Parameters:
    signal (numpy.ndarray): The input signal (array-like).

    Returns:
    numpy.ndarray: The functional matrix representing PLI values between pairs of nodes.
    """
    # find phases of signal (over time) using the Hilbert transform
    phases = np.angle(signal)

    # initialize functional matrix (lower triangular)
    N, T = signal.shape  
    F = np.zeros((N,N))

    # compute MPCs for each node pair
    for c in range(N):
        for r in range(c+1,N):
            diff_phase = phases[r,:] - phases[c,:]

            pli_i = np.sum(np.sign(np.sign(diff_phase))) / T
            pli_i = abs(pli_i)

            F[r,c] = pli_i
            F[c,r] = pli_i
    
    # we're done
    return F

def compute_phase_coherence_old(data):
    """
    Computes the phase-coherence order parameter of a 2D NumPy array of oscillators.

    Parameters:
    data (numpy.ndarray): The 2D NumPy array of oscillators, where each row is an oscillator and each column is a time domain.

    Returns:
    float: The phase-coherence order parameter of the oscillators.
    """
    # Compute the complex phases of the oscillators
    #complex_phases = np.exp(1j * data)
    complex_phases = np.exp(1j * data * 2*pi / np.amax(np.abs(data),axis=1))
    mean_phase = np.mean(complex_phases, axis=1)
    
    # Compute the magnitude of the mean phase
    coherence_parameter = np.abs(mean_phase)

    return coherence_parameter

def compute_phase_coherence(data):
    """
    Computes the phase-coherence order parameter of a 2D NumPy array of oscillators.

    Parameters:
    data (numpy.ndarray): The 2D NumPy array of oscillators, where each row is an oscillator and each column is a time domain.

    Returns:
    float: The phase-coherence order parameter of the oscillators.
    """
    # Compute the complex phases of the oscillators
    hil = hilbert(data)
    phases = np.angle(hil)
    complex_phases = np.exp(1j * phases)
    mean_phase = np.mean(complex_phases, axis=0)
    
    # Compute the magnitude of the mean phase
    coherence_parameter = np.abs(mean_phase)

    return coherence_parameter


def compute_phase_coherence_from_complex(data):
    """
    Computes the phase-coherence order parameter of a 2D NumPy array of oscillators.

    Parameters:
    data (numpy.ndarray): The 2D NumPy array of oscillators, where each row is an oscillator and each column is a time domain.

    Returns:
    float: The phase-coherence order parameter of the oscillators.
    """
    # Compute the complex phases of the oscillators
    phases = np.angle(data)
    complex_phases = np.exp(1j * phases)
    mean_phase = np.mean(complex_phases, axis=0)
    
    # Compute the magnitude of the mean phase
    coherence_parameter = np.abs(mean_phase)

    return coherence_parameter

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.

    Parameters:
    lowcut (float): Lower cutoff frequency.
    highcut (float): Upper cutoff frequency.
    fs (float): Sampling frequency.
    order (int): Order of the Butterworth filter.

    Returns:
    array: Second-order sections (sos) of the Butterworth filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the input data.

    Parameters:
    data (numpy.ndarray): Input data to filter.
    lowcut (float): Lower cutoff frequency.
    highcut (float): Upper cutoff frequency.
    fs (float): Sampling frequency.
    order (int): Order of the Butterworth filter.

    Returns:
    numpy.ndarray: Filtered output data.
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


# taken from https://raphaelvallat.com/bandpower.html
def bandpower(data, sf, band, window_sec=None, relative=False, modified=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch, periodogram
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Compute the (modified) periodogram 
    if modified:
        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
    else:
        freqs, psd = periodogram(data, sf)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        glob_idx = np.logical_and(freqs >= 0, freqs <= 40)
        bp /= simps(psd[glob_idx], dx=freq_res)
    return bp


# --------------------------------------
# plot average functional connectomes
# --------------------------------------
def plot_functional_connectomes(avg_F, t_stamps=False, bands=[], region_names=False, \
        colours=False, regions=False, coordinates=False, vmax=False, title=False, \
        edge_threshold='90.0%'):
    from itertools import chain
    from nilearn import plotting
    from matplotlib.colors import ListedColormap
    # check if we have a single connectome
    if len(avg_F.shape) == 2:
        avg_F = np.array([[[avg_F]]]) 

    # initialize
    B, I, L, N, N = avg_F.shape
    figs = []
    brain_figs = []
    
    # if colours, rearrange by node instead of region
    node_colours = []
    if colours is not False and regions is not False:
        for node in range(N):
            for r, region in enumerate(regions):
                if node in region:
                    node_colours.append(colours[r])
    else:
        node_colours = ['blue' for _ in range(N)]
        

    # if regions, reorganize matrices in the order of regions 2D list
    if regions is not False:
        node_map = list(chain(*regions))
        for b in range(B):
            for t in range(I):
                for l in range(L):
                    i = 0
                    for region in regions:
                        for node in region:
                            avg_F[b,t,l][[i,node], [i,node]] = avg_F[b,t,l][[node,i], [node,i]]
                            i += 1
                            if i == N:  # if node is in two regions, we need to break
                                break
    else:
        node_map = [n for n in range(N)]

    # rearrange region names after region
    if region_names is not False:
        new_region_names = []
        for n in range(N):
            new_region_names.append(region_names[node_map[n]])
    
    # iterate through each band and time point
    for b in range(B):
        if not vmax:
            vmax = np.amax(avg_F[b])
        for i in reversed(range(I)):
            # set plotting settings
            fig = plt.figure() 
            if title:
                plt.title(title)
            elif len(bands) and len(t_stamps):
                plt.title(f'band = {bands[b]}, t = {round(t_stamps[i],1)}')
            else:
                plt.title(f'b = {b}, i = {i}')

            # compute average functional matrix
            F = np.mean(avg_F[b,i], axis=0)

            # plot functional matrix as heatmap, either with regions names or without
            if region_names is not False:
                heatmap = sns.heatmap(F, xticklabels=new_region_names, yticklabels=new_region_names, \
                        vmin=0, vmax=vmax)
                heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 4)
                heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 4)
                if colours is not False:
                    for i, ticklabel in enumerate(heatmap.xaxis.get_majorticklabels()):
                        ticklabel.set_color(node_colours[node_map[i]])
                    for i, ticklabel in enumerate(heatmap.yaxis.get_majorticklabels()):
                        ticklabel.set_color(node_colours[node_map[i]])
            else:
                heatmap = sns.heatmap(F, vmin=0, vmax=vmax)

            # append figure to list of figures
            figs.append(fig)
            #plt.close()

	    # map functional connectome unto brain slices
            brain_map = None
            if coordinates is not False:
                # brain map settings
                node_size = 20
                cmap = ListedColormap(sns.color_palette("rocket"),1000)
                cmap = plt.get_cmap('magma')
                alpha_brain = 0.5
                alpha_edge = 0.5
                colorbar = True

                brain_map = plotting.plot_connectome(F, coordinates, edge_threshold=edge_threshold, \
                         node_color=node_colours, \
                        node_size=node_size, edge_cmap=cmap, edge_vmin=np.amin(F), edge_vmax=vmax, \
                        alpha=alpha_brain, colorbar=colorbar, edge_kwargs={'alpha':alpha_edge})
                #brain_map.close() 
            # append figure to list of figures
            brain_figs.append(brain_map)

    # we're done
    return figs, brain_figs
