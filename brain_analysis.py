import numpy as np
from scipy.signal import hilbert
from scipy.signal import butter, sosfilt, sosfreqz

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


def compute_phase_coherence(data):
    """
    Computes the phase-coherence order parameter of a 2D NumPy array of oscillators.

    Parameters:
    data (numpy.ndarray): The 2D NumPy array of oscillators, where each row is an oscillator and each column is a time domain.

    Returns:
    float: The phase-coherence order parameter of the oscillators.
    """
    # Compute the complex phases of the oscillators
    phases = np.exp(1j * data)

    # Compute the mean of the complex phases
    mean_phase = np.mean(phases, axis=0)

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

