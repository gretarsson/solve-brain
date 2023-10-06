import numpy as np
#from EI_nott import solve_EI, f
from matplotlib import pyplot as plt
from numba import jit

# this is just a self-tutorial

def EI_ft(Z_E, t):
    # make sure input is numpy arrays
    f_E = np.array(f(Z_E))
    t = np.array(t)

    # fourier transform
    fourier = np.fft.fft(f_E)/len(f_E)
    fourier = fourier[range(int(len(f_E)/2))]

    # find frequencies and delete first output i.e the signal sum
    values = np.arange(int(len(f_E)/2))
    timeperiod = t[-1] - t[0]
    frequencies = values/timeperiod

    fourier = np.delete(fourier, 0)
    frequencies = np.delete(frequencies,0)

    # find top frequency
    max_ind = np.argmax(abs(fourier))
    top_ampl = abs(fourier[max_ind])
    top_freq = frequencies[max_ind]

    return top_ampl, top_freq


def EI_fourier(y, t, plot=False, bandpass=False):
    # make sure input is numpy arrays
    t = np.array(t)

    # fourier transform
    fourier = np.fft.fft(y)/len(y)
    fourier = fourier[range(int(len(y)/2))]

    # find frequencies and delete first output i.e the signal sum
    values = np.arange(int(len(y)/2))
    timeperiod = t[-1] - t[0]
    frequencies = values/timeperiod

    fourier = np.delete(fourier, 0)
    frequencies = np.delete(frequencies,0)

    # bandpass filter
    if bandpass:
        low = bandpass[0]
        top = bandpass[1]
        inds = np.array([i for i in range(len(frequencies)) if low < abs(frequencies[i]) < top])
        fourier_passed = fourier[inds]
        frequencies_passed = frequencies[inds]

    # find most prominent frequency
    max_ind = np.argmax(abs(fourier_passed))
    top_ampl = abs(fourier_passed[max_ind])
    top_freq = frequencies_passed[max_ind]

    # find relative power within band


    if plot:
        plt.plot(frequencies_passed, fourier_passed)
        plt.show()

    return frequencies_passed, fourier_passed

def power_spectrum(y, t, plot=False):
    y = np.array(y)
    y = np.add(y, -np.mean(y))  # remove 0 frequency
    ps = np.abs(np.fft.fft(y))**2
    ps = ps
    time_step = abs(t[1] - t[0])
    freqs = np.fft.fftfreq(y.size, time_step)
    idx = np.argsort(freqs)


    if plot:
        plt.plot(freqs[idx], ps[idx])
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlim((-100, 100))
        #plt.ylim((0, 0.1))
        plt.show()

    # find largest power
    argmax = np.argmax(ps[idx])


    return freqs[idx], ps[idx], ps[idx][argmax]

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

# modified the above function to return spectrogram peaks
def frequency_peaks(data, sf, band=None, window_sec=None, tol=10**-3, modified=False):
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
    tol : float
        tolerance for ignoring maximum peak and set frequency to zero

    Return
    ------
    peak : float
        Largest PSD peak in frequency.
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
            nperseg = (2/low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
    else:
        freqs, psd = periodogram(data, sf)

    # plot periodigram
    #plt.plot(freqs, psd)
    #plt.xlim([0, 14])
    #plt.show()

    # find peaks in psd
    if band.any():
        low, high = band
        filtered = np.array([i for i in range(len(freqs)) if (freqs[i] > low and freqs[i] < high)])
        psd = psd[filtered]
        freqs = freqs[filtered]

    max_peak = np.argmax(abs(psd))
    if max_peak is None or abs(psd[max_peak]) < tol:
        #freq_peak = 0
        freq_peak = float("NaN")
    else:
        freq_peak = freqs[max_peak]
    # we're done
    return freq_peak

#from scipy.signal import butter, lfilter
#
#def butter_bandpass(lowcut, highcut, fs, order=5):
#    nyq = 0.5 * fs
#    low = lowcut / nyq
#    high = highcut / nyq
#    b, a = butter(order, [low, high], btype='band')
#    return b, a
#
#def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#    print(f'lowcut = {lowcut}, highcut={highcut}, fs={fs}')
#    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
#    return y

from scipy.signal import butter, sosfilt, sosfreqz
def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y




# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#sol = solve_EI(t=200, plot=False, eta_E=5, a_IE=19.0, a_EI=20)
#Z_E = sol[0]
#t = sol[-1]
#print(t)
#
#f_E = np.array([f(Z_E[i]) for i in range(len(t)) if t[i] > 0])
#Z_E = np.array([Z_E[i] for i in range(len(t)) if t[i] > 0])
#t = np.array([t[i] for i in range(len(t)) if t[i] > 0])
##f_E = np.cos(2*np.pi*10*t)
#
#fig, ax = plt.subplots(2, 1)
#ax[0].plot(t, f_E)
#
#
#fourier = np.fft.fft(f_E)/len(f_E)
#fourier = fourier[range(int(len(f_E)/2))]
#
#values = np.arange(int(len(f_E)/2))
#timeperiod = t[-1] - t[0]
#frequencies = values/timeperiod
#
#fourier = np.delete(fourier, 0)
#frequencies = np.delete(frequencies,0)
#
#fourier = np.array([fourier[i] for i in range(len(frequencies)) if frequencies[i] < 2])
#frequencies = np.array([freq for freq in frequencies if freq < 2])
#
#print(EI_ft(Z_E, t))
#ax[1].plot(frequencies, abs(fourier))
#plt.show()
#
#
