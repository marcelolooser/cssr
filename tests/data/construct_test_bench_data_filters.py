"""
Module to construct test bench data for the filters module of the cssr package.

@author: marcelo looser
"""

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import cssr

# =============================================================================
# Preliminaries
# =============================================================================

dir_frames = "data_frames/"
dir_filters = "data_filters/"

# Set up configurations:
# ======================

# Filters:
# --------

# Signals:
# ........

set_signal_filters_gaussian = True
save_signal_filters_gaussian = True


# Frames:
# ........

set_frame_filters_gaussian = True
set_frame_filters_gaussian_overcomplete = True
save_frame_filters_gaussian = True


# =============================================================================
# Helper functions
# =============================================================================

def load_frame_data(dir_frames, frame_name):
    return np.load(dir_frames + frame_name + ".npz")

# =============================================================================
# Trial test data
# =============================================================================

x = np.arange(0, 20, 0.1)
dim = len(x)

n_peaks = 16 # np.mod(8*dim//10, n_peaks) == 0
amps = np.arange(0.01, 1, 1/n_peaks).reshape((-1,1))
peaks = np.arange(dim//10, 9*dim//10, 8*dim//10//n_peaks)

y_sparse = np.zeros((dim, 1))
y_sparse[peaks] = amps

# =============================================================================
# Start test data construction
# =============================================================================

cutoffs = [[1, None], [0.6, 1.2], [70, None]] # where 70 is 70° celsius

# Signal:
# =============================================================================


# Gaussian:
# ---------

if set_signal_filters_gaussian:

    data = load_frame_data(dir_frames, "gaussian")

    a0_gaussian2 = data["a0_gaussian2"]

    y_gaussian2 = a0_gaussian2.dot(y_sparse) # to be saved

    csFr_y2_gaussian = cssr.Filters(y_gaussian2, x, cutoffs[0][0], filter_signal=True)

    y_gaussian_heaviside2 = csFr_y2_gaussian.heaviside_lowpass_filter()
    y_gaussian_fir2 = csFr_y2_gaussian.fir_filter()

    csFr_y2_gaussian.cutoff = cutoffs[1]

    y_gaussian_fir5 = csFr_y2_gaussian.fir_filter(numtabs=5, pass_zero="bandstop")
    y_gaussian_butter2 = csFr_y2_gaussian.butter_filter(order=5, btype="bandstop")
    y_gaussian2_filtered1 = csFr_y2_gaussian.truncated_a0 # to be saved
    y_gaussian_filter2_record1 = csFr_y2_gaussian.filter_record(name_only=True) # to be saved

    csFr_y2_gaussian.reset()

    y_gaussian_butter5 = csFr_y2_gaussian.butter_filter()
    y_gaussian_instrumental2 = csFr_y2_gaussian.instrumental_lowpass_filter()

    csFr_y2_gaussian.cutoff = cutoffs[2][0]

    y_gaussian_thermal2 = csFr_y2_gaussian.thermal_lowpass_filter()
    y_gaussian2_filtered2 = csFr_y2_gaussian.truncated_a0 # to be saved
    y_gaussian_filter2_record2 = csFr_y2_gaussian.filter_record(name_only=True) # to be saved


if save_signal_filters_gaussian and set_signal_filters_gaussian:
    np.savez(dir_filters + "gaussian_signal",
                 cutoffs = cutoffs,
                 y_sparse = y_sparse,

                 y_gaussian2 = y_gaussian2,

                 y_gaussian2_filtered1 = y_gaussian2_filtered1,
                 y_gaussian_filter2_record1 = y_gaussian_filter2_record1,

                 y_gaussian2_filtered2 = y_gaussian2_filtered2,
                 y_gaussian_filter2_record2 = y_gaussian_filter2_record2
                 )


# Frames:
# =============================================================================


# Gaussian:
# ---------

# Dicionaries:
# ............

if set_frame_filters_gaussian:

    data = load_frame_data(dir_frames, "gaussian")

    a0_gaussian2 = data["a0_gaussian2"]

    csFr_a02_gaussian = cssr.Filters(a0_gaussian2, x, cutoffs[0][0])

    a0_gaussian_heaviside2 = csFr_a02_gaussian.heaviside_lowpass_filter()
    a0_gaussian_fir2 = csFr_a02_gaussian.fir_filter()

    csFr_a02_gaussian.cutoff = cutoffs[1]

    a0_gaussian_fir5 = csFr_a02_gaussian.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_gaussian_butter2 = csFr_a02_gaussian.butter_filter(order=5, btype="bandstop")
    a0_gaussian2_filtered1 = csFr_a02_gaussian.truncated_a0 # to be saved
    a0_gaussian_filter2_record1 = csFr_a02_gaussian.filter_record(name_only=True) # to be saved

    csFr_a02_gaussian.reset()

    a0_gaussian_butter5 = csFr_a02_gaussian.butter_filter()
    a0_gaussian_instrumental2 = csFr_a02_gaussian.instrumental_lowpass_filter()

    csFr_a02_gaussian.cutoff = cutoffs[2][0]

    a0_gaussian_thermal2 = csFr_a02_gaussian.thermal_lowpass_filter()
    a0_gaussian2_filtered2 = csFr_a02_gaussian.truncated_a0 # to be saved
    a0_gaussian_filter2_record2 = csFr_a02_gaussian.filter_record(name_only=True) # to be saved


# Overcomplete dicionaries:
# .........................

if set_frame_filters_gaussian_overcomplete:

    data = load_frame_data(dir_frames, "gaussian")

    a0_gaussian_overcomplete2 = data["a0_gaussian_overcomplete2"]

    csFr_a02_gaussian_overcomplete = cssr.Filters(a0_gaussian_overcomplete2, x, cutoffs[0][0])

    a0_gaussian_overcomplete_heaviside2 = csFr_a02_gaussian_overcomplete.heaviside_lowpass_filter()
    a0_gaussian_overcomplete_fir2 = csFr_a02_gaussian_overcomplete.fir_filter()

    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[1]

    a0_gaussian_overcomplete_fir5 = csFr_a02_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_gaussian_overcomplete_butter2 = csFr_a02_gaussian_overcomplete.butter_filter(order=5, btype="bandstop")
    a0_gaussian_overcomplete2_filtered1 = csFr_a02_gaussian_overcomplete.truncated_a0 # to be saved
    a0_gaussian_overcomplete_filter2_record1 = csFr_a02_gaussian_overcomplete.filter_record(name_only=True) # to be saved

    csFr_a02_gaussian_overcomplete.reset()

    a0_gaussian_overcomplete_butter5 = csFr_a02_gaussian_overcomplete.butter_filter()
    a0_gaussian_overcomplete_instrumental2 = csFr_a02_gaussian_overcomplete.instrumental_lowpass_filter()

    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[2][0]

    a0_gaussian_overcomplete_thermal2 = csFr_a02_gaussian_overcomplete.thermal_lowpass_filter()
    a0_gaussian_overcomplete2_filtered2 = csFr_a02_gaussian_overcomplete.truncated_a0 # to be saved
    a0_gaussian_overcomplete_filter2_record2 = csFr_a02_gaussian_overcomplete.filter_record(name_only=True) # to be saved


if save_frame_filters_gaussian and set_frame_filters_gaussian and set_frame_filters_gaussian_overcomplete:
    np.savez(dir_filters + "gaussian_frame",
                 cutoffs = cutoffs,

                 a0_gaussian2_filtered1 = a0_gaussian2_filtered1,
                 a0_gaussian_filter2_record1 = a0_gaussian_filter2_record1,

                 a0_gaussian2_filtered2 = a0_gaussian2_filtered2,
                 a0_gaussian_filter2_record2 = a0_gaussian_filter2_record2,


                 a0_gaussian_overcomplete2_filtered1 = a0_gaussian_overcomplete2_filtered1,
                 a0_gaussian_overcomplete_filter2_record1 = a0_gaussian_overcomplete_filter2_record1,

                 a0_gaussian_overcomplete2_filtered2 = a0_gaussian_overcomplete2_filtered2,
                 a0_gaussian_overcomplete_filter2_record2 = a0_gaussian_overcomplete_filter2_record2
                 )


# =============================================================================
