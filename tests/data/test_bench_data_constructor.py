"""
Create and save test bench data for the cssr package.

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
dir_measurement_matrices = "data_measurement_matrices/"

# Set up configurations:
# ======================

# Frames:
# -------
set_frames_heaviside = True
set_frames_heaviside_overcomplete = True
save_frames_heaviside = True

set_frames_gaussian = True
set_frames_gaussian_overcomplete = True
save_frames_gaussian = True

set_frames_cauchy = True
set_frames_cauchy_overcomplete = True
save_frames_cauchy = True

set_frames_fourier = True
save_frames_fourier = True


# Filters:
# --------

# Signals:
# ........
set_signal_filters_heaviside = True
save_signal_filters_heaviside = True

set_signal_filters_gaussian = True
save_signal_filters_gaussian = True

set_signal_filters_cauchy = True
save_signal_filters_cauchy = True

set_signal_filters_fourier = True
save_signal_filters_fourier = True


# Frammes:
# ........
set_frame_filters_heaviside = True
set_frame_filters_heaviside_overcomplete = True
save_frame_filters_heaviside = True

set_frame_filters_gaussian = True
set_frame_filters_gaussian_overcomplete = True
save_frame_filters_gaussian = True

set_frame_filters_cauchy = True
set_frame_filters_cauchy_overcomplete = True
save_frame_filters_cauchy = True

set_frame_filters_fourier = True
save_frame_filters_fourier = True


# Measurement matrices:
# ---------------------
save_measurment_marices = True


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
# =============================================================================
# Frames
# =============================================================================

csF = cssr.Frames(x)

# Heaviside:
# ==========

bw_heaviside = [1, dim//2, dim - 1]
bw_heaviside_overcomplete = [[None, None], [dim//2 - 3, dim//2 + 3], [dim//2 - 3, dim//2 + 3]]
ss_heaviside_overcomplete = [None, None, 3]

if set_frames_heaviside:
    a0_heaviside1 = csF.heaviside(bw_heaviside[0])
    a0_heaviside2 = csF.heaviside(bw_heaviside[1])
    a0_heaviside3 = csF.heaviside(bw_heaviside[2])

if set_frames_heaviside_overcomplete:
    a0_heaviside_overcomplete1 = csF.heaviside_overcomplete(
        bw_heaviside_overcomplete[0][0], ss_heaviside_overcomplete[0]
        )
    a0_heaviside_overcomplete2 = csF.heaviside_overcomplete(
        bw_heaviside_overcomplete[1], ss_heaviside_overcomplete[1]
        )
    a0_heaviside_overcomplete3 = csF.heaviside_overcomplete(
        bw_heaviside_overcomplete[2], ss_heaviside_overcomplete[2]
        )

if save_frames_heaviside and set_frames_heaviside and set_frames_heaviside_overcomplete:
    np.savez(dir_frames + "heaviside",
             x = x,
             bw_heaviside = bw_heaviside,
             bw_heaviside_overcomplete = np.array(bw_heaviside_overcomplete, dtype=float),
             ss_heaviside_overcomplete = np.array(ss_heaviside_overcomplete, dtype=float),

             a0_heaviside1 = a0_heaviside1,
             a0_heaviside2 = a0_heaviside2,
             a0_heaviside3 = a0_heaviside3,

             a0_heaviside_overcomplete1 = a0_heaviside_overcomplete1,
             a0_heaviside_overcomplete2 = a0_heaviside_overcomplete2,
             a0_heaviside_overcomplete3 = a0_heaviside_overcomplete3
             )


# Gaussian:
# =========

bw_gaussian = [abs(x[1] - x[0]), abs(x[-1] - x[0])/2, abs(x[-1] - x[0]) - abs(x[1] - x[0])]
bw_gaussian_overcomplete = [[None, None], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])]]
ss_gaussian_overcomplete = [None, None, 3*abs(x[1] - x[0])]

if set_frames_gaussian:
    a0_gaussian1 = csF.gaussian(bw_gaussian[0])
    a0_gaussian2 = csF.gaussian(bw_gaussian[1])
    a0_gaussian3 = csF.gaussian(bw_gaussian[2])

if set_frames_gaussian_overcomplete:
    a0_gaussian_overcomplete1 = csF.gaussian_overcomplete(
        bw_gaussian_overcomplete[0][0], ss_gaussian_overcomplete[0]
        )
    a0_gaussian_overcomplete2 = csF.gaussian_overcomplete(
        bw_gaussian_overcomplete[1], ss_gaussian_overcomplete[1]
        )
    a0_gaussian_overcomplete3 = csF.gaussian_overcomplete(
        bw_gaussian_overcomplete[2], ss_gaussian_overcomplete[2]
        )

if save_frames_gaussian and set_frames_gaussian and set_frames_gaussian_overcomplete:
    np.savez(dir_frames + "gaussian",
             x = x,
             bw_gaussian = bw_gaussian,
             bw_gaussian_overcomplete = np.array(bw_gaussian_overcomplete, dtype=float),
             ss_gaussian_overcomplete = np.array(ss_gaussian_overcomplete, dtype=float),

             a0_gaussian1 = a0_gaussian1,
             a0_gaussian2 = a0_gaussian2,
             a0_gaussian3 = a0_gaussian3,

             a0_gaussian_overcomplete1 = a0_gaussian_overcomplete1,
             a0_gaussian_overcomplete2 = a0_gaussian_overcomplete2,
             a0_gaussian_overcomplete3 = a0_gaussian_overcomplete3
             )


# Cauchy:
# =======

bw_cauchy = [abs(x[1] - x[0]), abs(x[-1] - x[0])/2, abs(x[-1] - x[0]) - abs(x[1] - x[0])]
bw_cauchy_overcomplete = [[None, None], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])]]
ss_cauchy_overcomplete = [None, None, 3*abs(x[1] - x[0])]

if set_frames_cauchy:
    a0_cauchy1 = csF.cauchy(bw_cauchy[0])
    a0_cauchy2 = csF.cauchy(bw_cauchy[1])
    a0_cauchy3 = csF.cauchy(bw_cauchy[2])

if set_frames_cauchy_overcomplete:
    a0_cauchy_overcomplete1 = csF.cauchy_overcomplete(
        bw_cauchy_overcomplete[0][0], ss_cauchy_overcomplete[0]
        )
    a0_cauchy_overcomplete2 = csF.cauchy_overcomplete(
        bw_cauchy_overcomplete[1], ss_cauchy_overcomplete[1]
        )
    a0_cauchy_overcomplete3 = csF.cauchy_overcomplete(
        bw_cauchy_overcomplete[2], ss_cauchy_overcomplete[2]
        )

if save_frames_cauchy and set_frames_cauchy and set_frames_cauchy_overcomplete:
    np.savez(dir_frames + "cauchy",
             x = x,
             bw_cauchy = bw_cauchy,
             bw_cauchy_overcomplete = np.array(bw_cauchy_overcomplete, dtype=float),
             ss_cauchy_overcomplete = np.array(ss_cauchy_overcomplete, dtype=float),

             a0_cauchy1 = a0_cauchy1,
             a0_cauchy2 = a0_cauchy2,
             a0_cauchy3 = a0_cauchy3,

             a0_cauchy_overcomplete1 = a0_cauchy_overcomplete1,
             a0_cauchy_overcomplete2 = a0_cauchy_overcomplete2,
             a0_cauchy_overcomplete3 = a0_cauchy_overcomplete3
             )



# Fourier:
# ========

if set_frames_fourier:
    a0_fourier1 = csF.fourier()

if save_frames_fourier and set_frames_fourier:
    np.savez(dir_frames + "fourier",
             x = x,
             a0_fourier1 = a0_fourier1
             )


# =============================================================================
# Filters
# =============================================================================

cutoffs = [[1, None], [0.6, 1.2], [70, None]] # where 70 is 70° celsius

# Signal:
# =============================================================================

# Heaviside:
# ----------

if set_signal_filters_heaviside:
    y_heaviside1 = a0_heaviside1.dot(y_sparse) # to be saved
    y_heaviside2 = a0_heaviside2.dot(y_sparse) # to be saved
    y_heaviside3 = a0_heaviside3.dot(y_sparse) # to be saved

    csFr_y1_heaviside = cssr.Filters(y_heaviside1, x, cutoffs[0][0], filter_signal=True)
    csFr_y2_heaviside = cssr.Filters(y_heaviside2, x, cutoffs[0][0], filter_signal=True)
    csFr_y3_heaviside = cssr.Filters(y_heaviside3, x, cutoffs[0][0], filter_signal=True)

    y_heaviside_heaviside1 = csFr_y1_heaviside.heaviside_lowpass_filter()
    y_heaviside_heaviside2 = csFr_y2_heaviside.heaviside_lowpass_filter()
    y_heaviside_heaviside3 = csFr_y3_heaviside.heaviside_lowpass_filter()

    y_heaviside_fir1 = csFr_y1_heaviside.fir_filter()
    y_heaviside_fir2 = csFr_y2_heaviside.fir_filter()
    y_heaviside_fir3 = csFr_y3_heaviside.fir_filter()

    csFr_y1_heaviside.cutoff = cutoffs[1]
    csFr_y2_heaviside.cutoff = cutoffs[1]
    csFr_y3_heaviside.cutoff = cutoffs[1]

    y_heaviside_fir4 = csFr_y1_heaviside.fir_filter(numtabs=5, pass_zero="bandstop")
    y_heaviside_fir5 = csFr_y2_heaviside.fir_filter(numtabs=5, pass_zero="bandstop")
    y_heaviside_fir6 = csFr_y3_heaviside.fir_filter(numtabs=5, pass_zero="bandstop")

    y_heaviside_butter1 = csFr_y1_heaviside.butter_filter(order=5, btype="bandstop")
    y_heaviside_butter2 = csFr_y2_heaviside.butter_filter(order=5, btype="bandstop")
    y_heaviside_butter3 = csFr_y3_heaviside.butter_filter(order=5, btype="bandstop")

    y_heaviside1_filtered1 = csFr_y1_heaviside.truncated_a0 # to be saved
    y_heaviside2_filtered1 = csFr_y2_heaviside.truncated_a0 # to be saved
    y_heaviside3_filtered1 = csFr_y3_heaviside.truncated_a0 # to be saved

    y_heaviside_filter1_record1 = csFr_y1_heaviside.filter_record(name_only=True) # to be saved
    y_heaviside_filter2_record1 = csFr_y2_heaviside.filter_record(name_only=True) # to be saved
    y_heaviside_filter3_record1 = csFr_y3_heaviside.filter_record(name_only=True) # to be saved

    csFr_y1_heaviside.rest()
    csFr_y2_heaviside.rest()
    csFr_y3_heaviside.rest()

    y_heaviside_butter4 = csFr_y1_heaviside.butter_filter()
    y_heaviside_butter5 = csFr_y2_heaviside.butter_filter()
    y_heaviside_butter6 = csFr_y3_heaviside.butter_filter()

    y_heaviside_instrumental1 = csFr_y1_heaviside.instrumental_lowpass_filter()
    y_heaviside_instrumental2 = csFr_y2_heaviside.instrumental_lowpass_filter()
    y_heaviside_instrumental3 = csFr_y3_heaviside.instrumental_lowpass_filter()

    csFr_y1_heaviside.cutoff = cutoffs[2][0]
    csFr_y2_heaviside.cutoff = cutoffs[2][0]
    csFr_y3_heaviside.cutoff = cutoffs[2][0]

    y_heaviside_thermal1 = csFr_y1_heaviside.thermal_lowpass_filter()
    y_heaviside_thermal2 = csFr_y2_heaviside.thermal_lowpass_filter()
    y_heaviside_thermal3 = csFr_y3_heaviside.thermal_lowpass_filter()

    y_heaviside1_filtered2 = csFr_y1_heaviside.truncated_a0 # to be saved
    y_heaviside2_filtered2 = csFr_y2_heaviside.truncated_a0 # to be saved
    y_heaviside3_filtered2 = csFr_y3_heaviside.truncated_a0 # to be saved

    y_heaviside_filter1_record2 = csFr_y1_heaviside.filter_record(name_only=True) # to be saved
    y_heaviside_filter2_record2 = csFr_y2_heaviside.filter_record(name_only=True) # to be saved
    y_heaviside_filter3_record2 = csFr_y3_heaviside.filter_record(name_only=True) # to be saved


if save_signal_filters_heaviside and set_signal_filters_heaviside:
    np.savez(dir_filters + "heaviside_signal",
                 cutoffs = cutoffs,
                 y_sparse = y_sparse,

                 y_heaviside1 = y_heaviside1,
                 y_heaviside2 = y_heaviside2,
                 y_heaviside3 = y_heaviside3,

                 y_heaviside1_filtered1 = y_heaviside1_filtered1,
                 y_heaviside2_filtered1 = y_heaviside2_filtered1,
                 y_heaviside3_filtered1 = y_heaviside3_filtered1,

                 y_heaviside_filter1_record1 = y_heaviside_filter1_record1,
                 y_heaviside_filter2_record1 = y_heaviside_filter2_record1,
                 y_heaviside_filter3_record1 = y_heaviside_filter3_record1,

                 y_heaviside1_filtered2 = y_heaviside1_filtered2,
                 y_heaviside2_filtered2 = y_heaviside2_filtered2,
                 y_heaviside3_filtered2 = y_heaviside3_filtered2,

                 y_heaviside_filter1_record2 = y_heaviside_filter1_record2,
                 y_heaviside_filter2_record2 = y_heaviside_filter2_record2,
                 y_heaviside_filter3_record2 = y_heaviside_filter3_record2
                 )


# Gaussian:
# ---------

if set_signal_filters_gaussian:
    y_gaussian1 = a0_gaussian1.dot(y_sparse) # to be saved
    y_gaussian2 = a0_gaussian2.dot(y_sparse) # to be saved
    y_gaussian3 = a0_gaussian3.dot(y_sparse) # to be saved

    csFr_y1_gaussian = cssr.Filters(y_gaussian1, x, cutoffs[0][0], filter_signal=True)
    csFr_y2_gaussian = cssr.Filters(y_gaussian2, x, cutoffs[0][0], filter_signal=True)
    csFr_y3_gaussian = cssr.Filters(y_gaussian3, x, cutoffs[0][0], filter_signal=True)

    y_gaussian_heaviside1 = csFr_y1_gaussian.heaviside_lowpass_filter()
    y_gaussian_heaviside2 = csFr_y2_gaussian.heaviside_lowpass_filter()
    y_gaussian_heaviside3 = csFr_y3_gaussian.heaviside_lowpass_filter()

    y_gaussian_fir1 = csFr_y1_gaussian.fir_filter()
    y_gaussian_fir2 = csFr_y2_gaussian.fir_filter()
    y_gaussian_fir3 = csFr_y3_gaussian.fir_filter()

    csFr_y1_gaussian.cutoff = cutoffs[1]
    csFr_y2_gaussian.cutoff = cutoffs[1]
    csFr_y3_gaussian.cutoff = cutoffs[1]

    y_gaussian_fir4 = csFr_y1_gaussian.fir_filter(numtabs=5, pass_zero="bandstop")
    y_gaussian_fir5 = csFr_y2_gaussian.fir_filter(numtabs=5, pass_zero="bandstop")
    y_gaussian_fir6 = csFr_y3_gaussian.fir_filter(numtabs=5, pass_zero="bandstop")

    y_gaussian_butter1 = csFr_y1_gaussian.butter_filter(order=5, btype="bandstop")
    y_gaussian_butter2 = csFr_y2_gaussian.butter_filter(order=5, btype="bandstop")
    y_gaussian_butter3 = csFr_y3_gaussian.butter_filter(order=5, btype="bandstop")

    y_gaussian1_filtered1 = csFr_y1_gaussian.truncated_a0 # to be saved
    y_gaussian2_filtered1 = csFr_y2_gaussian.truncated_a0 # to be saved
    y_gaussian3_filtered1 = csFr_y3_gaussian.truncated_a0 # to be saved

    y_gaussian_filter1_record1 = csFr_y1_gaussian.filter_record(name_only=True) # to be saved
    y_gaussian_filter2_record1 = csFr_y2_gaussian.filter_record(name_only=True) # to be saved
    y_gaussian_filter3_record1 = csFr_y3_gaussian.filter_record(name_only=True) # to be saved

    csFr_y1_gaussian.rest()
    csFr_y2_gaussian.rest()
    csFr_y3_gaussian.rest()

    y_gaussian_butter4 = csFr_y1_gaussian.butter_filter()
    y_gaussian_butter5 = csFr_y2_gaussian.butter_filter()
    y_gaussian_butter6 = csFr_y3_gaussian.butter_filter()

    y_gaussian_instrumental1 = csFr_y1_gaussian.instrumental_lowpass_filter()
    y_gaussian_instrumental2 = csFr_y2_gaussian.instrumental_lowpass_filter()
    y_gaussian_instrumental3 = csFr_y3_gaussian.instrumental_lowpass_filter()

    csFr_y1_gaussian.cutoff = cutoffs[2][0]
    csFr_y2_gaussian.cutoff = cutoffs[2][0]
    csFr_y3_gaussian.cutoff = cutoffs[2][0]

    y_gaussian_thermal1 = csFr_y1_gaussian.thermal_lowpass_filter()
    y_gaussian_thermal2 = csFr_y2_gaussian.thermal_lowpass_filter()
    y_gaussian_thermal3 = csFr_y3_gaussian.thermal_lowpass_filter()

    y_gaussian1_filtered2 = csFr_y1_gaussian.truncated_a0 # to be saved
    y_gaussian2_filtered2 = csFr_y2_gaussian.truncated_a0 # to be saved
    y_gaussian3_filtered2 = csFr_y3_gaussian.truncated_a0 # to be saved

    y_gaussian_filter1_record2 = csFr_y1_gaussian.filter_record(name_only=True) # to be saved
    y_gaussian_filter2_record2 = csFr_y2_gaussian.filter_record(name_only=True) # to be saved
    y_gaussian_filter3_record2 = csFr_y3_gaussian.filter_record(name_only=True) # to be saved


if save_signal_filters_gaussian and set_signal_filters_gaussian:
    np.savez(dir_filters + "gaussian_signal",
                 cutoffs = cutoffs,
                 y_sparse = y_sparse,

                 y_gaussian1 = y_gaussian1,
                 y_gaussian2 = y_gaussian2,
                 y_gaussian3 = y_gaussian3,

                 y_gaussian1_filtered1 = y_gaussian1_filtered1,
                 y_gaussian2_filtered1 = y_gaussian2_filtered1,
                 y_gaussian3_filtered1 = y_gaussian3_filtered1,

                 y_gaussian_filter1_record1 = y_gaussian_filter1_record1,
                 y_gaussian_filter2_record1 = y_gaussian_filter2_record1,
                 y_gaussian_filter3_record1 = y_gaussian_filter3_record1,

                 y_gaussian1_filtered2 = y_gaussian1_filtered2,
                 y_gaussian2_filtered2 = y_gaussian2_filtered2,
                 y_gaussian3_filtered2 = y_gaussian3_filtered2,

                 y_gaussian_filter1_record2 = y_gaussian_filter1_record2,
                 y_gaussian_filter2_record2 = y_gaussian_filter2_record2,
                 y_gaussian_filter3_record2 = y_gaussian_filter3_record2
                 )


# Cauchy:
# -------

if set_signal_filters_cauchy:
    y_cauchy1 = a0_cauchy1.dot(y_sparse) # to be saved
    y_cauchy2 = a0_cauchy2.dot(y_sparse) # to be saved
    y_cauchy3 = a0_cauchy3.dot(y_sparse) # to be saved

    csFr_y1_cauchy = cssr.Filters(y_cauchy1, x, cutoffs[0][0], filter_signal=True)
    csFr_y2_cauchy = cssr.Filters(y_cauchy2, x, cutoffs[0][0], filter_signal=True)
    csFr_y3_cauchy = cssr.Filters(y_cauchy3, x, cutoffs[0][0], filter_signal=True)

    y_cauchy_heaviside1 = csFr_y1_cauchy.heaviside_lowpass_filter()
    y_cauchy_heaviside2 = csFr_y2_cauchy.heaviside_lowpass_filter()
    y_cauchy_heaviside3 = csFr_y3_cauchy.heaviside_lowpass_filter()

    y_cauchy_fir1 = csFr_y1_cauchy.fir_filter()
    y_cauchy_fir2 = csFr_y2_cauchy.fir_filter()
    y_cauchy_fir3 = csFr_y3_cauchy.fir_filter()

    csFr_y1_cauchy.cutoff = cutoffs[1]
    csFr_y2_cauchy.cutoff = cutoffs[1]
    csFr_y3_cauchy.cutoff = cutoffs[1]

    y_cauchy_fir4 = csFr_y1_cauchy.fir_filter(numtabs=5, pass_zero="bandstop")
    y_cauchy_fir5 = csFr_y2_cauchy.fir_filter(numtabs=5, pass_zero="bandstop")
    y_cauchy_fir6 = csFr_y3_cauchy.fir_filter(numtabs=5, pass_zero="bandstop")

    y_cauchy_butter1 = csFr_y1_cauchy.butter_filter(order=5, btype="bandstop")
    y_cauchy_butter2 = csFr_y2_cauchy.butter_filter(order=5, btype="bandstop")
    y_cauchy_butter3 = csFr_y3_cauchy.butter_filter(order=5, btype="bandstop")

    y_cauchy1_filtered1 = csFr_y1_cauchy.truncated_a0 # to be saved
    y_cauchy2_filtered1 = csFr_y2_cauchy.truncated_a0 # to be saved
    y_cauchy3_filtered1 = csFr_y3_cauchy.truncated_a0 # to be saved

    y_cauchy_filter1_record1 = csFr_y1_cauchy.filter_record(name_only=True) # to be saved
    y_cauchy_filter2_record1 = csFr_y2_cauchy.filter_record(name_only=True) # to be saved
    y_cauchy_filter3_record1 = csFr_y3_cauchy.filter_record(name_only=True) # to be saved

    csFr_y1_cauchy.rest()
    csFr_y2_cauchy.rest()
    csFr_y3_cauchy.rest()

    y_cauchy_butter4 = csFr_y1_cauchy.butter_filter()
    y_cauchy_butter5 = csFr_y2_cauchy.butter_filter()
    y_cauchy_butter6 = csFr_y3_cauchy.butter_filter()

    y_cauchy_instrumental1 = csFr_y1_cauchy.instrumental_lowpass_filter()
    y_cauchy_instrumental2 = csFr_y2_cauchy.instrumental_lowpass_filter()
    y_cauchy_instrumental3 = csFr_y3_cauchy.instrumental_lowpass_filter()

    csFr_y1_cauchy.cutoff = cutoffs[2][0]
    csFr_y2_cauchy.cutoff = cutoffs[2][0]
    csFr_y3_cauchy.cutoff = cutoffs[2][0]

    y_cauchy_thermal1 = csFr_y1_cauchy.thermal_lowpass_filter()
    y_cauchy_thermal2 = csFr_y2_cauchy.thermal_lowpass_filter()
    y_cauchy_thermal3 = csFr_y3_cauchy.thermal_lowpass_filter()

    y_cauchy1_filtered2 = csFr_y1_cauchy.truncated_a0 # to be saved
    y_cauchy2_filtered2 = csFr_y2_cauchy.truncated_a0 # to be saved
    y_cauchy3_filtered2 = csFr_y3_cauchy.truncated_a0 # to be saved

    y_cauchy_filter1_record2 = csFr_y1_cauchy.filter_record(name_only=True) # to be saved
    y_cauchy_filter2_record2 = csFr_y2_cauchy.filter_record(name_only=True) # to be saved
    y_cauchy_filter3_record2 = csFr_y3_cauchy.filter_record(name_only=True) # to be saved


if save_signal_filters_cauchy and set_signal_filters_cauchy:
    np.savez(dir_filters + "cauchy_signal",
                 cutoffs = cutoffs,
                 y_sparse = y_sparse,

                 y_cauchy1 = y_cauchy1,
                 y_cauchy2 = y_cauchy2,
                 y_cauchy3 = y_cauchy3,

                 y_cauchy1_filtered1 = y_cauchy1_filtered1,
                 y_cauchy2_filtered1 = y_cauchy2_filtered1,
                 y_cauchy3_filtered1 = y_cauchy3_filtered1,

                 y_cauchy_filter1_record1 = y_cauchy_filter1_record1,
                 y_cauchy_filter2_record1 = y_cauchy_filter2_record1,
                 y_cauchy_filter3_record1 = y_cauchy_filter3_record1,

                 y_cauchy1_filtered2 = y_cauchy1_filtered2,
                 y_cauchy2_filtered2 = y_cauchy2_filtered2,
                 y_cauchy3_filtered2 = y_cauchy3_filtered2,

                 y_cauchy_filter1_record2 = y_cauchy_filter1_record2,
                 y_cauchy_filter2_record2 = y_cauchy_filter2_record2,
                 y_cauchy_filter3_record2 = y_cauchy_filter3_record2
                 )

# Fourier:
# --------

if set_signal_filters_fourier:
    y_fourier1 = a0_fourier1.dot(y_sparse) # to be saved

    csFr_y1_fourier = cssr.Filters(y_fourier1, x, cutoffs[0][0], filter_signal=True)

    y_fourier_heaviside1 = csFr_y1_fourier.heaviside_lowpass_filter()
    y_fourier_fir1 = csFr_y1_fourier.fir_filter()

    csFr_y1_fourier.cutoff = cutoffs[1]

    y_fourier_fir2 = csFr_y1_fourier.fir_filter(numtabs=5, pass_zero="bandstop")
    y_fourier_butter1 = csFr_y1_fourier.butter_filter(order=5, btype="bandstop")

    y_fourier1_filtered1 = csFr_y1_fourier.truncated_a0 # to be saved
    y_fourier_filter1_record1 = csFr_y1_fourier.filter_record(name_only=True) # to be saved
    csFr_y1_fourier.rest()

    y_fourier_butter2 = csFr_y1_fourier.butter_filter()
    y_fourier_instrumental1 = csFr_y1_fourier.instrumental_lowpass_filter()
    csFr_y1_fourier.cutoff = cutoffs[2][0]

    y_fourier_thermal1 = csFr_y1_fourier.thermal_lowpass_filter()

    y_fourier1_filtered2 = csFr_y1_fourier.truncated_a0 # to be saved
    y_fourier_filter1_record2 = csFr_y1_fourier.filter_record(name_only=True) # to be saved


if save_signal_filters_fourier and set_signal_filters_fourier:
    np.savez(dir_filters + "fourier_signal",
                 cutoffs = cutoffs,
                 y_sparse = y_sparse,

                 y_fourier1 = y_fourier1,
                 y_fourier1_filtered1 = y_fourier1_filtered1,
                 y_fourier_filter1_record1 = y_fourier_filter1_record1,
                 y_fourier1_filtered2 = y_fourier1_filtered2,
                 y_fourier_filter1_record2 = y_fourier_filter1_record2
                 )

# Frames:
# =============================================================================

# Heaviside:
# ----------

# Dicionaries:
# ............

if set_frame_filters_heaviside:
    csFr_a01_heaviside = cssr.Filters(a0_heaviside1, x, cutoffs[0][0])
    csFr_a02_heaviside = cssr.Filters(a0_heaviside2, x, cutoffs[0][0])
    csFr_a03_heaviside = cssr.Filters(a0_heaviside3, x, cutoffs[0][0])

    a0_heaviside_heaviside1 = csFr_a01_heaviside.heaviside_lowpass_filter()
    a0_heaviside_heaviside2 = csFr_a02_heaviside.heaviside_lowpass_filter()
    a0_heaviside_heaviside3 = csFr_a03_heaviside.heaviside_lowpass_filter()

    a0_heaviside_fir1 = csFr_a01_heaviside.fir_filter()
    a0_heaviside_fir2 = csFr_a02_heaviside.fir_filter()
    a0_heaviside_fir3 = csFr_a03_heaviside.fir_filter()

    csFr_a01_heaviside.cutoff = cutoffs[1]
    csFr_a02_heaviside.cutoff = cutoffs[1]
    csFr_a03_heaviside.cutoff = cutoffs[1]

    a0_heaviside_fir4 = csFr_a01_heaviside.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_heaviside_fir5 = csFr_a02_heaviside.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_heaviside_fir6 = csFr_a03_heaviside.fir_filter(numtabs=5, pass_zero="bandstop")

    a0_heaviside_butter1 = csFr_a01_heaviside.butter_filter(order=5, btype="bandstop")
    a0_heaviside_butter2 = csFr_a02_heaviside.butter_filter(order=5, btype="bandstop")
    a0_heaviside_butter3 = csFr_a03_heaviside.butter_filter(order=5, btype="bandstop")

    a0_heaviside1_filtered1 = csFr_a01_heaviside.truncated_a0 # to be saved
    a0_heaviside2_filtered1 = csFr_a02_heaviside.truncated_a0 # to be saved
    a0_heaviside3_filtered1 = csFr_a03_heaviside.truncated_a0 # to be saved

    a0_heaviside_filter1_record1 = csFr_a01_heaviside.filter_record(name_only=True) # to be saved
    a0_heaviside_filter2_record1 = csFr_a02_heaviside.filter_record(name_only=True) # to be saved
    a0_heaviside_filter3_record1 = csFr_a03_heaviside.filter_record(name_only=True) # to be saved

    csFr_a01_heaviside.rest()
    csFr_a02_heaviside.rest()
    csFr_a03_heaviside.rest()

    a0_heaviside_butter4 = csFr_a01_heaviside.butter_filter()
    a0_heaviside_butter5 = csFr_a02_heaviside.butter_filter()
    a0_heaviside_butter6 = csFr_a03_heaviside.butter_filter()

    a0_heaviside_instrumental1 = csFr_a01_heaviside.instrumental_lowpass_filter()
    a0_heaviside_instrumental2 = csFr_a02_heaviside.instrumental_lowpass_filter()
    a0_heaviside_instrumental3 = csFr_a03_heaviside.instrumental_lowpass_filter()


    csFr_a01_heaviside.cutoff = cutoffs[2][0]
    csFr_a02_heaviside.cutoff = cutoffs[2][0]
    csFr_a03_heaviside.cutoff = cutoffs[2][0]

    a0_heaviside_thermal1 = csFr_a01_heaviside.thermal_lowpass_filter()
    a0_heaviside_thermal2 = csFr_a02_heaviside.thermal_lowpass_filter()
    a0_heaviside_thermal3 = csFr_a03_heaviside.thermal_lowpass_filter()

    a0_heaviside1_filtered2 = csFr_a01_heaviside.truncated_a0 # to be saved
    a0_heaviside2_filtered2 = csFr_a02_heaviside.truncated_a0 # to be saved
    a0_heaviside3_filtered2 = csFr_a03_heaviside.truncated_a0 # to be saved

    a0_heaviside_filter1_record2 = csFr_a01_heaviside.filter_record(name_only=True) # to be saved
    a0_heaviside_filter2_record2 = csFr_a02_heaviside.filter_record(name_only=True) # to be saved
    a0_heaviside_filter3_record2 = csFr_a03_heaviside.filter_record(name_only=True) # to be saved


# Overcomplete dicionaries:
# .........................

if set_frame_filters_heaviside_overcomplete:
    csFr_a01_heaviside_overcomplete = cssr.Filters(a0_heaviside_overcomplete1, x, cutoffs[0][0])
    csFr_a02_heaviside_overcomplete = cssr.Filters(a0_heaviside_overcomplete2, x, cutoffs[0][0])
    csFr_a03_heaviside_overcomplete = cssr.Filters(a0_heaviside_overcomplete3, x, cutoffs[0][0])

    a0_heaviside_overcomplete_heaviside1 = csFr_a01_heaviside_overcomplete.heaviside_lowpass_filter()
    a0_heaviside_overcomplete_heaviside2 = csFr_a02_heaviside_overcomplete.heaviside_lowpass_filter()
    a0_heaviside_overcomplete_heaviside3 = csFr_a03_heaviside_overcomplete.heaviside_lowpass_filter()

    a0_heaviside_overcomplete_fir1 = csFr_a01_heaviside_overcomplete.fir_filter()
    a0_heaviside_overcomplete_fir2 = csFr_a02_heaviside_overcomplete.fir_filter()
    a0_heaviside_overcomplete_fir3 = csFr_a03_heaviside_overcomplete.fir_filter()

    csFr_a01_heaviside_overcomplete.cutoff = cutoffs[1]
    csFr_a02_heaviside_overcomplete.cutoff = cutoffs[1]
    csFr_a03_heaviside_overcomplete.cutoff = cutoffs[1]

    a0_heaviside_overcomplete_fir4 = csFr_a01_heaviside_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_heaviside_overcomplete_fir5 = csFr_a02_heaviside_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_heaviside_overcomplete_fir6 = csFr_a03_heaviside_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")

    a0_heaviside_overcomplete_butter1 = csFr_a01_heaviside_overcomplete.butter_filter(order=5, btype="bandstop")
    a0_heaviside_overcomplete_butter2 = csFr_a02_heaviside_overcomplete.butter_filter(order=5, btype="bandstop")
    a0_heaviside_overcomplete_butter3 = csFr_a03_heaviside_overcomplete.butter_filter(order=5, btype="bandstop")

    a0_heaviside_overcomplete1_filtered1 = csFr_a01_heaviside_overcomplete.truncated_a0 # to be saved
    a0_heaviside_overcomplete2_filtered1 = csFr_a02_heaviside_overcomplete.truncated_a0 # to be saved
    a0_heaviside_overcomplete3_filtered1 = csFr_a03_heaviside_overcomplete.truncated_a0 # to be saved

    a0_heaviside_overcomplete_filter1_record1 = csFr_a01_heaviside_overcomplete.filter_record(name_only=True) # to be saved
    a0_heaviside_overcomplete_filter2_record1 = csFr_a02_heaviside_overcomplete.filter_record(name_only=True) # to be saved
    a0_heaviside_overcomplete_filter3_record1 = csFr_a03_heaviside_overcomplete.filter_record(name_only=True) # to be saved

    csFr_a01_heaviside_overcomplete.rest()
    csFr_a02_heaviside_overcomplete.rest()
    csFr_a03_heaviside_overcomplete.rest()

    a0_heaviside_overcomplete_butter4 = csFr_a01_heaviside_overcomplete.butter_filter()
    a0_heaviside_overcomplete_butter5 = csFr_a02_heaviside_overcomplete.butter_filter()
    a0_heaviside_overcomplete_butter6 = csFr_a03_heaviside_overcomplete.butter_filter()

    a0_heaviside_overcomplete_instrumental1 = csFr_a01_heaviside_overcomplete.instrumental_lowpass_filter()
    a0_heaviside_overcomplete_instrumental2 = csFr_a02_heaviside_overcomplete.instrumental_lowpass_filter()
    a0_heaviside_overcomplete_instrumental3 = csFr_a03_heaviside_overcomplete.instrumental_lowpass_filter()

    csFr_a01_heaviside_overcomplete.cutoff = cutoffs[2][0]
    csFr_a02_heaviside_overcomplete.cutoff = cutoffs[2][0]
    csFr_a03_heaviside_overcomplete.cutoff = cutoffs[2][0]

    a0_heaviside_overcomplete_thermal1 = csFr_a01_heaviside_overcomplete.thermal_lowpass_filter()
    a0_heaviside_overcomplete_thermal2 = csFr_a02_heaviside_overcomplete.thermal_lowpass_filter()
    a0_heaviside_overcomplete_thermal3 = csFr_a03_heaviside_overcomplete.thermal_lowpass_filter()

    a0_heaviside_overcomplete1_filtered2 = csFr_a01_heaviside_overcomplete.truncated_a0 # to be saved
    a0_heaviside_overcomplete2_filtered2 = csFr_a02_heaviside_overcomplete.truncated_a0 # to be saved
    a0_heaviside_overcomplete3_filtered2 = csFr_a03_heaviside_overcomplete.truncated_a0 # to be saved

    a0_heaviside_overcomplete_filter1_record2 = csFr_a01_heaviside_overcomplete.filter_record(name_only=True) # to be saved
    a0_heaviside_overcomplete_filter2_record2 = csFr_a02_heaviside_overcomplete.filter_record(name_only=True) # to be saved
    a0_heaviside_overcomplete_filter3_record2 = csFr_a03_heaviside_overcomplete.filter_record(name_only=True) # to be saved


if save_frame_filters_heaviside and set_frame_filters_heaviside and set_frame_filters_heaviside_overcomplete:
    np.savez(dir_filters + "heaviside_frame",
                 cutoffs = cutoffs,

                 a0_heaviside1_filtered1 = a0_heaviside1_filtered1,
                 a0_heaviside2_filtered1 = a0_heaviside2_filtered1,
                 a0_heaviside3_filtered1 = a0_heaviside3_filtered1,

                 a0_heaviside_filter1_record1 = a0_heaviside_filter1_record1,
                 a0_heaviside_filter2_record1 = a0_heaviside_filter2_record1,
                 a0_heaviside_filter3_record1 = a0_heaviside_filter3_record1,

                 a0_heaviside1_filtered2 = a0_heaviside1_filtered2,
                 a0_heaviside2_filtered2 = a0_heaviside2_filtered2,
                 a0_heaviside3_filtered2 = a0_heaviside3_filtered2,

                 a0_heaviside_filter1_record2 = a0_heaviside_filter1_record2,
                 a0_heaviside_filter2_record2 = a0_heaviside_filter2_record2,
                 a0_heaviside_filter3_record2 = a0_heaviside_filter3_record2,


                 a0_heaviside_overcomplete1_filtered1 = a0_heaviside_overcomplete1_filtered1,
                 a0_heaviside_overcomplete2_filtered1 = a0_heaviside_overcomplete2_filtered1,
                 a0_heaviside_overcomplete3_filtered1 = a0_heaviside_overcomplete3_filtered1,

                 a0_heaviside_overcomplete_filter1_record1 = a0_heaviside_overcomplete_filter1_record1,
                 a0_heaviside_overcomplete_filter2_record1 = a0_heaviside_overcomplete_filter2_record1,
                 a0_heaviside_overcomplete_filter3_record1 = a0_heaviside_overcomplete_filter3_record1,

                 a0_heaviside_overcomplete1_filtered2 = a0_heaviside_overcomplete1_filtered2,
                 a0_heaviside_overcomplete2_filtered2 = a0_heaviside_overcomplete2_filtered2,
                 a0_heaviside_overcomplete3_filtered2 = a0_heaviside_overcomplete3_filtered2,

                 a0_heaviside_overcomplete_filter1_record2 = a0_heaviside_overcomplete_filter1_record2,
                 a0_heaviside_overcomplete_filter2_record2 = a0_heaviside_overcomplete_filter2_record2,
                 a0_heaviside_overcomplete_filter3_record2 = a0_heaviside_overcomplete_filter3_record2
                 )

# Gaussian:
# ---------

# Dicionaries:
# ............

if set_frame_filters_gaussian:
    csFr_a01_gaussian = cssr.Filters(a0_gaussian1, x, cutoffs[0][0])
    csFr_a02_gaussian = cssr.Filters(a0_gaussian2, x, cutoffs[0][0])
    csFr_a03_gaussian = cssr.Filters(a0_gaussian3, x, cutoffs[0][0])

    a0_gaussian_heaviside1 = csFr_a01_gaussian.heaviside_lowpass_filter()
    a0_gaussian_heaviside2 = csFr_a02_gaussian.heaviside_lowpass_filter()
    a0_gaussian_heaviside3 = csFr_a03_gaussian.heaviside_lowpass_filter()

    a0_gaussian_fir1 = csFr_a01_gaussian.fir_filter()
    a0_gaussian_fir2 = csFr_a02_gaussian.fir_filter()
    a0_gaussian_fir3 = csFr_a03_gaussian.fir_filter()

    csFr_a01_gaussian.cutoff = cutoffs[1]
    csFr_a02_gaussian.cutoff = cutoffs[1]
    csFr_a03_gaussian.cutoff = cutoffs[1]

    a0_gaussian_fir4 = csFr_a01_gaussian.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_gaussian_fir5 = csFr_a02_gaussian.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_gaussian_fir6 = csFr_a03_gaussian.fir_filter(numtabs=5, pass_zero="bandstop")

    a0_gaussian_butter1 = csFr_a01_gaussian.butter_filter(order=5, btype="bandstop")
    a0_gaussian_butter2 = csFr_a02_gaussian.butter_filter(order=5, btype="bandstop")
    a0_gaussian_butter3 = csFr_a03_gaussian.butter_filter(order=5, btype="bandstop")

    a0_gaussian1_filtered1 = csFr_a01_gaussian.truncated_a0 # to be saved
    a0_gaussian2_filtered1 = csFr_a02_gaussian.truncated_a0 # to be saved
    a0_gaussian3_filtered1 = csFr_a03_gaussian.truncated_a0 # to be saved

    a0_gaussian_filter1_record1 = csFr_a01_gaussian.filter_record(name_only=True) # to be saved
    a0_gaussian_filter2_record1 = csFr_a02_gaussian.filter_record(name_only=True) # to be saved
    a0_gaussian_filter3_record1 = csFr_a03_gaussian.filter_record(name_only=True) # to be saved

    csFr_a01_gaussian.rest()
    csFr_a02_gaussian.rest()
    csFr_a03_gaussian.rest()

    a0_gaussian_butter4 = csFr_a01_gaussian.butter_filter()
    a0_gaussian_butter5 = csFr_a02_gaussian.butter_filter()
    a0_gaussian_butter6 = csFr_a03_gaussian.butter_filter()

    a0_gaussian_instrumental1 = csFr_a01_gaussian.instrumental_lowpass_filter()
    a0_gaussian_instrumental2 = csFr_a02_gaussian.instrumental_lowpass_filter()
    a0_gaussian_instrumental3 = csFr_a03_gaussian.instrumental_lowpass_filter()

    csFr_a01_gaussian.cutoff = cutoffs[2][0]
    csFr_a02_gaussian.cutoff = cutoffs[2][0]
    csFr_a03_gaussian.cutoff = cutoffs[2][0]

    a0_gaussian_thermal1 = csFr_a01_gaussian.thermal_lowpass_filter()
    a0_gaussian_thermal2 = csFr_a02_gaussian.thermal_lowpass_filter()
    a0_gaussian_thermal3 = csFr_a03_gaussian.thermal_lowpass_filter()

    a0_gaussian1_filtered2 = csFr_a01_gaussian.truncated_a0 # to be saved
    a0_gaussian2_filtered2 = csFr_a02_gaussian.truncated_a0 # to be saved
    a0_gaussian3_filtered2 = csFr_a03_gaussian.truncated_a0 # to be saved

    a0_gaussian_filter1_record2 = csFr_a01_gaussian.filter_record(name_only=True) # to be saved
    a0_gaussian_filter2_record2 = csFr_a02_gaussian.filter_record(name_only=True) # to be saved
    a0_gaussian_filter3_record2 = csFr_a03_gaussian.filter_record(name_only=True) # to be saved


# Overcomplete dicionaries:
# .........................

if set_frame_filters_gaussian_overcomplete:
    csFr_a01_gaussian_overcomplete = cssr.Filters(a0_gaussian_overcomplete1, x, cutoffs[0][0])
    csFr_a02_gaussian_overcomplete = cssr.Filters(a0_gaussian_overcomplete2, x, cutoffs[0][0])
    csFr_a03_gaussian_overcomplete = cssr.Filters(a0_gaussian_overcomplete3, x, cutoffs[0][0])

    a0_gaussian_overcomplete_heaviside1 = csFr_a01_gaussian_overcomplete.heaviside_lowpass_filter()
    a0_gaussian_overcomplete_heaviside2 = csFr_a02_gaussian_overcomplete.heaviside_lowpass_filter()
    a0_gaussian_overcomplete_heaviside3 = csFr_a03_gaussian_overcomplete.heaviside_lowpass_filter()

    a0_gaussian_overcomplete_fir1 = csFr_a01_gaussian_overcomplete.fir_filter()
    a0_gaussian_overcomplete_fir2 = csFr_a02_gaussian_overcomplete.fir_filter()
    a0_gaussian_overcomplete_fir3 = csFr_a03_gaussian_overcomplete.fir_filter()

    csFr_a01_gaussian_overcomplete.cutoff = cutoffs[1]
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[1]
    csFr_a03_gaussian_overcomplete.cutoff = cutoffs[1]

    a0_gaussian_overcomplete_fir4 = csFr_a01_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_gaussian_overcomplete_fir5 = csFr_a02_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_gaussian_overcomplete_fir6 = csFr_a03_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")

    a0_gaussian_overcomplete_butter1 = csFr_a01_gaussian_overcomplete.butter_filter(order=5, btype="bandstop")
    a0_gaussian_overcomplete_butter2 = csFr_a02_gaussian_overcomplete.butter_filter(order=5, btype="bandstop")
    a0_gaussian_overcomplete_butter3 = csFr_a03_gaussian_overcomplete.butter_filter(order=5, btype="bandstop")

    a0_gaussian_overcomplete1_filtered1 = csFr_a01_gaussian_overcomplete.truncated_a0 # to be saved
    a0_gaussian_overcomplete2_filtered1 = csFr_a02_gaussian_overcomplete.truncated_a0 # to be saved
    a0_gaussian_overcomplete3_filtered1 = csFr_a03_gaussian_overcomplete.truncated_a0 # to be saved

    a0_gaussian_overcomplete_filter1_record1 = csFr_a01_gaussian_overcomplete.filter_record(name_only=True) # to be saved
    a0_gaussian_overcomplete_filter2_record1 = csFr_a02_gaussian_overcomplete.filter_record(name_only=True) # to be saved
    a0_gaussian_overcomplete_filter3_record1 = csFr_a03_gaussian_overcomplete.filter_record(name_only=True) # to be saved

    csFr_a01_gaussian_overcomplete.rest()
    csFr_a02_gaussian_overcomplete.rest()
    csFr_a03_gaussian_overcomplete.rest()

    a0_gaussian_overcomplete_butter4 = csFr_a01_gaussian_overcomplete.butter_filter()
    a0_gaussian_overcomplete_butter5 = csFr_a02_gaussian_overcomplete.butter_filter()
    a0_gaussian_overcomplete_butter6 = csFr_a03_gaussian_overcomplete.butter_filter()

    a0_gaussian_overcomplete_instrumental1 = csFr_a01_gaussian_overcomplete.instrumental_lowpass_filter()
    a0_gaussian_overcomplete_instrumental2 = csFr_a02_gaussian_overcomplete.instrumental_lowpass_filter()
    a0_gaussian_overcomplete_instrumental3 = csFr_a03_gaussian_overcomplete.instrumental_lowpass_filter()

    csFr_a01_gaussian_overcomplete.cutoff = cutoffs[2][0]
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[2][0]
    csFr_a03_gaussian_overcomplete.cutoff = cutoffs[2][0]

    a0_gaussian_overcomplete_thermal1 = csFr_a01_gaussian_overcomplete.thermal_lowpass_filter()
    a0_gaussian_overcomplete_thermal2 = csFr_a02_gaussian_overcomplete.thermal_lowpass_filter()
    a0_gaussian_overcomplete_thermal3 = csFr_a03_gaussian_overcomplete.thermal_lowpass_filter()

    a0_gaussian_overcomplete1_filtered2 = csFr_a01_gaussian_overcomplete.truncated_a0 # to be saved
    a0_gaussian_overcomplete2_filtered2 = csFr_a02_gaussian_overcomplete.truncated_a0 # to be saved
    a0_gaussian_overcomplete3_filtered2 = csFr_a03_gaussian_overcomplete.truncated_a0 # to be saved

    a0_gaussian_overcomplete_filter1_record2 = csFr_a01_gaussian_overcomplete.filter_record(name_only=True) # to be saved
    a0_gaussian_overcomplete_filter2_record2 = csFr_a02_gaussian_overcomplete.filter_record(name_only=True) # to be saved
    a0_gaussian_overcomplete_filter3_record2 = csFr_a03_gaussian_overcomplete.filter_record(name_only=True) # to be saved


if save_frame_filters_gaussian and set_frame_filters_gaussian and set_frame_filters_gaussian_overcomplete:
    np.savez(dir_filters + "gaussian_frame",
                 cutoffs = cutoffs,

                 a0_gaussian1_filtered1 = a0_gaussian1_filtered1,
                 a0_gaussian2_filtered1 = a0_gaussian2_filtered1,
                 a0_gaussian3_filtered1 = a0_gaussian3_filtered1,

                 a0_gaussian_filter1_record1 = a0_gaussian_filter1_record1,
                 a0_gaussian_filter2_record1 = a0_gaussian_filter2_record1,
                 a0_gaussian_filter3_record1 = a0_gaussian_filter3_record1,

                 a0_gaussian1_filtered2 = a0_gaussian1_filtered2,
                 a0_gaussian2_filtered2 = a0_gaussian2_filtered2,
                 a0_gaussian3_filtered2 = a0_gaussian3_filtered2,

                 a0_gaussian_filter1_record2 = a0_gaussian_filter1_record2,
                 a0_gaussian_filter2_record2 = a0_gaussian_filter2_record2,
                 a0_gaussian_filter3_record2 = a0_gaussian_filter3_record2,


                 a0_gaussian_overcomplete1_filtered1 = a0_gaussian_overcomplete1_filtered1,
                 a0_gaussian_overcomplete2_filtered1 = a0_gaussian_overcomplete2_filtered1,
                 a0_gaussian_overcomplete3_filtered1 = a0_gaussian_overcomplete3_filtered1,

                 a0_gaussian_overcomplete_filter1_record1 = a0_gaussian_overcomplete_filter1_record1,
                 a0_gaussian_overcomplete_filter2_record1 = a0_gaussian_overcomplete_filter2_record1,
                 a0_gaussian_overcomplete_filter3_record1 = a0_gaussian_overcomplete_filter3_record1,

                 a0_gaussian_overcomplete1_filtered2 = a0_gaussian_overcomplete1_filtered2,
                 a0_gaussian_overcomplete2_filtered2 = a0_gaussian_overcomplete2_filtered2,
                 a0_gaussian_overcomplete3_filtered2 = a0_gaussian_overcomplete3_filtered2,

                 a0_gaussian_overcomplete_filter1_record2 = a0_gaussian_overcomplete_filter1_record2,
                 a0_gaussian_overcomplete_filter2_record2 = a0_gaussian_overcomplete_filter2_record2,
                 a0_gaussian_overcomplete_filter3_record2 = a0_gaussian_overcomplete_filter3_record2
                 )

# Cauchy:
# -------

# Dicionaries:
# ............
if set_frame_filters_cauchy:
    csFr_a01_cauchy = cssr.Filters(a0_cauchy1, x, cutoffs[0][0])
    csFr_a02_cauchy = cssr.Filters(a0_cauchy2, x, cutoffs[0][0])
    csFr_a03_cauchy = cssr.Filters(a0_cauchy3, x, cutoffs[0][0])

    a0_cauchy_heaviside1 = csFr_a01_cauchy.heaviside_lowpass_filter()
    a0_cauchy_heaviside2 = csFr_a02_cauchy.heaviside_lowpass_filter()
    a0_cauchy_heaviside3 = csFr_a03_cauchy.heaviside_lowpass_filter()

    a0_cauchy_fir1 = csFr_a01_cauchy.fir_filter()
    a0_cauchy_fir2 = csFr_a02_cauchy.fir_filter()
    a0_cauchy_fir3 = csFr_a03_cauchy.fir_filter()

    csFr_a01_cauchy.cutoff = cutoffs[1]
    csFr_a02_cauchy.cutoff = cutoffs[1]
    csFr_a03_cauchy.cutoff = cutoffs[1]

    a0_cauchy_fir4 = csFr_a01_cauchy.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_cauchy_fir5 = csFr_a02_cauchy.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_cauchy_fir6 = csFr_a03_cauchy.fir_filter(numtabs=5, pass_zero="bandstop")

    a0_cauchy_butter1 = csFr_a01_cauchy.butter_filter(order=5, btype="bandstop")
    a0_cauchy_butter2 = csFr_a02_cauchy.butter_filter(order=5, btype="bandstop")
    a0_cauchy_butter3 = csFr_a03_cauchy.butter_filter(order=5, btype="bandstop")

    a0_cauchy1_filtered1 = csFr_a01_cauchy.truncated_a0 # to be saved
    a0_cauchy2_filtered1 = csFr_a02_cauchy.truncated_a0 # to be saved
    a0_cauchy3_filtered1 = csFr_a03_cauchy.truncated_a0 # to be saved

    a0_cauchy_filter1_record1 = csFr_a01_cauchy.filter_record(name_only=True) # to be saved
    a0_cauchy_filter2_record1 = csFr_a02_cauchy.filter_record(name_only=True) # to be saved
    a0_cauchy_filter3_record1 = csFr_a03_cauchy.filter_record(name_only=True) # to be saved

    csFr_a01_cauchy.rest()
    csFr_a02_cauchy.rest()
    csFr_a03_cauchy.rest()

    a0_cauchy_butter4 = csFr_a01_cauchy.butter_filter()
    a0_cauchy_butter5 = csFr_a02_cauchy.butter_filter()
    a0_cauchy_butter6 = csFr_a03_cauchy.butter_filter()

    a0_cauchy_instrumental1 = csFr_a01_cauchy.instrumental_lowpass_filter()
    a0_cauchy_instrumental2 = csFr_a02_cauchy.instrumental_lowpass_filter()
    a0_cauchy_instrumental3 = csFr_a03_cauchy.instrumental_lowpass_filter()

    csFr_a01_cauchy.cutoff = cutoffs[2][0]
    csFr_a02_cauchy.cutoff = cutoffs[2][0]
    csFr_a03_cauchy.cutoff = cutoffs[2][0]

    a0_cauchy_thermal1 = csFr_a01_cauchy.thermal_lowpass_filter()
    a0_cauchy_thermal2 = csFr_a02_cauchy.thermal_lowpass_filter()
    a0_cauchy_thermal3 = csFr_a03_cauchy.thermal_lowpass_filter()

    a0_cauchy1_filtered2 = csFr_a01_cauchy.truncated_a0 # to be saved
    a0_cauchy2_filtered2 = csFr_a02_cauchy.truncated_a0 # to be saved
    a0_cauchy3_filtered2 = csFr_a03_cauchy.truncated_a0 # to be saved

    a0_cauchy_filter1_record2 = csFr_a01_cauchy.filter_record(name_only=True) # to be saved
    a0_cauchy_filter2_record2 = csFr_a02_cauchy.filter_record(name_only=True) # to be saved
    a0_cauchy_filter3_record2 = csFr_a03_cauchy.filter_record(name_only=True) # to be saved


# Overcomplete dicionaries:
# .........................
if set_frame_filters_cauchy_overcomplete:
    csFr_a01_cauchy_overcomplete = cssr.Filters(a0_cauchy_overcomplete1, x, cutoffs[0][0])
    csFr_a02_cauchy_overcomplete = cssr.Filters(a0_cauchy_overcomplete2, x, cutoffs[0][0])
    csFr_a03_cauchy_overcomplete = cssr.Filters(a0_cauchy_overcomplete3, x, cutoffs[0][0])

    a0_cauchy_overcomplete_heaviside1 = csFr_a01_cauchy_overcomplete.heaviside_lowpass_filter()
    a0_cauchy_overcomplete_heaviside2 = csFr_a02_cauchy_overcomplete.heaviside_lowpass_filter()
    a0_cauchy_overcomplete_heaviside3 = csFr_a03_cauchy_overcomplete.heaviside_lowpass_filter()

    a0_cauchy_overcomplete_fir1 = csFr_a01_cauchy_overcomplete.fir_filter()
    a0_cauchy_overcomplete_fir2 = csFr_a02_cauchy_overcomplete.fir_filter()
    a0_cauchy_overcomplete_fir3 = csFr_a03_cauchy_overcomplete.fir_filter()

    csFr_a01_cauchy_overcomplete.cutoff = cutoffs[1]
    csFr_a02_cauchy_overcomplete.cutoff = cutoffs[1]
    csFr_a03_cauchy_overcomplete.cutoff = cutoffs[1]

    a0_cauchy_overcomplete_fir4 = csFr_a01_cauchy_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_cauchy_overcomplete_fir5 = csFr_a02_cauchy_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_cauchy_overcomplete_fir6 = csFr_a03_cauchy_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop")

    a0_cauchy_overcomplete_butter1 = csFr_a01_cauchy_overcomplete.butter_filter(order=5, btype="bandstop")
    a0_cauchy_overcomplete_butter2 = csFr_a02_cauchy_overcomplete.butter_filter(order=5, btype="bandstop")
    a0_cauchy_overcomplete_butter3 = csFr_a03_cauchy_overcomplete.butter_filter(order=5, btype="bandstop")

    a0_cauchy_overcomplete1_filtered1 = csFr_a01_cauchy_overcomplete.truncated_a0 # to be saved
    a0_cauchy_overcomplete2_filtered1 = csFr_a02_cauchy_overcomplete.truncated_a0 # to be saved
    a0_cauchy_overcomplete3_filtered1 = csFr_a03_cauchy_overcomplete.truncated_a0 # to be saved

    a0_cauchy_overcomplete_filter1_record1 = csFr_a01_cauchy_overcomplete.filter_record(name_only=True) # to be saved
    a0_cauchy_overcomplete_filter2_record1 = csFr_a02_cauchy_overcomplete.filter_record(name_only=True) # to be saved
    a0_cauchy_overcomplete_filter3_record1 = csFr_a03_cauchy_overcomplete.filter_record(name_only=True) # to be saved

    csFr_a01_cauchy_overcomplete.rest()
    csFr_a02_cauchy_overcomplete.rest()
    csFr_a03_cauchy_overcomplete.rest()

    a0_cauchy_overcomplete_butter4 = csFr_a01_cauchy_overcomplete.butter_filter()
    a0_cauchy_overcomplete_butter5 = csFr_a02_cauchy_overcomplete.butter_filter()
    a0_cauchy_overcomplete_butter6 = csFr_a03_cauchy_overcomplete.butter_filter()

    a0_cauchy_overcomplete_instrumental1 = csFr_a01_cauchy_overcomplete.instrumental_lowpass_filter()
    a0_cauchy_overcomplete_instrumental2 = csFr_a02_cauchy_overcomplete.instrumental_lowpass_filter()
    a0_cauchy_overcomplete_instrumental3 = csFr_a03_cauchy_overcomplete.instrumental_lowpass_filter()

    csFr_a01_cauchy_overcomplete.cutoff = cutoffs[2][0]
    csFr_a02_cauchy_overcomplete.cutoff = cutoffs[2][0]
    csFr_a03_cauchy_overcomplete.cutoff = cutoffs[2][0]

    a0_cauchy_overcomplete_thermal1 = csFr_a01_cauchy_overcomplete.thermal_lowpass_filter()
    a0_cauchy_overcomplete_thermal2 = csFr_a02_cauchy_overcomplete.thermal_lowpass_filter()
    a0_cauchy_overcomplete_thermal3 = csFr_a03_cauchy_overcomplete.thermal_lowpass_filter()

    a0_cauchy_overcomplete1_filtered2 = csFr_a01_cauchy_overcomplete.truncated_a0 # to be saved
    a0_cauchy_overcomplete2_filtered2 = csFr_a02_cauchy_overcomplete.truncated_a0 # to be saved
    a0_cauchy_overcomplete3_filtered2 = csFr_a03_cauchy_overcomplete.truncated_a0 # to be saved

    a0_cauchy_overcomplete_filter1_record2 = csFr_a01_cauchy_overcomplete.filter_record(name_only=True) # to be saved
    a0_cauchy_overcomplete_filter2_record2 = csFr_a02_cauchy_overcomplete.filter_record(name_only=True) # to be saved
    a0_cauchy_overcomplete_filter3_record2 = csFr_a03_cauchy_overcomplete.filter_record(name_only=True) # to be saved


if save_frame_filters_cauchy and set_frame_filters_cauchy and set_frame_filters_cauchy_overcomplete:
    np.savez(dir_filters + "cauchy_frame",
                 cutoffs = cutoffs,

                 a0_cauchy1_filtered1 = a0_cauchy1_filtered1,
                 a0_cauchy2_filtered1 = a0_cauchy2_filtered1,
                 a0_cauchy3_filtered1 = a0_cauchy3_filtered1,

                 a0_cauchy_filter1_record1 = a0_cauchy_filter1_record1,
                 a0_cauchy_filter2_record1 = a0_cauchy_filter2_record1,
                 a0_cauchy_filter3_record1 = a0_cauchy_filter3_record1,

                 a0_cauchy1_filtered2 = a0_cauchy1_filtered2,
                 a0_cauchy2_filtered2 = a0_cauchy2_filtered2,
                 a0_cauchy3_filtered2 = a0_cauchy3_filtered2,

                 a0_cauchy_filter1_record2 = a0_cauchy_filter1_record2,
                 a0_cauchy_filter2_record2 = a0_cauchy_filter2_record2,
                 a0_cauchy_filter3_record2 = a0_cauchy_filter3_record2,


                 a0_cauchy_overcomplete1_filtered1 = a0_cauchy_overcomplete1_filtered1,
                 a0_cauchy_overcomplete2_filtered1 = a0_cauchy_overcomplete2_filtered1,
                 a0_cauchy_overcomplete3_filtered1 = a0_cauchy_overcomplete3_filtered1,

                 a0_cauchy_overcomplete_filter1_record1 = a0_cauchy_overcomplete_filter1_record1,
                 a0_cauchy_overcomplete_filter2_record1 = a0_cauchy_overcomplete_filter2_record1,
                 a0_cauchy_overcomplete_filter3_record1 = a0_cauchy_overcomplete_filter3_record1,

                 a0_cauchy_overcomplete1_filtered2 = a0_cauchy_overcomplete1_filtered2,
                 a0_cauchy_overcomplete2_filtered2 = a0_cauchy_overcomplete2_filtered2,
                 a0_cauchy_overcomplete3_filtered2 = a0_cauchy_overcomplete3_filtered2,

                 a0_cauchy_overcomplete_filter1_record2 = a0_cauchy_overcomplete_filter1_record2,
                 a0_cauchy_overcomplete_filter2_record2 = a0_cauchy_overcomplete_filter2_record2,
                 a0_cauchy_overcomplete_filter3_record2 = a0_cauchy_overcomplete_filter3_record2
                 )


# Fourier:
# --------

# Dicionaries:
# ............
if set_frame_filters_fourier:
    csFr_a01_fourier = cssr.Filters(a0_fourier1, x, cutoffs[0][0])

    a0_fourier_heaviside1 = csFr_a01_fourier.heaviside_lowpass_filter()
    a0_fourier_fir1 = csFr_a01_fourier.fir_filter()

    csFr_a01_fourier.cutoff = cutoffs[1]

    a0_fourier_fir2 = csFr_a01_fourier.fir_filter(numtabs=5, pass_zero="bandstop")
    a0_fourier_butter1 = csFr_a01_fourier.butter_filter(order=5, btype="bandstop")

    a0_fourier1_filtered1 = csFr_a01_fourier.truncated_a0 # to be saved
    a0_fourier_filter1_record1 = csFr_a01_fourier.filter_record(name_only=True) # to be saved
    csFr_a01_fourier.rest()

    a0_fourier_butter2 = csFr_a01_fourier.butter_filter()
    a0_fourier_instrumental1 = csFr_a01_fourier.instrumental_lowpass_filter()
    csFr_a01_fourier.cutoff = cutoffs[2][0]

    a0_fourier_thermal1 = csFr_a01_fourier.thermal_lowpass_filter()

    a0_fourier1_filtered2 = csFr_a01_fourier.truncated_a0 # to be saved
    a0_fourier_filter1_record2 = csFr_a01_fourier.filter_record(name_only=True) # to be saved


if save_frame_filters_fourier and set_frame_filters_fourier:
    np.savez(dir_filters + "fourier_frame",
                 cutoffs = cutoffs,

                 a0_fourier1_filtered1 = a0_fourier1_filtered1,

                 a0_fourier_filter1_record1 = a0_fourier_filter1_record1,

                 a0_fourier1_filtered2 = a0_fourier1_filtered2,

                 a0_fourier_filter1_record2 = a0_fourier_filter1_record2
                 )

# =============================================================================
# Measurement matrices
# =============================================================================

# under construction

