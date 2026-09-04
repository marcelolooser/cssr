"""
Module to construct test bench data for the measurement_matrices module of the cssr package.

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
dir_superresolvers = "data_superresolvers/"

# Set up configurations:
# ======================

set_superresolvers_gaussian = True
save_superresolvers_gaussian = True


# =============================================================================
# Superresolver configurations
# =============================================================================

max_iter = 8
noise_level = 1e-8

# =============================================================================
# Helper functions
# =============================================================================

def load_frame_data(dir_frames, frame_name):
    return np.load(dir_frames + frame_name + ".npz")


def load_filter_data(dir_filters, frame_name):
    return np.load(dir_filters + frame_name + ".npz")


def load_measurement_matrices_data(dir_measurement_matrices, frame_name):
    return np.load(dir_measurement_matrices + frame_name + ".npz")

# =============================================================================
# Start test data construction
# =============================================================================

# Gaussian:
# ---------

if set_superresolvers_gaussian:
    data_a0_gaussian = load_frame_data(dir_frames, "gaussian")
    data_y_gaussian = load_filter_data(dir_filters, "gaussian_signal")
    data_a_tr_gaussian = load_filter_data(dir_filters, "gaussian_frame")
    data_ar_gaussian = load_measurement_matrices_data(dir_measurement_matrices, "gaussian")

    # Dicionaries:
    # ............

    csS_frame21_gaussian_gauss = cssr.Superresolvers(data_a0_gaussian["a0_gaussian2"],
                                                   data_a_tr_gaussian["a0_gaussian2_filtered1"],
                                                   data_ar_gaussian["ar_gaussian21_gauss"])

    # Super-resolutions:
    # ^^^^^^^^^^^^^^^^^^

    y_gaussian21_gauss_sr_bp = csS_frame21_gaussian_gauss.bp(data_y_gaussian["y_gaussian2_filtered1"])
    y_gaussian21_gauss_sr_bpd = csS_frame21_gaussian_gauss.bpd(data_y_gaussian["y_gaussian2_filtered1"], noise_level)
    y_gaussian21_gauss_sr_ic = csS_frame21_gaussian_gauss.ic(data_y_gaussian["y_gaussian2_filtered1"], max_iter=max_iter)
    y_gaussian21_gauss_sr_nlht = csS_frame21_gaussian_gauss.nlht(data_y_gaussian["y_gaussian2_filtered1"], noise_level)
    y_gaussian21_gauss_sr_nlht_lasso = csS_frame21_gaussian_gauss.nlht_lasso(data_y_gaussian["y_gaussian2_filtered1"], max_iter=max_iter)



    # Overcomplete dicionaries:
    # .........................

    csS_frame21_gaussian_overcomplete_gauss = cssr.Superresolvers(data_a0_gaussian["a0_gaussian_overcomplete2"],
                                                   data_a_tr_gaussian["a0_gaussian_overcomplete2_filtered1"],
                                                   data_ar_gaussian["ar_gaussian_overcomplete21_gauss"])


    # Super-resolutions:
    # ^^^^^^^^^^^^^^^^^^

    y_gaussian_overcomplete21_gauss_sr_bp = csS_frame21_gaussian_overcomplete_gauss.bp(data_y_gaussian["y_gaussian2_filtered1"])
    y_gaussian_overcomplete21_gauss_sr_bpd = csS_frame21_gaussian_overcomplete_gauss.bpd(data_y_gaussian["y_gaussian2_filtered1"], noise_level)
    y_gaussian_overcomplete21_gauss_sr_ic = csS_frame21_gaussian_overcomplete_gauss.ic(data_y_gaussian["y_gaussian2_filtered1"], max_iter=max_iter)
    y_gaussian_overcomplete21_gauss_sr_nlht = csS_frame21_gaussian_overcomplete_gauss.nlht(data_y_gaussian["y_gaussian2_filtered1"], noise_level)
    y_gaussian_overcomplete21_gauss_sr_nlht_lasso = csS_frame21_gaussian_overcomplete_gauss.nlht_lasso(data_y_gaussian["y_gaussian2_filtered1"], max_iter=max_iter)




if save_superresolvers_gaussian and set_superresolvers_gaussian:
    np.savez(dir_superresolvers + "gaussian",

        y_gaussian21_gauss_sr_bp = y_gaussian21_gauss_sr_bp,
        y_gaussian21_gauss_sr_bpd = y_gaussian21_gauss_sr_bpd,
        y_gaussian21_gauss_sr_ic = y_gaussian21_gauss_sr_ic,
        y_gaussian21_gauss_sr_nlht = y_gaussian21_gauss_sr_nlht,
        y_gaussian21_gauss_sr_nlht_lasso = y_gaussian21_gauss_sr_nlht_lasso,

        y_gaussian_overcomplete21_gauss_sr_bp = y_gaussian_overcomplete21_gauss_sr_bp,
        y_gaussian_overcomplete21_gauss_sr_bpd = y_gaussian_overcomplete21_gauss_sr_bpd,
        y_gaussian_overcomplete21_gauss_sr_ic = y_gaussian_overcomplete21_gauss_sr_ic,
        y_gaussian_overcomplete21_gauss_sr_nlht = y_gaussian_overcomplete21_gauss_sr_nlht,
        y_gaussian_overcomplete21_gauss_sr_nlht_lasso = y_gaussian_overcomplete21_gauss_sr_nlht_lasso

        )

# =============================================================================