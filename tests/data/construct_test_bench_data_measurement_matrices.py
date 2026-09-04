"""
Module to construct test bench data for the measurement_matrices module of the cssr package.

@author: marcelo looser
"""

import scipy
import random
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import cssr

# =============================================================================
# Preliminaries
# =============================================================================

dir_stubs = "data_stubs/"
dir_frames = "data_frames/"
dir_filters = "data_filters/"
dir_measurement_matrices = "data_measurement_matrices/"

# Set up configurations:
# ======================


# Measurement matrices:
# ---------------------
set_measurment_marices_gaussian = True
set_measurment_marices_gaussian_overcomplete = True
save_measurment_marices_gaussian = True


# =============================================================================
# Trial test data
# =============================================================================

x = np.arange(0, 20, 0.1)
dim = len(x)

number_samples = 25


# Number of iterations:
# ---------------------

max_iter = 8
l = 3
p = 3


# =============================================================================
# Helper functions
# =============================================================================


def load_stub_data(dir_stubs, frame_name):
    return np.load(dir_stubs + frame_name + ".npz")


def load_filter_data(dir_filters, frame_name):
    return np.load(dir_filters + frame_name + ".npz")


def stub_random_random_from_data(data):
    def stub(*, size=None):
        try:
            return data[size]
        except KeyError:
            raise ValueError(f"No stub data available for the shape {size}.")
    return stub


def stub_random_binomial_from_data(data):
    def stub(*, n=None, p=None, size=None):
        try:
            return data[(n, p, size)]
        except KeyError:
            raise ValueError(f"No stub data available for {(n, p, size)}.")
    return stub


def stub_random_sample_from_data(data):
    def stub(*args):
        try:
            return data[args]
        except KeyError:
            raise ValueError(f"No stub data available for the shape {args}.")
    return stub


def stub_random_choice_from_data(data):
    def stub(*args, size=None):
        try:
            return data[(args[0], size)]
        except KeyError:
            raise ValueError(f"No stub data available for {(args[0], size)}.")
    return stub


def stub_random_randn_from_data(data):
    def stub(*args):
        try:
            return data[args]
        except KeyError:
            raise ValueError(f"No stub data available for the shape {args}.")
    return stub


def stub_random_unitary_from_data(data):
    def stub(*args):
        try:
            return data[args[0]]
        except KeyError:
            raise ValueError(f"No stub data available for the shape {args}.")
    return stub


# =============================================================================
# Start test data construction
# =============================================================================


# Gaussian:
# ---------

data_stub_gaussian = load_stub_data(dir_stubs, "gaussian")
shapes_gaussian = data_stub_gaussian["tracked_shapes_gaussian"]
shapes_gaussian = [tuple(item.tolist()) for item in shapes_gaussian]

np.random.random = stub_random_random_from_data(
    {
     shapes_gaussian[0] : data_stub_gaussian["uniform_ns_m_gaussian2"],
     shapes_gaussian[1] : data_stub_gaussian["uniform_ns_n_gaussian2"],

     shapes_gaussian[2] : data_stub_gaussian["uniform_ns_m_gaussian_overcomplete2"],
     shapes_gaussian[3] : data_stub_gaussian["uniform_ns_n_gaussian_overcomplete2"]
     })


probability = 0.5
np.random.binomial = stub_random_binomial_from_data(
    {
     (1, probability, shapes_gaussian[4]) : data_stub_gaussian["bernoulli_ns_m_gaussian2"],

     (1, probability, shapes_gaussian[5]) : data_stub_gaussian["bernoulli_ns_m_gaussian_overcomplete2"]
     })


random.sample = stub_random_sample_from_data(
    {
     (range(shapes_gaussian[6][0]), shapes_gaussian[6][1]) : data_stub_gaussian["samples_m_ns_gaussian2"],
     (range(shapes_gaussian[7][0]), shapes_gaussian[7][1]) : data_stub_gaussian["samples_n_m_gaussian2"],

     (range(shapes_gaussian[8][0]), shapes_gaussian[8][1]) : data_stub_gaussian["samples_m_ns_gaussian_overcomplete2"],
     (range(shapes_gaussian[9][0]), shapes_gaussian[9][1]) : data_stub_gaussian["samples_n_m_gaussian_overcomplete2"]
     })


choices = (-1,1)
np.random.choice = stub_random_choice_from_data(
    {
     (choices, shapes_gaussian[10][0]) : data_stub_gaussian["samples_binary_m_gaussian2"],

     (choices, shapes_gaussian[11][0]) : data_stub_gaussian["samples_binary_m_gaussian_overcomplete2"]
     })


np.random.randn = stub_random_randn_from_data(
    {
     shapes_gaussian[12] : data_stub_gaussian["gaussian_ns_m_gaussian2"],

     shapes_gaussian[13] : data_stub_gaussian["gaussian_ns_m_gaussian_overcomplete2"]
     })


scipy.stats.unitary_group.rvs = stub_random_unitary_from_data(
    {
     shapes_gaussian[14][0] : data_stub_gaussian["random_unitary_ns_m_gaussian"]
     })



# Dicionaries:
# ............

if set_measurment_marices_gaussian:
    data_gaussian = load_filter_data(dir_filters, "gaussian_frame")

    a0_gaussian2_filtered1 = data_gaussian["a0_gaussian2_filtered1"]
    a0_gaussian2_filtered2 = data_gaussian["a0_gaussian2_filtered2"]

    csMM_gaussian21 = cssr.MeasurementMatrices(a0_gaussian2_filtered1, number_samples)
    csMM_gaussian22 = cssr.MeasurementMatrices(a0_gaussian2_filtered2, number_samples)


    ar_gaussian21_gauss = csMM_gaussian21.random_gauss_matrix()
    ar_gaussian21_bernoulli = csMM_gaussian21.random_bernoulli_matrix(probability=0.5)
    ar_gaussian21_partial_fourier = csMM_gaussian21.random_partial_fourier_matrix()
    ar_gaussian21_partial_dct = csMM_gaussian21.random_partial_dct_matrix()
    ar_gaussian21_toeplitz = csMM_gaussian21.random_toeplitz_matrix()
    ar_gaussian21_binary_block = csMM_gaussian21.binary_block()
    ar_gaussian21_sgn = csMM_gaussian21.random_sgn_matrix()
    ar_gaussian21_gdo = csMM_gaussian21.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian22_gdo = csMM_gaussian22.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian21_gdo_adaptive = csMM_gaussian21.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian22_gdo_adaptive = csMM_gaussian22.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian21_ajs = csMM_gaussian21.ajs(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian21_afms = csMM_gaussian21.afms(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian22_afms = csMM_gaussian22.afms(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian21_hblz = csMM_gaussian21.hblz(l=l, p=p, rtol_estimate=False)
    ar_gaussian22_hblz = csMM_gaussian22.hblz(l=l, p=p, rtol_estimate=False)
    ar_gaussian21_ycwg = csMM_gaussian21.ycwg(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian22_ycwg = csMM_gaussian22.ycwg(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian21_xsfz = csMM_gaussian21.xsfz(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian22_xsfz = csMM_gaussian22.xsfz(max_iter=max_iter, rtol_estimate=False)


# Overcomplete dicionaries:
# .........................

if set_measurment_marices_gaussian_overcomplete:
    data_gaussian = load_filter_data(dir_filters, "gaussian_frame")

    a0_gaussian_overcomplete2_filtered1 = data_gaussian["a0_gaussian_overcomplete2_filtered1"]
    a0_gaussian_overcomplete2_filtered2 = data_gaussian["a0_gaussian_overcomplete2_filtered2"]

    csMM_gaussian_overcomplete21 = cssr.MeasurementMatrices(a0_gaussian_overcomplete2_filtered1, number_samples)
    csMM_gaussian_overcomplete22 = cssr.MeasurementMatrices(a0_gaussian_overcomplete2_filtered2, number_samples)

    ar_gaussian_overcomplete21_gauss = csMM_gaussian_overcomplete21.random_gauss_matrix()
    ar_gaussian_overcomplete21_bernoulli = csMM_gaussian_overcomplete21.random_bernoulli_matrix(probability=0.5)
    ar_gaussian_overcomplete21_partial_fourier = csMM_gaussian_overcomplete21.random_partial_fourier_matrix()
    ar_gaussian_overcomplete21_partial_dct = csMM_gaussian_overcomplete21.random_partial_dct_matrix()
    ar_gaussian_overcomplete21_toeplitz = csMM_gaussian_overcomplete21.random_toeplitz_matrix()
    ar_gaussian_overcomplete21_binary_block = csMM_gaussian_overcomplete21.binary_block()
    ar_gaussian_overcomplete21_sgn = csMM_gaussian_overcomplete21.random_sgn_matrix()
    ar_gaussian_overcomplete21_gdo = csMM_gaussian_overcomplete21.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian_overcomplete22_gdo = csMM_gaussian_overcomplete22.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian_overcomplete21_gdo_adaptive = csMM_gaussian_overcomplete21.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian_overcomplete22_gdo_adaptive = csMM_gaussian_overcomplete22.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian_overcomplete21_ajs = csMM_gaussian_overcomplete21.ajs(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian_overcomplete21_afms = csMM_gaussian_overcomplete21.afms(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian_overcomplete22_afms = csMM_gaussian_overcomplete22.afms(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian_overcomplete21_hblz = csMM_gaussian_overcomplete21.hblz(l=l, p=p, rtol_estimate=False)
    ar_gaussian_overcomplete22_hblz = csMM_gaussian_overcomplete22.hblz(l=l, p=p, rtol_estimate=False)
    ar_gaussian_overcomplete21_ycwg = csMM_gaussian_overcomplete21.ycwg(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian_overcomplete22_ycwg = csMM_gaussian_overcomplete22.ycwg(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian_overcomplete21_xsfz = csMM_gaussian_overcomplete21.xsfz(max_iter=max_iter, rtol_estimate=False)
    ar_gaussian_overcomplete22_xsfz = csMM_gaussian_overcomplete22.xsfz(max_iter=max_iter, rtol_estimate=False)


if save_measurment_marices_gaussian and set_measurment_marices_gaussian and set_measurment_marices_gaussian_overcomplete:
    np.savez(dir_measurement_matrices + "gaussian",
             probability = 0.5,

             ar_gaussian21_gauss = ar_gaussian21_gauss,
             ar_gaussian21_bernoulli = ar_gaussian21_bernoulli,
             ar_gaussian21_partial_fourier = ar_gaussian21_partial_fourier,
             ar_gaussian21_partial_dct = ar_gaussian21_partial_dct,
             ar_gaussian21_toeplitz = ar_gaussian21_toeplitz,
             ar_gaussian21_binary_block = ar_gaussian21_binary_block,
             ar_gaussian21_sgn = ar_gaussian21_sgn,
             ar_gaussian21_gdo = ar_gaussian21_gdo,
             ar_gaussian22_gdo = ar_gaussian22_gdo,
             ar_gaussian21_gdo_adaptive = ar_gaussian21_gdo_adaptive,
             ar_gaussian22_gdo_adaptive = ar_gaussian22_gdo_adaptive,
             ar_gaussian21_ajs = ar_gaussian21_ajs,
             ar_gaussian21_afms = ar_gaussian21_afms,
             ar_gaussian22_afms = ar_gaussian22_afms,
             ar_gaussian21_hblz = ar_gaussian21_hblz,
             ar_gaussian22_hblz = ar_gaussian22_hblz,
             ar_gaussian21_ycwg = ar_gaussian21_ycwg,
             ar_gaussian22_ycwg = ar_gaussian22_ycwg,
             ar_gaussian21_xsfz = ar_gaussian21_xsfz,
             ar_gaussian22_xsfz = ar_gaussian22_xsfz,

             ar_gaussian_overcomplete21_gauss = ar_gaussian_overcomplete21_gauss,
             ar_gaussian_overcomplete21_bernoulli = ar_gaussian_overcomplete21_bernoulli,
             ar_gaussian_overcomplete21_partial_fourier = ar_gaussian_overcomplete21_partial_fourier,
             ar_gaussian_overcomplete21_partial_dct = ar_gaussian_overcomplete21_partial_dct,
             ar_gaussian_overcomplete21_toeplitz = ar_gaussian_overcomplete21_toeplitz,
             ar_gaussian_overcomplete21_binary_block = ar_gaussian_overcomplete21_binary_block,
             ar_gaussian_overcomplete21_sgn = ar_gaussian_overcomplete21_sgn,
             ar_gaussian_overcomplete21_gdo = ar_gaussian_overcomplete21_gdo,
             ar_gaussian_overcomplete22_gdo = ar_gaussian_overcomplete22_gdo,
             ar_gaussian_overcomplete21_gdo_adaptive = ar_gaussian_overcomplete21_gdo_adaptive,
             ar_gaussian_overcomplete22_gdo_adaptive = ar_gaussian_overcomplete22_gdo_adaptive,
             ar_gaussian_overcomplete21_ajs = ar_gaussian_overcomplete21_ajs,
             ar_gaussian_overcomplete21_afms = ar_gaussian_overcomplete21_afms,
             ar_gaussian_overcomplete22_afms = ar_gaussian_overcomplete22_afms,
             ar_gaussian_overcomplete21_hblz = ar_gaussian_overcomplete21_hblz,
             ar_gaussian_overcomplete22_hblz = ar_gaussian_overcomplete22_hblz,
             ar_gaussian_overcomplete21_ycwg = ar_gaussian_overcomplete21_ycwg,
             ar_gaussian_overcomplete22_ycwg = ar_gaussian_overcomplete22_ycwg,
             ar_gaussian_overcomplete21_xsfz = ar_gaussian_overcomplete21_xsfz,
             ar_gaussian_overcomplete22_xsfz = ar_gaussian_overcomplete22_xsfz

             )


# =============================================================================