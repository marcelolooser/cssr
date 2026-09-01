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
set_measurment_marices_heaviside = True
set_measurment_marices_heaviside_overcomplete = True
save_measurment_marices_heaviside = True

set_measurment_marices_gaussian = True
set_measurment_marices_gaussian_overcomplete = True
save_measurment_marices_gaussian = True

set_measurment_marices_cauchy = True
set_measurment_marices_cauchy_overcomplete = True
save_measurment_marices_cauchy = True

set_measurment_marices_fourier = True
save_measurment_marices_fourier = True


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

# Heaviside:
# ----------

# Stub:
# .....

data_stub_heaviside = load_stub_data(dir_stubs, "heaviside")
shapes_heaviside = data_stub_heaviside["tracked_shapes_heaviside"]
shapes_heaviside = [tuple(item.tolist()) for item in shapes_heaviside]

np.random.random = stub_random_random_from_data(
    {
     shapes_heaviside[0] : data_stub_heaviside["uniform_ns_m_heaviside1"],
     shapes_heaviside[1] : data_stub_heaviside["uniform_ns_m_heaviside2"],
     shapes_heaviside[2] : data_stub_heaviside["uniform_ns_m_heaviside3"],

     shapes_heaviside[3] : data_stub_heaviside["uniform_ns_n_heaviside1"],
     shapes_heaviside[4] : data_stub_heaviside["uniform_ns_n_heaviside2"],
     shapes_heaviside[5] : data_stub_heaviside["uniform_ns_n_heaviside3"],

     shapes_heaviside[6] : data_stub_heaviside["uniform_ns_m_heaviside_overcomplete1"],
     shapes_heaviside[7] : data_stub_heaviside["uniform_ns_m_heaviside_overcomplete2"],
     shapes_heaviside[8] : data_stub_heaviside["uniform_ns_m_heaviside_overcomplete3"],

     shapes_heaviside[9] : data_stub_heaviside["uniform_ns_n_heaviside_overcomplete1"],
     shapes_heaviside[10] : data_stub_heaviside["uniform_ns_n_heaviside_overcomplete2"],
     shapes_heaviside[11] : data_stub_heaviside["uniform_ns_n_heaviside_overcomplete3"]
     })


probability = 0.5
np.random.binomial = stub_random_binomial_from_data(
    {
     (1, probability, shapes_heaviside[12]) : data_stub_heaviside["bernoulli_ns_m_heaviside1"],
     (1, probability, shapes_heaviside[13]) : data_stub_heaviside["bernoulli_ns_m_heaviside2"],
     (1, probability, shapes_heaviside[14]) : data_stub_heaviside["bernoulli_ns_m_heaviside3"],

     (1, probability, shapes_heaviside[15]) : data_stub_heaviside["bernoulli_ns_m_heaviside_overcomplete1"],
     (1, probability, shapes_heaviside[16]) : data_stub_heaviside["bernoulli_ns_m_heaviside_overcomplete2"],
     (1, probability, shapes_heaviside[17]) : data_stub_heaviside["bernoulli_ns_m_heaviside_overcomplete3"]
     })


random.sample = stub_random_sample_from_data(
    {
     (range(shapes_heaviside[18][0]), shapes_heaviside[18][1]) : data_stub_heaviside["samples_m_ns_heaviside1"],
     (range(shapes_heaviside[19][0]), shapes_heaviside[19][1]) : data_stub_heaviside["samples_m_ns_heaviside2"],
     (range(shapes_heaviside[20][0]), shapes_heaviside[20][1]) : data_stub_heaviside["samples_m_ns_heaviside3"],

     (range(shapes_heaviside[21][0]), shapes_heaviside[21][1]) : data_stub_heaviside["samples_n_m_heaviside1"],
     (range(shapes_heaviside[22][0]), shapes_heaviside[22][1]) : data_stub_heaviside["samples_n_m_heaviside2"],
     (range(shapes_heaviside[23][0]), shapes_heaviside[23][1]) : data_stub_heaviside["samples_n_m_heaviside3"],

     (range(shapes_heaviside[24][0]), shapes_heaviside[24][1]) : data_stub_heaviside["samples_m_ns_heaviside_overcomplete1"],
     (range(shapes_heaviside[25][0]), shapes_heaviside[25][1]) : data_stub_heaviside["samples_m_ns_heaviside_overcomplete2"],
     (range(shapes_heaviside[26][0]), shapes_heaviside[26][1]) : data_stub_heaviside["samples_m_ns_heaviside_overcomplete3"],

     (range(shapes_heaviside[27][0]), shapes_heaviside[27][1]) : data_stub_heaviside["samples_n_m_heaviside_overcomplete1"],
     (range(shapes_heaviside[28][0]), shapes_heaviside[28][1]) : data_stub_heaviside["samples_n_m_heaviside_overcomplete2"],
     (range(shapes_heaviside[29][0]), shapes_heaviside[29][1]) : data_stub_heaviside["samples_n_m_heaviside_overcomplete3"]
     })


choices = (-1,1)
np.random.choice = stub_random_choice_from_data(
    {
     (choices, shapes_heaviside[30][0]) : data_stub_heaviside["samples_binary_m_heaviside1"],
     (choices, shapes_heaviside[31][0]) : data_stub_heaviside["samples_binary_m_heaviside2"],
     (choices, shapes_heaviside[32][0]) : data_stub_heaviside["samples_binary_m_heaviside3"],

     (choices, shapes_heaviside[33][0]) : data_stub_heaviside["samples_binary_m_heaviside_overcomplete1"],
     (choices, shapes_heaviside[34][0]) : data_stub_heaviside["samples_binary_m_heaviside_overcomplete2"],
     (choices, shapes_heaviside[35][0]) : data_stub_heaviside["samples_binary_m_heaviside_overcomplete3"]
     })


np.random.randn = stub_random_randn_from_data(
    {
     shapes_heaviside[36] : data_stub_heaviside["gaussian_ns_m_heaviside1"],
     shapes_heaviside[37] : data_stub_heaviside["gaussian_ns_m_heaviside2"],
     shapes_heaviside[38] : data_stub_heaviside["gaussian_ns_m_heaviside3"],

     shapes_heaviside[39] : data_stub_heaviside["gaussian_ns_m_heaviside_overcomplete1"],
     shapes_heaviside[40] : data_stub_heaviside["gaussian_ns_m_heaviside_overcomplete2"],
     shapes_heaviside[41] : data_stub_heaviside["gaussian_ns_m_heaviside_overcomplete3"]
     })

scipy.stats.unitary_group.rvs = stub_random_unitary_from_data(
    {
     shapes_heaviside[42][0] : data_stub_heaviside["random_unitary_ns_m_heaviside"]
     })


# Dicionaries:
# ............

if set_measurment_marices_heaviside:
    data_heaviside = load_filter_data(dir_filters, "heaviside_frame")

    a0_heaviside1_filtered1 = data_heaviside["a0_heaviside1_filtered1"]
    a0_heaviside2_filtered1 = data_heaviside["a0_heaviside2_filtered1"]
    a0_heaviside3_filtered1 = data_heaviside["a0_heaviside3_filtered1"]

    a0_heaviside1_filtered2 = data_heaviside["a0_heaviside1_filtered2"]
    a0_heaviside2_filtered2 = data_heaviside["a0_heaviside2_filtered2"]
    a0_heaviside3_filtered2 = data_heaviside["a0_heaviside3_filtered2"]

    csMM_heaviside11 = cssr.MeasurementMatrices(a0_heaviside1_filtered1, number_samples)
    csMM_heaviside21 = cssr.MeasurementMatrices(a0_heaviside2_filtered1, number_samples)
    csMM_heaviside31 = cssr.MeasurementMatrices(a0_heaviside3_filtered1, number_samples)

    csMM_heaviside12 = cssr.MeasurementMatrices(a0_heaviside1_filtered2, number_samples)
    csMM_heaviside22 = cssr.MeasurementMatrices(a0_heaviside2_filtered2, number_samples)
    csMM_heaviside32 = cssr.MeasurementMatrices(a0_heaviside3_filtered2, number_samples)


    ar_heaviside11_gauss = csMM_heaviside11.random_gauss_matrix()
    ar_heaviside21_gauss = csMM_heaviside21.random_gauss_matrix()
    ar_heaviside31_gauss = csMM_heaviside31.random_gauss_matrix()

    ar_heaviside11_bernoulli = csMM_heaviside11.random_bernoulli_matrix(probability=0.5)
    ar_heaviside21_bernoulli = csMM_heaviside21.random_bernoulli_matrix(probability=0.5)
    ar_heaviside31_bernoulli = csMM_heaviside31.random_bernoulli_matrix(probability=0.5)

    ar_heaviside11_partial_fourier = csMM_heaviside11.random_partial_fourier_matrix()
    ar_heaviside21_partial_fourier = csMM_heaviside21.random_partial_fourier_matrix()
    ar_heaviside31_partial_fourier = csMM_heaviside31.random_partial_fourier_matrix()

    ar_heaviside11_partial_dct = csMM_heaviside11.random_partial_dct_matrix()
    ar_heaviside21_partial_dct = csMM_heaviside21.random_partial_dct_matrix()
    ar_heaviside31_partial_dct = csMM_heaviside31.random_partial_dct_matrix()

    ar_heaviside11_toeplitz = csMM_heaviside11.random_toeplitz_matrix()
    ar_heaviside21_toeplitz = csMM_heaviside21.random_toeplitz_matrix()
    ar_heaviside31_toeplitz = csMM_heaviside31.random_toeplitz_matrix()

    ar_heaviside11_binary_block = csMM_heaviside11.binary_block()
    ar_heaviside21_binary_block = csMM_heaviside21.binary_block()
    ar_heaviside31_binary_block = csMM_heaviside31.binary_block()

    ar_heaviside11_sgn = csMM_heaviside11.random_sgn_matrix()
    ar_heaviside21_sgn = csMM_heaviside21.random_sgn_matrix()
    ar_heaviside31_sgn = csMM_heaviside31.random_sgn_matrix()

    ar_heaviside11_gdo = csMM_heaviside11.gdo_measurement_matrix(l=l, p=p)
    ar_heaviside21_gdo = csMM_heaviside21.gdo_measurement_matrix(l=l, p=p)
    ar_heaviside31_gdo = csMM_heaviside31.gdo_measurement_matrix(l=l, p=p)

    ar_heaviside12_gdo = csMM_heaviside12.gdo_measurement_matrix(l=l, p=p)
    ar_heaviside22_gdo = csMM_heaviside22.gdo_measurement_matrix(l=l, p=p)
    ar_heaviside32_gdo = csMM_heaviside32.gdo_measurement_matrix(l=l, p=p)

    ar_heaviside11_gdo_adaptive = csMM_heaviside11.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_heaviside21_gdo_adaptive = csMM_heaviside21.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_heaviside31_gdo_adaptive = csMM_heaviside31.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_heaviside12_gdo_adaptive = csMM_heaviside12.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_heaviside22_gdo_adaptive = csMM_heaviside22.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_heaviside32_gdo_adaptive = csMM_heaviside32.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_heaviside11_ajs = csMM_heaviside11.ajs(max_iter=max_iter)
    ar_heaviside21_ajs = csMM_heaviside21.ajs(max_iter=max_iter)
    ar_heaviside31_ajs = csMM_heaviside31.ajs(max_iter=max_iter)

    ar_heaviside11_afms = csMM_heaviside11.afms(max_iter=max_iter)
    ar_heaviside21_afms = csMM_heaviside21.afms(max_iter=max_iter)
    ar_heaviside31_afms = csMM_heaviside31.afms(max_iter=max_iter)

    ar_heaviside12_afms = csMM_heaviside12.afms(max_iter=max_iter)
    ar_heaviside22_afms = csMM_heaviside22.afms(max_iter=max_iter)
    ar_heaviside32_afms = csMM_heaviside32.afms(max_iter=max_iter)

    ar_heaviside11_hblz = csMM_heaviside11.hblz(l=l, p=p)
    ar_heaviside21_hblz = csMM_heaviside21.hblz(l=l, p=p)
    ar_heaviside31_hblz = csMM_heaviside31.hblz(l=l, p=p)

    ar_heaviside12_hblz = csMM_heaviside12.hblz(l=l, p=p)
    ar_heaviside22_hblz = csMM_heaviside22.hblz(l=l, p=p)
    ar_heaviside32_hblz = csMM_heaviside32.hblz(l=l, p=p)

    ar_heaviside11_ycwg = csMM_heaviside11.ycwg(max_iter=max_iter)
    ar_heaviside21_ycwg = csMM_heaviside21.ycwg(max_iter=max_iter)
    ar_heaviside31_ycwg = csMM_heaviside31.ycwg(max_iter=max_iter)

    ar_heaviside12_ycwg = csMM_heaviside12.ycwg(max_iter=max_iter)
    ar_heaviside22_ycwg = csMM_heaviside22.ycwg(max_iter=max_iter)
    ar_heaviside32_ycwg = csMM_heaviside32.ycwg(max_iter=max_iter)

    ar_heaviside11_xsfz = csMM_heaviside11.xsfz(max_iter=max_iter)
    ar_heaviside21_xsfz = csMM_heaviside21.xsfz(max_iter=max_iter)
    ar_heaviside31_xsfz = csMM_heaviside31.xsfz(max_iter=max_iter)

    ar_heaviside12_xsfz = csMM_heaviside12.xsfz(max_iter=max_iter)
    ar_heaviside22_xsfz = csMM_heaviside22.xsfz(max_iter=max_iter)
    ar_heaviside32_xsfz = csMM_heaviside32.xsfz(max_iter=max_iter)



# Overcomplete dicionaries:
# .........................

if set_measurment_marices_heaviside_overcomplete:
    data_heaviside = load_filter_data(dir_filters, "heaviside_frame")

    a0_heaviside_overcomplete1_filtered1 = data_heaviside["a0_heaviside_overcomplete1_filtered1"]
    a0_heaviside_overcomplete2_filtered1 = data_heaviside["a0_heaviside_overcomplete2_filtered1"]
    a0_heaviside_overcomplete3_filtered1 = data_heaviside["a0_heaviside_overcomplete3_filtered1"]

    a0_heaviside_overcomplete1_filtered2 = data_heaviside["a0_heaviside_overcomplete1_filtered2"]
    a0_heaviside_overcomplete2_filtered2 = data_heaviside["a0_heaviside_overcomplete2_filtered2"]
    a0_heaviside_overcomplete3_filtered2 = data_heaviside["a0_heaviside_overcomplete3_filtered2"]

    csMM_heaviside_overcomplete11 = cssr.MeasurementMatrices(a0_heaviside_overcomplete1_filtered1, number_samples)
    csMM_heaviside_overcomplete21 = cssr.MeasurementMatrices(a0_heaviside_overcomplete2_filtered1, number_samples)
    csMM_heaviside_overcomplete31 = cssr.MeasurementMatrices(a0_heaviside_overcomplete3_filtered1, number_samples)

    csMM_heaviside_overcomplete12 = cssr.MeasurementMatrices(a0_heaviside_overcomplete1_filtered2, number_samples)
    csMM_heaviside_overcomplete22 = cssr.MeasurementMatrices(a0_heaviside_overcomplete2_filtered2, number_samples)
    csMM_heaviside_overcomplete32 = cssr.MeasurementMatrices(a0_heaviside_overcomplete3_filtered2, number_samples)

    ar_heaviside_overcomplete11_gauss = csMM_heaviside_overcomplete11.random_gauss_matrix()
    ar_heaviside_overcomplete21_gauss = csMM_heaviside_overcomplete21.random_gauss_matrix()
    ar_heaviside_overcomplete31_gauss = csMM_heaviside_overcomplete31.random_gauss_matrix()

    ar_heaviside_overcomplete11_bernoulli = csMM_heaviside_overcomplete11.random_bernoulli_matrix(probability=0.5)
    ar_heaviside_overcomplete21_bernoulli = csMM_heaviside_overcomplete21.random_bernoulli_matrix(probability=0.5)
    ar_heaviside_overcomplete31_bernoulli = csMM_heaviside_overcomplete31.random_bernoulli_matrix(probability=0.5)

    ar_heaviside_overcomplete11_partial_fourier = csMM_heaviside_overcomplete11.random_partial_fourier_matrix()
    ar_heaviside_overcomplete21_partial_fourier = csMM_heaviside_overcomplete21.random_partial_fourier_matrix()
    ar_heaviside_overcomplete31_partial_fourier = csMM_heaviside_overcomplete31.random_partial_fourier_matrix()

    ar_heaviside_overcomplete11_partial_dct = csMM_heaviside_overcomplete11.random_partial_dct_matrix()
    ar_heaviside_overcomplete21_partial_dct = csMM_heaviside_overcomplete21.random_partial_dct_matrix()
    ar_heaviside_overcomplete31_partial_dct = csMM_heaviside_overcomplete31.random_partial_dct_matrix()

    ar_heaviside_overcomplete11_toeplitz = csMM_heaviside_overcomplete11.random_toeplitz_matrix()
    ar_heaviside_overcomplete21_toeplitz = csMM_heaviside_overcomplete21.random_toeplitz_matrix()
    ar_heaviside_overcomplete31_toeplitz = csMM_heaviside_overcomplete31.random_toeplitz_matrix()

    ar_heaviside_overcomplete11_binary_block = csMM_heaviside_overcomplete11.binary_block()
    ar_heaviside_overcomplete21_binary_block = csMM_heaviside_overcomplete21.binary_block()
    ar_heaviside_overcomplete31_binary_block = csMM_heaviside_overcomplete31.binary_block()

    ar_heaviside_overcomplete11_sgn = csMM_heaviside_overcomplete11.random_sgn_matrix()
    ar_heaviside_overcomplete21_sgn = csMM_heaviside_overcomplete21.random_sgn_matrix()
    ar_heaviside_overcomplete31_sgn = csMM_heaviside_overcomplete31.random_sgn_matrix()

    ar_heaviside_overcomplete11_gdo = csMM_heaviside_overcomplete11.gdo_measurement_matrix(l=l, p=p)
    ar_heaviside_overcomplete21_gdo = csMM_heaviside_overcomplete21.gdo_measurement_matrix(l=l, p=p)
    ar_heaviside_overcomplete31_gdo = csMM_heaviside_overcomplete31.gdo_measurement_matrix(l=l, p=p)

    ar_heaviside_overcomplete12_gdo = csMM_heaviside_overcomplete12.gdo_measurement_matrix(l=l, p=p)
    ar_heaviside_overcomplete22_gdo = csMM_heaviside_overcomplete22.gdo_measurement_matrix(l=l, p=p)
    ar_heaviside_overcomplete32_gdo = csMM_heaviside_overcomplete32.gdo_measurement_matrix(l=l, p=p)

    ar_heaviside_overcomplete11_gdo_adaptive = csMM_heaviside_overcomplete11.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_heaviside_overcomplete21_gdo_adaptive = csMM_heaviside_overcomplete21.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_heaviside_overcomplete31_gdo_adaptive = csMM_heaviside_overcomplete31.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_heaviside_overcomplete12_gdo_adaptive = csMM_heaviside_overcomplete12.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_heaviside_overcomplete22_gdo_adaptive = csMM_heaviside_overcomplete22.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_heaviside_overcomplete32_gdo_adaptive = csMM_heaviside_overcomplete32.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_heaviside_overcomplete11_ajs = csMM_heaviside_overcomplete11.ajs(max_iter=max_iter)
    ar_heaviside_overcomplete21_ajs = csMM_heaviside_overcomplete21.ajs(max_iter=max_iter)
    ar_heaviside_overcomplete31_ajs = csMM_heaviside_overcomplete31.ajs(max_iter=max_iter)

    ar_heaviside_overcomplete11_afms = csMM_heaviside_overcomplete11.afms(max_iter=max_iter)
    ar_heaviside_overcomplete21_afms = csMM_heaviside_overcomplete21.afms(max_iter=max_iter)
    ar_heaviside_overcomplete31_afms = csMM_heaviside_overcomplete31.afms(max_iter=max_iter)

    ar_heaviside_overcomplete12_afms = csMM_heaviside_overcomplete12.afms(max_iter=max_iter)
    ar_heaviside_overcomplete22_afms = csMM_heaviside_overcomplete22.afms(max_iter=max_iter)
    ar_heaviside_overcomplete32_afms = csMM_heaviside_overcomplete32.afms(max_iter=max_iter)

    ar_heaviside_overcomplete11_hblz = csMM_heaviside_overcomplete11.hblz(l=l, p=p)
    ar_heaviside_overcomplete21_hblz = csMM_heaviside_overcomplete21.hblz(l=l, p=p)
    ar_heaviside_overcomplete31_hblz = csMM_heaviside_overcomplete31.hblz(l=l, p=p)

    ar_heaviside_overcomplete12_hblz = csMM_heaviside_overcomplete12.hblz(l=l, p=p)
    ar_heaviside_overcomplete22_hblz = csMM_heaviside_overcomplete22.hblz(l=l, p=p)
    ar_heaviside_overcomplete32_hblz = csMM_heaviside_overcomplete32.hblz(l=l, p=p)

    ar_heaviside_overcomplete11_ycwg = csMM_heaviside_overcomplete11.ycwg(max_iter=max_iter)
    ar_heaviside_overcomplete21_ycwg = csMM_heaviside_overcomplete21.ycwg(max_iter=max_iter)
    ar_heaviside_overcomplete31_ycwg = csMM_heaviside_overcomplete31.ycwg(max_iter=max_iter)

    ar_heaviside_overcomplete12_ycwg = csMM_heaviside_overcomplete12.ycwg(max_iter=max_iter)
    ar_heaviside_overcomplete22_ycwg = csMM_heaviside_overcomplete22.ycwg(max_iter=max_iter)
    ar_heaviside_overcomplete32_ycwg = csMM_heaviside_overcomplete32.ycwg(max_iter=max_iter)

    ar_heaviside_overcomplete11_xsfz = csMM_heaviside_overcomplete11.xsfz(max_iter=max_iter)
    ar_heaviside_overcomplete21_xsfz = csMM_heaviside_overcomplete21.xsfz(max_iter=max_iter)
    ar_heaviside_overcomplete31_xsfz = csMM_heaviside_overcomplete31.xsfz(max_iter=max_iter)

    ar_heaviside_overcomplete12_xsfz = csMM_heaviside_overcomplete12.xsfz(max_iter=max_iter)
    ar_heaviside_overcomplete22_xsfz = csMM_heaviside_overcomplete22.xsfz(max_iter=max_iter)
    ar_heaviside_overcomplete32_xsfz = csMM_heaviside_overcomplete32.xsfz(max_iter=max_iter)


if save_measurment_marices_heaviside and set_measurment_marices_heaviside and set_measurment_marices_heaviside_overcomplete:
    np.savez(dir_measurement_matrices + "heaviside",
             probability = 0.5,

             ar_heaviside11_gauss = ar_heaviside11_gauss,
             ar_heaviside21_gauss = ar_heaviside21_gauss,
             ar_heaviside31_gauss = ar_heaviside31_gauss,

             ar_heaviside11_bernoulli = ar_heaviside11_bernoulli,
             ar_heaviside21_bernoulli = ar_heaviside21_bernoulli,
             ar_heaviside31_bernoulli = ar_heaviside31_bernoulli,

             ar_heaviside11_partial_fourier = ar_heaviside11_partial_fourier,
             ar_heaviside21_partial_fourier = ar_heaviside21_partial_fourier,
             ar_heaviside31_partial_fourier = ar_heaviside31_partial_fourier,

             ar_heaviside11_partial_dct = ar_heaviside11_partial_dct,
             ar_heaviside21_partial_dct = ar_heaviside21_partial_dct,
             ar_heaviside31_partial_dct = ar_heaviside31_partial_dct,

             ar_heaviside11_toeplitz = ar_heaviside11_toeplitz,
             ar_heaviside21_toeplitz = ar_heaviside21_toeplitz,
             ar_heaviside31_toeplitz = ar_heaviside31_toeplitz,

             ar_heaviside11_binary_block = ar_heaviside11_binary_block,
             ar_heaviside21_binary_block = ar_heaviside21_binary_block,
             ar_heaviside31_binary_block = ar_heaviside31_binary_block,

             ar_heaviside11_sgn = ar_heaviside11_sgn,
             ar_heaviside21_sgn = ar_heaviside21_sgn,
             ar_heaviside31_sgn = ar_heaviside31_sgn,

             ar_heaviside11_gdo = ar_heaviside11_gdo,
             ar_heaviside21_gdo = ar_heaviside21_gdo,
             ar_heaviside31_gdo = ar_heaviside31_gdo,

             ar_heaviside12_gdo = ar_heaviside12_gdo,
             ar_heaviside22_gdo = ar_heaviside22_gdo,
             ar_heaviside32_gdo = ar_heaviside32_gdo,

             ar_heaviside11_gdo_adaptive = ar_heaviside11_gdo_adaptive,
             ar_heaviside21_gdo_adaptive = ar_heaviside21_gdo_adaptive,
             ar_heaviside31_gdo_adaptive = ar_heaviside31_gdo_adaptive,

             ar_heaviside12_gdo_adaptive = ar_heaviside12_gdo_adaptive,
             ar_heaviside22_gdo_adaptive = ar_heaviside22_gdo_adaptive,
             ar_heaviside32_gdo_adaptive = ar_heaviside32_gdo_adaptive,

             ar_heaviside11_ajs = ar_heaviside11_ajs,
             ar_heaviside21_ajs = ar_heaviside21_ajs,
             ar_heaviside31_ajs = ar_heaviside31_ajs,

             ar_heaviside11_afms = ar_heaviside11_afms,
             ar_heaviside21_afms = ar_heaviside21_afms,
             ar_heaviside31_afms = ar_heaviside31_afms,

             ar_heaviside12_afms = ar_heaviside12_afms,
             ar_heaviside22_afms = ar_heaviside22_afms,
             ar_heaviside32_afms = ar_heaviside32_afms,

             ar_heaviside11_hblz = ar_heaviside11_hblz,
             ar_heaviside21_hblz = ar_heaviside21_hblz,
             ar_heaviside31_hblz = ar_heaviside31_hblz,

             ar_heaviside12_hblz = ar_heaviside12_hblz,
             ar_heaviside22_hblz = ar_heaviside22_hblz,
             ar_heaviside32_hblz = ar_heaviside32_hblz,

             ar_heaviside11_ycwg = ar_heaviside11_ycwg,
             ar_heaviside21_ycwg = ar_heaviside21_ycwg,
             ar_heaviside31_ycwg = ar_heaviside31_ycwg,

             ar_heaviside12_ycwg = ar_heaviside12_ycwg,
             ar_heaviside22_ycwg = ar_heaviside22_ycwg,
             ar_heaviside32_ycwg = ar_heaviside32_ycwg,

             ar_heaviside11_xsfz = ar_heaviside11_xsfz,
             ar_heaviside21_xsfz = ar_heaviside21_xsfz,
             ar_heaviside31_xsfz = ar_heaviside31_xsfz,

             ar_heaviside12_xsfz = ar_heaviside12_xsfz,
             ar_heaviside22_xsfz = ar_heaviside22_xsfz,
             ar_heaviside32_xsfz = ar_heaviside32_xsfz,


             ar_heaviside_overcomplete11_gauss = ar_heaviside_overcomplete11_gauss,
             ar_heaviside_overcomplete21_gauss = ar_heaviside_overcomplete21_gauss,
             ar_heaviside_overcomplete31_gauss = ar_heaviside_overcomplete31_gauss,

             ar_heaviside_overcomplete11_bernoulli = ar_heaviside_overcomplete11_bernoulli,
             ar_heaviside_overcomplete21_bernoulli = ar_heaviside_overcomplete21_bernoulli,
             ar_heaviside_overcomplete31_bernoulli = ar_heaviside_overcomplete31_bernoulli,

             ar_heaviside_overcomplete11_partial_fourier = ar_heaviside_overcomplete11_partial_fourier,
             ar_heaviside_overcomplete21_partial_fourier = ar_heaviside_overcomplete21_partial_fourier,
             ar_heaviside_overcomplete31_partial_fourier = ar_heaviside_overcomplete31_partial_fourier,

             ar_heaviside_overcomplete11_partial_dct = ar_heaviside_overcomplete11_partial_dct,
             ar_heaviside_overcomplete21_partial_dct = ar_heaviside_overcomplete21_partial_dct,
             ar_heaviside_overcomplete31_partial_dct = ar_heaviside_overcomplete31_partial_dct,

             ar_heaviside_overcomplete11_toeplitz = ar_heaviside_overcomplete11_toeplitz,
             ar_heaviside_overcomplete21_toeplitz = ar_heaviside_overcomplete21_toeplitz,
             ar_heaviside_overcomplete31_toeplitz = ar_heaviside_overcomplete31_toeplitz,

             ar_heaviside_overcomplete11_binary_block = ar_heaviside_overcomplete11_binary_block,
             ar_heaviside_overcomplete21_binary_block = ar_heaviside_overcomplete21_binary_block,
             ar_heaviside_overcomplete31_binary_block = ar_heaviside_overcomplete31_binary_block,

             ar_heaviside_overcomplete11_sgn = ar_heaviside_overcomplete11_sgn,
             ar_heaviside_overcomplete21_sgn = ar_heaviside_overcomplete21_sgn,
             ar_heaviside_overcomplete31_sgn = ar_heaviside_overcomplete31_sgn,

             ar_heaviside_overcomplete11_gdo = ar_heaviside_overcomplete11_gdo,
             ar_heaviside_overcomplete21_gdo = ar_heaviside_overcomplete21_gdo,
             ar_heaviside_overcomplete31_gdo = ar_heaviside_overcomplete31_gdo,

             ar_heaviside_overcomplete12_gdo = ar_heaviside_overcomplete12_gdo,
             ar_heaviside_overcomplete22_gdo = ar_heaviside_overcomplete22_gdo,
             ar_heaviside_overcomplete32_gdo = ar_heaviside_overcomplete32_gdo,

             ar_heaviside_overcomplete11_gdo_adaptive = ar_heaviside_overcomplete11_gdo_adaptive,
             ar_heaviside_overcomplete21_gdo_adaptive = ar_heaviside_overcomplete21_gdo_adaptive,
             ar_heaviside_overcomplete31_gdo_adaptive = ar_heaviside_overcomplete31_gdo_adaptive,

             ar_heaviside_overcomplete12_gdo_adaptive = ar_heaviside_overcomplete12_gdo_adaptive,
             ar_heaviside_overcomplete22_gdo_adaptive = ar_heaviside_overcomplete22_gdo_adaptive,
             ar_heaviside_overcomplete32_gdo_adaptive = ar_heaviside_overcomplete32_gdo_adaptive,

             ar_heaviside_overcomplete11_ajs = ar_heaviside_overcomplete11_ajs,
             ar_heaviside_overcomplete21_ajs = ar_heaviside_overcomplete21_ajs,
             ar_heaviside_overcomplete31_ajs = ar_heaviside_overcomplete31_ajs,

             ar_heaviside_overcomplete11_afms = ar_heaviside_overcomplete11_afms,
             ar_heaviside_overcomplete21_afms = ar_heaviside_overcomplete21_afms,
             ar_heaviside_overcomplete31_afms = ar_heaviside_overcomplete31_afms,

             ar_heaviside_overcomplete12_afms = ar_heaviside_overcomplete12_afms,
             ar_heaviside_overcomplete22_afms = ar_heaviside_overcomplete22_afms,
             ar_heaviside_overcomplete32_afms = ar_heaviside_overcomplete32_afms,

             ar_heaviside_overcomplete11_hblz = ar_heaviside_overcomplete11_hblz,
             ar_heaviside_overcomplete21_hblz = ar_heaviside_overcomplete21_hblz,
             ar_heaviside_overcomplete31_hblz = ar_heaviside_overcomplete31_hblz,

             ar_heaviside_overcomplete12_hblz = ar_heaviside_overcomplete12_hblz,
             ar_heaviside_overcomplete22_hblz = ar_heaviside_overcomplete22_hblz,
             ar_heaviside_overcomplete32_hblz = ar_heaviside_overcomplete32_hblz,

             ar_heaviside_overcomplete11_ycwg = ar_heaviside_overcomplete11_ycwg,
             ar_heaviside_overcomplete21_ycwg = ar_heaviside_overcomplete21_ycwg,
             ar_heaviside_overcomplete31_ycwg = ar_heaviside_overcomplete31_ycwg,

             ar_heaviside_overcomplete12_ycwg = ar_heaviside_overcomplete12_ycwg,
             ar_heaviside_overcomplete22_ycwg = ar_heaviside_overcomplete22_ycwg,
             ar_heaviside_overcomplete32_ycwg = ar_heaviside_overcomplete32_ycwg,

             ar_heaviside_overcomplete11_xsfz = ar_heaviside_overcomplete11_xsfz,
             ar_heaviside_overcomplete21_xsfz = ar_heaviside_overcomplete21_xsfz,
             ar_heaviside_overcomplete31_xsfz = ar_heaviside_overcomplete31_xsfz,

             ar_heaviside_overcomplete12_xsfz = ar_heaviside_overcomplete12_xsfz,
             ar_heaviside_overcomplete22_xsfz = ar_heaviside_overcomplete22_xsfz,
             ar_heaviside_overcomplete32_xsfz = ar_heaviside_overcomplete32_xsfz

             )


# Gaussian:
# ---------

data_stub_gaussian = load_stub_data(dir_stubs, "gaussian")
shapes_gaussian = data_stub_gaussian["tracked_shapes_gaussian"]
shapes_gaussian = [tuple(item.tolist()) for item in shapes_gaussian]

np.random.random = stub_random_random_from_data(
    {
     shapes_gaussian[0] : data_stub_gaussian["uniform_ns_m_gaussian1"],
     shapes_gaussian[1] : data_stub_gaussian["uniform_ns_m_gaussian2"],
     shapes_gaussian[2] : data_stub_gaussian["uniform_ns_m_gaussian3"],

     shapes_gaussian[3] : data_stub_gaussian["uniform_ns_n_gaussian1"],
     shapes_gaussian[4] : data_stub_gaussian["uniform_ns_n_gaussian2"],
     shapes_gaussian[5] : data_stub_gaussian["uniform_ns_n_gaussian3"],

     shapes_gaussian[6] : data_stub_gaussian["uniform_ns_m_gaussian_overcomplete1"],
     shapes_gaussian[7] : data_stub_gaussian["uniform_ns_m_gaussian_overcomplete2"],
     shapes_gaussian[8] : data_stub_gaussian["uniform_ns_m_gaussian_overcomplete3"],

     shapes_gaussian[9] : data_stub_gaussian["uniform_ns_n_gaussian_overcomplete1"],
     shapes_gaussian[10] : data_stub_gaussian["uniform_ns_n_gaussian_overcomplete2"],
     shapes_gaussian[11] : data_stub_gaussian["uniform_ns_n_gaussian_overcomplete3"]
     })


probability = 0.5
np.random.binomial = stub_random_binomial_from_data(
    {
     (1, probability, shapes_gaussian[12]) : data_stub_gaussian["bernoulli_ns_m_gaussian1"],
     (1, probability, shapes_gaussian[13]) : data_stub_gaussian["bernoulli_ns_m_gaussian2"],
     (1, probability, shapes_gaussian[14]) : data_stub_gaussian["bernoulli_ns_m_gaussian3"],

     (1, probability, shapes_gaussian[15]) : data_stub_gaussian["bernoulli_ns_m_gaussian_overcomplete1"],
     (1, probability, shapes_gaussian[16]) : data_stub_gaussian["bernoulli_ns_m_gaussian_overcomplete2"],
     (1, probability, shapes_gaussian[17]) : data_stub_gaussian["bernoulli_ns_m_gaussian_overcomplete3"]
     })


random.sample = stub_random_sample_from_data(
    {
     (range(shapes_gaussian[18][0]), shapes_gaussian[18][1]) : data_stub_gaussian["samples_m_ns_gaussian1"],
     (range(shapes_gaussian[19][0]), shapes_gaussian[19][1]) : data_stub_gaussian["samples_m_ns_gaussian2"],
     (range(shapes_gaussian[20][0]), shapes_gaussian[20][1]) : data_stub_gaussian["samples_m_ns_gaussian3"],

     (range(shapes_gaussian[21][0]), shapes_gaussian[21][1]) : data_stub_gaussian["samples_n_m_gaussian1"],
     (range(shapes_gaussian[22][0]), shapes_gaussian[22][1]) : data_stub_gaussian["samples_n_m_gaussian2"],
     (range(shapes_gaussian[23][0]), shapes_gaussian[23][1]) : data_stub_gaussian["samples_n_m_gaussian3"],

     (range(shapes_gaussian[24][0]), shapes_gaussian[24][1]) : data_stub_gaussian["samples_m_ns_gaussian_overcomplete1"],
     (range(shapes_gaussian[25][0]), shapes_gaussian[25][1]) : data_stub_gaussian["samples_m_ns_gaussian_overcomplete2"],
     (range(shapes_gaussian[26][0]), shapes_gaussian[26][1]) : data_stub_gaussian["samples_m_ns_gaussian_overcomplete3"],

     (range(shapes_gaussian[27][0]), shapes_gaussian[27][1]) : data_stub_gaussian["samples_n_m_gaussian_overcomplete1"],
     (range(shapes_gaussian[28][0]), shapes_gaussian[28][1]) : data_stub_gaussian["samples_n_m_gaussian_overcomplete2"],
     (range(shapes_gaussian[29][0]), shapes_gaussian[29][1]) : data_stub_gaussian["samples_n_m_gaussian_overcomplete3"]
     })


choices = (-1,1)
np.random.choice = stub_random_choice_from_data(
    {
     (choices, shapes_gaussian[30][0]) : data_stub_gaussian["samples_binary_m_gaussian1"],
     (choices, shapes_gaussian[31][0]) : data_stub_gaussian["samples_binary_m_gaussian2"],
     (choices, shapes_gaussian[32][0]) : data_stub_gaussian["samples_binary_m_gaussian3"],

     (choices, shapes_gaussian[33][0]) : data_stub_gaussian["samples_binary_m_gaussian_overcomplete1"],
     (choices, shapes_gaussian[34][0]) : data_stub_gaussian["samples_binary_m_gaussian_overcomplete2"],
     (choices, shapes_gaussian[35][0]) : data_stub_gaussian["samples_binary_m_gaussian_overcomplete3"]
     })


np.random.randn = stub_random_randn_from_data(
    {
     shapes_gaussian[36] : data_stub_gaussian["gaussian_ns_m_gaussian1"],
     shapes_gaussian[37] : data_stub_gaussian["gaussian_ns_m_gaussian2"],
     shapes_gaussian[38] : data_stub_gaussian["gaussian_ns_m_gaussian3"],

     shapes_gaussian[39] : data_stub_gaussian["gaussian_ns_m_gaussian_overcomplete1"],
     shapes_gaussian[40] : data_stub_gaussian["gaussian_ns_m_gaussian_overcomplete2"],
     shapes_gaussian[41] : data_stub_gaussian["gaussian_ns_m_gaussian_overcomplete3"]
     })


scipy.stats.unitary_group.rvs = stub_random_unitary_from_data(
    {
     shapes_gaussian[42][0] : data_stub_gaussian["random_unitary_ns_m_gaussian"]
     })



# Dicionaries:
# ............

if set_measurment_marices_gaussian:
    data_gaussian = load_filter_data(dir_filters, "gaussian_frame")

    a0_gaussian1_filtered1 = data_gaussian["a0_gaussian1_filtered1"]
    a0_gaussian2_filtered1 = data_gaussian["a0_gaussian2_filtered1"]
    a0_gaussian3_filtered1 = data_gaussian["a0_gaussian3_filtered1"]

    a0_gaussian1_filtered2 = data_gaussian["a0_gaussian1_filtered2"]
    a0_gaussian2_filtered2 = data_gaussian["a0_gaussian2_filtered2"]
    a0_gaussian3_filtered2 = data_gaussian["a0_gaussian3_filtered2"]

    csMM_gaussian11 = cssr.MeasurementMatrices(a0_gaussian1_filtered1, number_samples)
    csMM_gaussian21 = cssr.MeasurementMatrices(a0_gaussian2_filtered1, number_samples)
    csMM_gaussian31 = cssr.MeasurementMatrices(a0_gaussian3_filtered1, number_samples)

    csMM_gaussian12 = cssr.MeasurementMatrices(a0_gaussian1_filtered2, number_samples)
    csMM_gaussian22 = cssr.MeasurementMatrices(a0_gaussian2_filtered2, number_samples)
    csMM_gaussian32 = cssr.MeasurementMatrices(a0_gaussian3_filtered2, number_samples)


    ar_gaussian11_gauss = csMM_gaussian11.random_gauss_matrix()
    ar_gaussian21_gauss = csMM_gaussian21.random_gauss_matrix()
    ar_gaussian31_gauss = csMM_gaussian31.random_gauss_matrix()

    ar_gaussian11_bernoulli = csMM_gaussian11.random_bernoulli_matrix(probability=0.5)
    ar_gaussian21_bernoulli = csMM_gaussian21.random_bernoulli_matrix(probability=0.5)
    ar_gaussian31_bernoulli = csMM_gaussian31.random_bernoulli_matrix(probability=0.5)

    ar_gaussian11_partial_fourier = csMM_gaussian11.random_partial_fourier_matrix()
    ar_gaussian21_partial_fourier = csMM_gaussian21.random_partial_fourier_matrix()
    ar_gaussian31_partial_fourier = csMM_gaussian31.random_partial_fourier_matrix()

    ar_gaussian11_partial_dct = csMM_gaussian11.random_partial_dct_matrix()
    ar_gaussian21_partial_dct = csMM_gaussian21.random_partial_dct_matrix()
    ar_gaussian31_partial_dct = csMM_gaussian31.random_partial_dct_matrix()

    ar_gaussian11_toeplitz = csMM_gaussian11.random_toeplitz_matrix()
    ar_gaussian21_toeplitz = csMM_gaussian21.random_toeplitz_matrix()
    ar_gaussian31_toeplitz = csMM_gaussian31.random_toeplitz_matrix()

    ar_gaussian11_binary_block = csMM_gaussian11.binary_block()
    ar_gaussian21_binary_block = csMM_gaussian21.binary_block()
    ar_gaussian31_binary_block = csMM_gaussian31.binary_block()

    ar_gaussian11_sgn = csMM_gaussian11.random_sgn_matrix()
    ar_gaussian21_sgn = csMM_gaussian21.random_sgn_matrix()
    ar_gaussian31_sgn = csMM_gaussian31.random_sgn_matrix()

    ar_gaussian11_gdo = csMM_gaussian11.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian21_gdo = csMM_gaussian21.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian31_gdo = csMM_gaussian31.gdo_measurement_matrix(l=l, p=p)

    ar_gaussian12_gdo = csMM_gaussian12.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian22_gdo = csMM_gaussian22.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian32_gdo = csMM_gaussian32.gdo_measurement_matrix(l=l, p=p)

    ar_gaussian11_gdo_adaptive = csMM_gaussian11.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian21_gdo_adaptive = csMM_gaussian21.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian31_gdo_adaptive = csMM_gaussian31.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_gaussian12_gdo_adaptive = csMM_gaussian12.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian22_gdo_adaptive = csMM_gaussian22.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian32_gdo_adaptive = csMM_gaussian32.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_gaussian11_ajs = csMM_gaussian11.ajs(max_iter=max_iter)
    ar_gaussian21_ajs = csMM_gaussian21.ajs(max_iter=max_iter)
    ar_gaussian31_ajs = csMM_gaussian31.ajs(max_iter=max_iter)

    ar_gaussian11_afms = csMM_gaussian11.afms(max_iter=max_iter)
    ar_gaussian21_afms = csMM_gaussian21.afms(max_iter=max_iter)
    ar_gaussian31_afms = csMM_gaussian31.afms(max_iter=max_iter)

    ar_gaussian12_afms = csMM_gaussian12.afms(max_iter=max_iter)
    ar_gaussian22_afms = csMM_gaussian22.afms(max_iter=max_iter)
    ar_gaussian32_afms = csMM_gaussian32.afms(max_iter=max_iter)

    ar_gaussian11_hblz = csMM_gaussian11.hblz(l=l, p=p)
    ar_gaussian21_hblz = csMM_gaussian21.hblz(l=l, p=p)
    ar_gaussian31_hblz = csMM_gaussian31.hblz(l=l, p=p)

    ar_gaussian12_hblz = csMM_gaussian12.hblz(l=l, p=p)
    ar_gaussian22_hblz = csMM_gaussian22.hblz(l=l, p=p)
    ar_gaussian32_hblz = csMM_gaussian32.hblz(l=l, p=p)

    ar_gaussian11_ycwg = csMM_gaussian11.ycwg(max_iter=max_iter)
    ar_gaussian21_ycwg = csMM_gaussian21.ycwg(max_iter=max_iter)
    ar_gaussian31_ycwg = csMM_gaussian31.ycwg(max_iter=max_iter)

    ar_gaussian12_ycwg = csMM_gaussian12.ycwg(max_iter=max_iter)
    ar_gaussian22_ycwg = csMM_gaussian22.ycwg(max_iter=max_iter)
    ar_gaussian32_ycwg = csMM_gaussian32.ycwg(max_iter=max_iter)

    ar_gaussian11_xsfz = csMM_gaussian11.xsfz(max_iter=max_iter)
    ar_gaussian21_xsfz = csMM_gaussian21.xsfz(max_iter=max_iter)
    ar_gaussian31_xsfz = csMM_gaussian31.xsfz(max_iter=max_iter)

    ar_gaussian12_xsfz = csMM_gaussian12.xsfz(max_iter=max_iter)
    ar_gaussian22_xsfz = csMM_gaussian22.xsfz(max_iter=max_iter)
    ar_gaussian32_xsfz = csMM_gaussian32.xsfz(max_iter=max_iter)


# Overcomplete dicionaries:
# .........................

if set_measurment_marices_gaussian_overcomplete:
    data_gaussian = load_filter_data(dir_filters, "gaussian_frame")

    a0_gaussian_overcomplete1_filtered1 = data_gaussian["a0_gaussian_overcomplete1_filtered1"]
    a0_gaussian_overcomplete2_filtered1 = data_gaussian["a0_gaussian_overcomplete2_filtered1"]
    a0_gaussian_overcomplete3_filtered1 = data_gaussian["a0_gaussian_overcomplete3_filtered1"]

    a0_gaussian_overcomplete1_filtered2 = data_gaussian["a0_gaussian_overcomplete1_filtered2"]
    a0_gaussian_overcomplete2_filtered2 = data_gaussian["a0_gaussian_overcomplete2_filtered2"]
    a0_gaussian_overcomplete3_filtered2 = data_gaussian["a0_gaussian_overcomplete3_filtered2"]

    csMM_gaussian_overcomplete11 = cssr.MeasurementMatrices(a0_gaussian_overcomplete1_filtered1, number_samples)
    csMM_gaussian_overcomplete21 = cssr.MeasurementMatrices(a0_gaussian_overcomplete2_filtered1, number_samples)
    csMM_gaussian_overcomplete31 = cssr.MeasurementMatrices(a0_gaussian_overcomplete3_filtered1, number_samples)

    csMM_gaussian_overcomplete12 = cssr.MeasurementMatrices(a0_gaussian_overcomplete1_filtered2, number_samples)
    csMM_gaussian_overcomplete22 = cssr.MeasurementMatrices(a0_gaussian_overcomplete2_filtered2, number_samples)
    csMM_gaussian_overcomplete32 = cssr.MeasurementMatrices(a0_gaussian_overcomplete3_filtered2, number_samples)

    ar_gaussian_overcomplete11_gauss = csMM_gaussian_overcomplete11.random_gauss_matrix()
    ar_gaussian_overcomplete21_gauss = csMM_gaussian_overcomplete21.random_gauss_matrix()
    ar_gaussian_overcomplete31_gauss = csMM_gaussian_overcomplete31.random_gauss_matrix()

    ar_gaussian_overcomplete11_bernoulli = csMM_gaussian_overcomplete11.random_bernoulli_matrix(probability=0.5)
    ar_gaussian_overcomplete21_bernoulli = csMM_gaussian_overcomplete21.random_bernoulli_matrix(probability=0.5)
    ar_gaussian_overcomplete31_bernoulli = csMM_gaussian_overcomplete31.random_bernoulli_matrix(probability=0.5)

    ar_gaussian_overcomplete11_partial_fourier = csMM_gaussian_overcomplete11.random_partial_fourier_matrix()
    ar_gaussian_overcomplete21_partial_fourier = csMM_gaussian_overcomplete21.random_partial_fourier_matrix()
    ar_gaussian_overcomplete31_partial_fourier = csMM_gaussian_overcomplete31.random_partial_fourier_matrix()

    ar_gaussian_overcomplete11_partial_dct = csMM_gaussian_overcomplete11.random_partial_dct_matrix()
    ar_gaussian_overcomplete21_partial_dct = csMM_gaussian_overcomplete21.random_partial_dct_matrix()
    ar_gaussian_overcomplete31_partial_dct = csMM_gaussian_overcomplete31.random_partial_dct_matrix()

    ar_gaussian_overcomplete11_toeplitz = csMM_gaussian_overcomplete11.random_toeplitz_matrix()
    ar_gaussian_overcomplete21_toeplitz = csMM_gaussian_overcomplete21.random_toeplitz_matrix()
    ar_gaussian_overcomplete31_toeplitz = csMM_gaussian_overcomplete31.random_toeplitz_matrix()

    ar_gaussian_overcomplete11_binary_block = csMM_gaussian_overcomplete11.binary_block()
    ar_gaussian_overcomplete21_binary_block = csMM_gaussian_overcomplete21.binary_block()
    ar_gaussian_overcomplete31_binary_block = csMM_gaussian_overcomplete31.binary_block()

    ar_gaussian_overcomplete11_sgn = csMM_gaussian_overcomplete11.random_sgn_matrix()
    ar_gaussian_overcomplete21_sgn = csMM_gaussian_overcomplete21.random_sgn_matrix()
    ar_gaussian_overcomplete31_sgn = csMM_gaussian_overcomplete31.random_sgn_matrix()

    ar_gaussian_overcomplete11_gdo = csMM_gaussian_overcomplete11.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian_overcomplete21_gdo = csMM_gaussian_overcomplete21.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian_overcomplete31_gdo = csMM_gaussian_overcomplete31.gdo_measurement_matrix(l=l, p=p)

    ar_gaussian_overcomplete12_gdo = csMM_gaussian_overcomplete12.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian_overcomplete22_gdo = csMM_gaussian_overcomplete22.gdo_measurement_matrix(l=l, p=p)
    ar_gaussian_overcomplete32_gdo = csMM_gaussian_overcomplete32.gdo_measurement_matrix(l=l, p=p)

    ar_gaussian_overcomplete11_gdo_adaptive = csMM_gaussian_overcomplete11.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian_overcomplete21_gdo_adaptive = csMM_gaussian_overcomplete21.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian_overcomplete31_gdo_adaptive = csMM_gaussian_overcomplete31.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_gaussian_overcomplete12_gdo_adaptive = csMM_gaussian_overcomplete12.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian_overcomplete22_gdo_adaptive = csMM_gaussian_overcomplete22.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_gaussian_overcomplete32_gdo_adaptive = csMM_gaussian_overcomplete32.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_gaussian_overcomplete11_ajs = csMM_gaussian_overcomplete11.ajs(max_iter=max_iter)
    ar_gaussian_overcomplete21_ajs = csMM_gaussian_overcomplete21.ajs(max_iter=max_iter)
    ar_gaussian_overcomplete31_ajs = csMM_gaussian_overcomplete31.ajs(max_iter=max_iter)

    ar_gaussian_overcomplete11_afms = csMM_gaussian_overcomplete11.afms(max_iter=max_iter)
    ar_gaussian_overcomplete21_afms = csMM_gaussian_overcomplete21.afms(max_iter=max_iter)
    ar_gaussian_overcomplete31_afms = csMM_gaussian_overcomplete31.afms(max_iter=max_iter)

    ar_gaussian_overcomplete12_afms = csMM_gaussian_overcomplete12.afms(max_iter=max_iter)
    ar_gaussian_overcomplete22_afms = csMM_gaussian_overcomplete22.afms(max_iter=max_iter)
    ar_gaussian_overcomplete32_afms = csMM_gaussian_overcomplete32.afms(max_iter=max_iter)

    ar_gaussian_overcomplete11_hblz = csMM_gaussian_overcomplete11.hblz(l=l, p=p)
    ar_gaussian_overcomplete21_hblz = csMM_gaussian_overcomplete21.hblz(l=l, p=p)
    ar_gaussian_overcomplete31_hblz = csMM_gaussian_overcomplete31.hblz(l=l, p=p)

    ar_gaussian_overcomplete12_hblz = csMM_gaussian_overcomplete12.hblz(l=l, p=p)
    ar_gaussian_overcomplete22_hblz = csMM_gaussian_overcomplete22.hblz(l=l, p=p)
    ar_gaussian_overcomplete32_hblz = csMM_gaussian_overcomplete32.hblz(l=l, p=p)

    ar_gaussian_overcomplete11_ycwg = csMM_gaussian_overcomplete11.ycwg(max_iter=max_iter)
    ar_gaussian_overcomplete21_ycwg = csMM_gaussian_overcomplete21.ycwg(max_iter=max_iter)
    ar_gaussian_overcomplete31_ycwg = csMM_gaussian_overcomplete31.ycwg(max_iter=max_iter)

    ar_gaussian_overcomplete12_ycwg = csMM_gaussian_overcomplete12.ycwg(max_iter=max_iter)
    ar_gaussian_overcomplete22_ycwg = csMM_gaussian_overcomplete22.ycwg(max_iter=max_iter)
    ar_gaussian_overcomplete32_ycwg = csMM_gaussian_overcomplete32.ycwg(max_iter=max_iter)

    ar_gaussian_overcomplete11_xsfz = csMM_gaussian_overcomplete11.xsfz(max_iter=max_iter)
    ar_gaussian_overcomplete21_xsfz = csMM_gaussian_overcomplete21.xsfz(max_iter=max_iter)
    ar_gaussian_overcomplete31_xsfz = csMM_gaussian_overcomplete31.xsfz(max_iter=max_iter)

    ar_gaussian_overcomplete12_xsfz = csMM_gaussian_overcomplete12.xsfz(max_iter=max_iter)
    ar_gaussian_overcomplete22_xsfz = csMM_gaussian_overcomplete22.xsfz(max_iter=max_iter)
    ar_gaussian_overcomplete32_xsfz = csMM_gaussian_overcomplete32.xsfz(max_iter=max_iter)


if save_measurment_marices_gaussian and set_measurment_marices_gaussian and set_measurment_marices_gaussian_overcomplete:
    np.savez(dir_measurement_matrices + "gaussian",
             probability = 0.5,

             ar_gaussian11_gauss = ar_gaussian11_gauss,
             ar_gaussian21_gauss = ar_gaussian21_gauss,
             ar_gaussian31_gauss = ar_gaussian31_gauss,

             ar_gaussian11_bernoulli = ar_gaussian11_bernoulli,
             ar_gaussian21_bernoulli = ar_gaussian21_bernoulli,
             ar_gaussian31_bernoulli = ar_gaussian31_bernoulli,

             ar_gaussian11_partial_fourier = ar_gaussian11_partial_fourier,
             ar_gaussian21_partial_fourier = ar_gaussian21_partial_fourier,
             ar_gaussian31_partial_fourier = ar_gaussian31_partial_fourier,

             ar_gaussian11_partial_dct = ar_gaussian11_partial_dct,
             ar_gaussian21_partial_dct = ar_gaussian21_partial_dct,
             ar_gaussian31_partial_dct = ar_gaussian31_partial_dct,

             ar_gaussian11_toeplitz = ar_gaussian11_toeplitz,
             ar_gaussian21_toeplitz = ar_gaussian21_toeplitz,
             ar_gaussian31_toeplitz = ar_gaussian31_toeplitz,

             ar_gaussian11_binary_block = ar_gaussian11_binary_block,
             ar_gaussian21_binary_block = ar_gaussian21_binary_block,
             ar_gaussian31_binary_block = ar_gaussian31_binary_block,

             ar_gaussian11_sgn = ar_gaussian11_sgn,
             ar_gaussian21_sgn = ar_gaussian21_sgn,
             ar_gaussian31_sgn = ar_gaussian31_sgn,

             ar_gaussian11_gdo = ar_gaussian11_gdo,
             ar_gaussian21_gdo = ar_gaussian21_gdo,
             ar_gaussian31_gdo = ar_gaussian31_gdo,

             ar_gaussian12_gdo = ar_gaussian12_gdo,
             ar_gaussian22_gdo = ar_gaussian22_gdo,
             ar_gaussian32_gdo = ar_gaussian32_gdo,

             ar_gaussian11_gdo_adaptive = ar_gaussian11_gdo_adaptive,
             ar_gaussian21_gdo_adaptive = ar_gaussian21_gdo_adaptive,
             ar_gaussian31_gdo_adaptive = ar_gaussian31_gdo_adaptive,

             ar_gaussian12_gdo_adaptive = ar_gaussian12_gdo_adaptive,
             ar_gaussian22_gdo_adaptive = ar_gaussian22_gdo_adaptive,
             ar_gaussian32_gdo_adaptive = ar_gaussian32_gdo_adaptive,

             ar_gaussian11_ajs = ar_gaussian11_ajs,
             ar_gaussian21_ajs = ar_gaussian21_ajs,
             ar_gaussian31_ajs = ar_gaussian31_ajs,

             ar_gaussian11_afms = ar_gaussian11_afms,
             ar_gaussian21_afms = ar_gaussian21_afms,
             ar_gaussian31_afms = ar_gaussian31_afms,

             ar_gaussian12_afms = ar_gaussian12_afms,
             ar_gaussian22_afms = ar_gaussian22_afms,
             ar_gaussian32_afms = ar_gaussian32_afms,

             ar_gaussian11_hblz = ar_gaussian11_hblz,
             ar_gaussian21_hblz = ar_gaussian21_hblz,
             ar_gaussian31_hblz = ar_gaussian31_hblz,

             ar_gaussian12_hblz = ar_gaussian12_hblz,
             ar_gaussian22_hblz = ar_gaussian22_hblz,
             ar_gaussian32_hblz = ar_gaussian32_hblz,

             ar_gaussian11_ycwg = ar_gaussian11_ycwg,
             ar_gaussian21_ycwg = ar_gaussian21_ycwg,
             ar_gaussian31_ycwg = ar_gaussian31_ycwg,

             ar_gaussian12_ycwg = ar_gaussian12_ycwg,
             ar_gaussian22_ycwg = ar_gaussian22_ycwg,
             ar_gaussian32_ycwg = ar_gaussian32_ycwg,

             ar_gaussian11_xsfz = ar_gaussian11_xsfz,
             ar_gaussian21_xsfz = ar_gaussian21_xsfz,
             ar_gaussian31_xsfz = ar_gaussian31_xsfz,

             ar_gaussian12_xsfz = ar_gaussian12_xsfz,
             ar_gaussian22_xsfz = ar_gaussian22_xsfz,
             ar_gaussian32_xsfz = ar_gaussian32_xsfz,


             ar_gaussian_overcomplete11_gauss = ar_gaussian_overcomplete11_gauss,
             ar_gaussian_overcomplete21_gauss = ar_gaussian_overcomplete21_gauss,
             ar_gaussian_overcomplete31_gauss = ar_gaussian_overcomplete31_gauss,

             ar_gaussian_overcomplete11_bernoulli = ar_gaussian_overcomplete11_bernoulli,
             ar_gaussian_overcomplete21_bernoulli = ar_gaussian_overcomplete21_bernoulli,
             ar_gaussian_overcomplete31_bernoulli = ar_gaussian_overcomplete31_bernoulli,

             ar_gaussian_overcomplete11_partial_fourier = ar_gaussian_overcomplete11_partial_fourier,
             ar_gaussian_overcomplete21_partial_fourier = ar_gaussian_overcomplete21_partial_fourier,
             ar_gaussian_overcomplete31_partial_fourier = ar_gaussian_overcomplete31_partial_fourier,

             ar_gaussian_overcomplete11_partial_dct = ar_gaussian_overcomplete11_partial_dct,
             ar_gaussian_overcomplete21_partial_dct = ar_gaussian_overcomplete21_partial_dct,
             ar_gaussian_overcomplete31_partial_dct = ar_gaussian_overcomplete31_partial_dct,

             ar_gaussian_overcomplete11_toeplitz = ar_gaussian_overcomplete11_toeplitz,
             ar_gaussian_overcomplete21_toeplitz = ar_gaussian_overcomplete21_toeplitz,
             ar_gaussian_overcomplete31_toeplitz = ar_gaussian_overcomplete31_toeplitz,

             ar_gaussian_overcomplete11_binary_block = ar_gaussian_overcomplete11_binary_block,
             ar_gaussian_overcomplete21_binary_block = ar_gaussian_overcomplete21_binary_block,
             ar_gaussian_overcomplete31_binary_block = ar_gaussian_overcomplete31_binary_block,

             ar_gaussian_overcomplete11_sgn = ar_gaussian_overcomplete11_sgn,
             ar_gaussian_overcomplete21_sgn = ar_gaussian_overcomplete21_sgn,
             ar_gaussian_overcomplete31_sgn = ar_gaussian_overcomplete31_sgn,

             ar_gaussian_overcomplete11_gdo = ar_gaussian_overcomplete11_gdo,
             ar_gaussian_overcomplete21_gdo = ar_gaussian_overcomplete21_gdo,
             ar_gaussian_overcomplete31_gdo = ar_gaussian_overcomplete31_gdo,

             ar_gaussian_overcomplete12_gdo = ar_gaussian_overcomplete12_gdo,
             ar_gaussian_overcomplete22_gdo = ar_gaussian_overcomplete22_gdo,
             ar_gaussian_overcomplete32_gdo = ar_gaussian_overcomplete32_gdo,

             ar_gaussian_overcomplete11_gdo_adaptive = ar_gaussian_overcomplete11_gdo_adaptive,
             ar_gaussian_overcomplete21_gdo_adaptive = ar_gaussian_overcomplete21_gdo_adaptive,
             ar_gaussian_overcomplete31_gdo_adaptive = ar_gaussian_overcomplete31_gdo_adaptive,

             ar_gaussian_overcomplete12_gdo_adaptive = ar_gaussian_overcomplete12_gdo_adaptive,
             ar_gaussian_overcomplete22_gdo_adaptive = ar_gaussian_overcomplete22_gdo_adaptive,
             ar_gaussian_overcomplete32_gdo_adaptive = ar_gaussian_overcomplete32_gdo_adaptive,

             ar_gaussian_overcomplete11_ajs = ar_gaussian_overcomplete11_ajs,
             ar_gaussian_overcomplete21_ajs = ar_gaussian_overcomplete21_ajs,
             ar_gaussian_overcomplete31_ajs = ar_gaussian_overcomplete31_ajs,

             ar_gaussian_overcomplete11_afms = ar_gaussian_overcomplete11_afms,
             ar_gaussian_overcomplete21_afms = ar_gaussian_overcomplete21_afms,
             ar_gaussian_overcomplete31_afms = ar_gaussian_overcomplete31_afms,

             ar_gaussian_overcomplete12_afms = ar_gaussian_overcomplete12_afms,
             ar_gaussian_overcomplete22_afms = ar_gaussian_overcomplete22_afms,
             ar_gaussian_overcomplete32_afms = ar_gaussian_overcomplete32_afms,

             ar_gaussian_overcomplete11_hblz = ar_gaussian_overcomplete11_hblz,
             ar_gaussian_overcomplete21_hblz = ar_gaussian_overcomplete21_hblz,
             ar_gaussian_overcomplete31_hblz = ar_gaussian_overcomplete31_hblz,

             ar_gaussian_overcomplete12_hblz = ar_gaussian_overcomplete12_hblz,
             ar_gaussian_overcomplete22_hblz = ar_gaussian_overcomplete22_hblz,
             ar_gaussian_overcomplete32_hblz = ar_gaussian_overcomplete32_hblz,

             ar_gaussian_overcomplete11_ycwg = ar_gaussian_overcomplete11_ycwg,
             ar_gaussian_overcomplete21_ycwg = ar_gaussian_overcomplete21_ycwg,
             ar_gaussian_overcomplete31_ycwg = ar_gaussian_overcomplete31_ycwg,

             ar_gaussian_overcomplete12_ycwg = ar_gaussian_overcomplete12_ycwg,
             ar_gaussian_overcomplete22_ycwg = ar_gaussian_overcomplete22_ycwg,
             ar_gaussian_overcomplete32_ycwg = ar_gaussian_overcomplete32_ycwg,

             ar_gaussian_overcomplete11_xsfz = ar_gaussian_overcomplete11_xsfz,
             ar_gaussian_overcomplete21_xsfz = ar_gaussian_overcomplete21_xsfz,
             ar_gaussian_overcomplete31_xsfz = ar_gaussian_overcomplete31_xsfz,

             ar_gaussian_overcomplete12_xsfz = ar_gaussian_overcomplete12_xsfz,
             ar_gaussian_overcomplete22_xsfz = ar_gaussian_overcomplete22_xsfz,
             ar_gaussian_overcomplete32_xsfz = ar_gaussian_overcomplete32_xsfz

             )


# Cauchy:
# -------

data_stub_cauchy = load_stub_data(dir_stubs, "cauchy")
shapes_cauchy = data_stub_cauchy["tracked_shapes_cauchy"]
shapes_cauchy = [tuple(item.tolist()) for item in shapes_cauchy]

np.random.random = stub_random_random_from_data(
    {
     shapes_cauchy[0] : data_stub_cauchy["uniform_ns_m_cauchy1"],
     shapes_cauchy[1] : data_stub_cauchy["uniform_ns_m_cauchy2"],
     shapes_cauchy[2] : data_stub_cauchy["uniform_ns_m_cauchy3"],

     shapes_cauchy[3] : data_stub_cauchy["uniform_ns_n_cauchy1"],
     shapes_cauchy[4] : data_stub_cauchy["uniform_ns_n_cauchy2"],
     shapes_cauchy[5] : data_stub_cauchy["uniform_ns_n_cauchy3"],

     shapes_cauchy[6] : data_stub_cauchy["uniform_ns_m_cauchy_overcomplete1"],
     shapes_cauchy[7] : data_stub_cauchy["uniform_ns_m_cauchy_overcomplete2"],
     shapes_cauchy[8] : data_stub_cauchy["uniform_ns_m_cauchy_overcomplete3"],

     shapes_cauchy[9] : data_stub_cauchy["uniform_ns_n_cauchy_overcomplete1"],
     shapes_cauchy[10] : data_stub_cauchy["uniform_ns_n_cauchy_overcomplete2"],
     shapes_cauchy[11] : data_stub_cauchy["uniform_ns_n_cauchy_overcomplete3"]
     })


probability = 0.5
np.random.binomial = stub_random_binomial_from_data(
    {
     (1, probability, shapes_cauchy[12]) : data_stub_cauchy["bernoulli_ns_m_cauchy1"],
     (1, probability, shapes_cauchy[13]) : data_stub_cauchy["bernoulli_ns_m_cauchy2"],
     (1, probability, shapes_cauchy[14]) : data_stub_cauchy["bernoulli_ns_m_cauchy3"],

     (1, probability, shapes_cauchy[15]) : data_stub_cauchy["bernoulli_ns_m_cauchy_overcomplete1"],
     (1, probability, shapes_cauchy[16]) : data_stub_cauchy["bernoulli_ns_m_cauchy_overcomplete2"],
     (1, probability, shapes_cauchy[17]) : data_stub_cauchy["bernoulli_ns_m_cauchy_overcomplete3"]
     })


random.sample = stub_random_sample_from_data(
    {
     (range(shapes_cauchy[18][0]), shapes_cauchy[18][1]) : data_stub_cauchy["samples_m_ns_cauchy1"],
     (range(shapes_cauchy[19][0]), shapes_cauchy[19][1]) : data_stub_cauchy["samples_m_ns_cauchy2"],
     (range(shapes_cauchy[20][0]), shapes_cauchy[20][1]) : data_stub_cauchy["samples_m_ns_cauchy3"],

     (range(shapes_cauchy[21][0]), shapes_cauchy[21][1]) : data_stub_cauchy["samples_n_m_cauchy1"],
     (range(shapes_cauchy[22][0]), shapes_cauchy[22][1]) : data_stub_cauchy["samples_n_m_cauchy2"],
     (range(shapes_cauchy[23][0]), shapes_cauchy[23][1]) : data_stub_cauchy["samples_n_m_cauchy3"],

     (range(shapes_cauchy[24][0]), shapes_cauchy[24][1]) : data_stub_cauchy["samples_m_ns_cauchy_overcomplete1"],
     (range(shapes_cauchy[25][0]), shapes_cauchy[25][1]) : data_stub_cauchy["samples_m_ns_cauchy_overcomplete2"],
     (range(shapes_cauchy[26][0]), shapes_cauchy[26][1]) : data_stub_cauchy["samples_m_ns_cauchy_overcomplete3"],

     (range(shapes_cauchy[27][0]), shapes_cauchy[27][1]) : data_stub_cauchy["samples_n_m_cauchy_overcomplete1"],
     (range(shapes_cauchy[28][0]), shapes_cauchy[28][1]) : data_stub_cauchy["samples_n_m_cauchy_overcomplete2"],
     (range(shapes_cauchy[29][0]), shapes_cauchy[29][1]) : data_stub_cauchy["samples_n_m_cauchy_overcomplete3"]
     })


choices = (-1,1)
np.random.choice = stub_random_choice_from_data(
    {
     (choices, shapes_cauchy[30][0]) : data_stub_cauchy["samples_binary_m_cauchy1"],
     (choices, shapes_cauchy[31][0]) : data_stub_cauchy["samples_binary_m_cauchy2"],
     (choices, shapes_cauchy[32][0]) : data_stub_cauchy["samples_binary_m_cauchy3"],

     (choices, shapes_cauchy[33][0]) : data_stub_cauchy["samples_binary_m_cauchy_overcomplete1"],
     (choices, shapes_cauchy[34][0]) : data_stub_cauchy["samples_binary_m_cauchy_overcomplete2"],
     (choices, shapes_cauchy[35][0]) : data_stub_cauchy["samples_binary_m_cauchy_overcomplete3"]
     })


np.random.randn = stub_random_randn_from_data(
    {
     shapes_cauchy[36] : data_stub_cauchy["gaussian_ns_m_cauchy1"],
     shapes_cauchy[37] : data_stub_cauchy["gaussian_ns_m_cauchy2"],
     shapes_cauchy[38] : data_stub_cauchy["gaussian_ns_m_cauchy3"],

     shapes_cauchy[39] : data_stub_cauchy["gaussian_ns_m_cauchy_overcomplete1"],
     shapes_cauchy[40] : data_stub_cauchy["gaussian_ns_m_cauchy_overcomplete2"],
     shapes_cauchy[41] : data_stub_cauchy["gaussian_ns_m_cauchy_overcomplete3"]
     })


scipy.stats.unitary_group.rvs = stub_random_unitary_from_data(
    {
     shapes_cauchy[42][0] : data_stub_cauchy["random_unitary_ns_m_cauchy"]
     })


# Dicionaries:
# ............
if set_measurment_marices_cauchy:
    data_cauchy = load_filter_data(dir_filters, "cauchy_frame")

    a0_cauchy1_filtered1 = data_cauchy["a0_cauchy1_filtered1"]
    a0_cauchy2_filtered1 = data_cauchy["a0_cauchy2_filtered1"]
    a0_cauchy3_filtered1 = data_cauchy["a0_cauchy3_filtered1"]

    a0_cauchy1_filtered2 = data_cauchy["a0_cauchy1_filtered2"]
    a0_cauchy2_filtered2 = data_cauchy["a0_cauchy2_filtered2"]
    a0_cauchy3_filtered2 = data_cauchy["a0_cauchy3_filtered2"]

    csMM_cauchy11 = cssr.MeasurementMatrices(a0_cauchy1_filtered1, number_samples)
    csMM_cauchy21 = cssr.MeasurementMatrices(a0_cauchy2_filtered1, number_samples)
    csMM_cauchy31 = cssr.MeasurementMatrices(a0_cauchy3_filtered1, number_samples)

    csMM_cauchy12 = cssr.MeasurementMatrices(a0_cauchy1_filtered2, number_samples)
    csMM_cauchy22 = cssr.MeasurementMatrices(a0_cauchy2_filtered2, number_samples)
    csMM_cauchy32 = cssr.MeasurementMatrices(a0_cauchy3_filtered2, number_samples)


    ar_cauchy11_gauss = csMM_cauchy11.random_gauss_matrix()
    ar_cauchy21_gauss = csMM_cauchy21.random_gauss_matrix()
    ar_cauchy31_gauss = csMM_cauchy31.random_gauss_matrix()

    ar_cauchy11_bernoulli = csMM_cauchy11.random_bernoulli_matrix(probability=0.5)
    ar_cauchy21_bernoulli = csMM_cauchy21.random_bernoulli_matrix(probability=0.5)
    ar_cauchy31_bernoulli = csMM_cauchy31.random_bernoulli_matrix(probability=0.5)

    ar_cauchy11_partial_fourier = csMM_cauchy11.random_partial_fourier_matrix()
    ar_cauchy21_partial_fourier = csMM_cauchy21.random_partial_fourier_matrix()
    ar_cauchy31_partial_fourier = csMM_cauchy31.random_partial_fourier_matrix()

    ar_cauchy11_partial_dct = csMM_cauchy11.random_partial_dct_matrix()
    ar_cauchy21_partial_dct = csMM_cauchy21.random_partial_dct_matrix()
    ar_cauchy31_partial_dct = csMM_cauchy31.random_partial_dct_matrix()

    ar_cauchy11_toeplitz = csMM_cauchy11.random_toeplitz_matrix()
    ar_cauchy21_toeplitz = csMM_cauchy21.random_toeplitz_matrix()
    ar_cauchy31_toeplitz = csMM_cauchy31.random_toeplitz_matrix()

    ar_cauchy11_binary_block = csMM_cauchy11.binary_block()
    ar_cauchy21_binary_block = csMM_cauchy21.binary_block()
    ar_cauchy31_binary_block = csMM_cauchy31.binary_block()

    ar_cauchy11_sgn = csMM_cauchy11.random_sgn_matrix()
    ar_cauchy21_sgn = csMM_cauchy21.random_sgn_matrix()
    ar_cauchy31_sgn = csMM_cauchy31.random_sgn_matrix()

    ar_cauchy11_gdo = csMM_cauchy11.gdo_measurement_matrix(l=l, p=p)
    ar_cauchy21_gdo = csMM_cauchy21.gdo_measurement_matrix(l=l, p=p)
    ar_cauchy31_gdo = csMM_cauchy31.gdo_measurement_matrix(l=l, p=p)

    ar_cauchy12_gdo = csMM_cauchy12.gdo_measurement_matrix(l=l, p=p)
    ar_cauchy22_gdo = csMM_cauchy22.gdo_measurement_matrix(l=l, p=p)
    ar_cauchy32_gdo = csMM_cauchy32.gdo_measurement_matrix(l=l, p=p)

    ar_cauchy11_gdo_adaptive = csMM_cauchy11.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_cauchy21_gdo_adaptive = csMM_cauchy21.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_cauchy31_gdo_adaptive = csMM_cauchy31.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_cauchy12_gdo_adaptive = csMM_cauchy12.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_cauchy22_gdo_adaptive = csMM_cauchy22.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_cauchy32_gdo_adaptive = csMM_cauchy32.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_cauchy11_ajs = csMM_cauchy11.ajs(max_iter=max_iter)
    ar_cauchy21_ajs = csMM_cauchy21.ajs(max_iter=max_iter)
    ar_cauchy31_ajs = csMM_cauchy31.ajs(max_iter=max_iter)

    ar_cauchy11_afms = csMM_cauchy11.afms(max_iter=max_iter)
    ar_cauchy21_afms = csMM_cauchy21.afms(max_iter=max_iter)
    ar_cauchy31_afms = csMM_cauchy31.afms(max_iter=max_iter)

    ar_cauchy12_afms = csMM_cauchy12.afms(max_iter=max_iter)
    ar_cauchy22_afms = csMM_cauchy22.afms(max_iter=max_iter)
    ar_cauchy32_afms = csMM_cauchy32.afms(max_iter=max_iter)

    ar_cauchy11_hblz = csMM_cauchy11.hblz(l=l, p=p)
    ar_cauchy21_hblz = csMM_cauchy21.hblz(l=l, p=p)
    ar_cauchy31_hblz = csMM_cauchy31.hblz(l=l, p=p)

    ar_cauchy12_hblz = csMM_cauchy12.hblz(l=l, p=p)
    ar_cauchy22_hblz = csMM_cauchy22.hblz(l=l, p=p)
    ar_cauchy32_hblz = csMM_cauchy32.hblz(l=l, p=p)

    ar_cauchy11_ycwg = csMM_cauchy11.ycwg(max_iter=max_iter)
    ar_cauchy21_ycwg = csMM_cauchy21.ycwg(max_iter=max_iter)
    ar_cauchy31_ycwg = csMM_cauchy31.ycwg(max_iter=max_iter)

    ar_cauchy12_ycwg = csMM_cauchy12.ycwg(max_iter=max_iter)
    ar_cauchy22_ycwg = csMM_cauchy22.ycwg(max_iter=max_iter)
    ar_cauchy32_ycwg = csMM_cauchy32.ycwg(max_iter=max_iter)

    ar_cauchy11_xsfz = csMM_cauchy11.xsfz(max_iter=max_iter)
    ar_cauchy21_xsfz = csMM_cauchy21.xsfz(max_iter=max_iter)
    ar_cauchy31_xsfz = csMM_cauchy31.xsfz(max_iter=max_iter)

    ar_cauchy12_xsfz = csMM_cauchy12.xsfz(max_iter=max_iter)
    ar_cauchy22_xsfz = csMM_cauchy22.xsfz(max_iter=max_iter)
    ar_cauchy32_xsfz = csMM_cauchy32.xsfz(max_iter=max_iter)


# Overcomplete dicionaries:
# .........................
if set_measurment_marices_cauchy_overcomplete:
    data_cauchy = load_filter_data(dir_filters, "cauchy_frame")

    a0_cauchy_overcomplete1_filtered1 = data_cauchy["a0_cauchy_overcomplete1_filtered1"]
    a0_cauchy_overcomplete2_filtered1 = data_cauchy["a0_cauchy_overcomplete2_filtered1"]
    a0_cauchy_overcomplete3_filtered1 = data_cauchy["a0_cauchy_overcomplete3_filtered1"]

    a0_cauchy_overcomplete1_filtered2 = data_cauchy["a0_cauchy_overcomplete1_filtered2"]
    a0_cauchy_overcomplete2_filtered2 = data_cauchy["a0_cauchy_overcomplete2_filtered2"]
    a0_cauchy_overcomplete3_filtered2 = data_cauchy["a0_cauchy_overcomplete3_filtered2"]

    csMM_cauchy_overcomplete11 = cssr.MeasurementMatrices(a0_cauchy_overcomplete1_filtered1, number_samples)
    csMM_cauchy_overcomplete21 = cssr.MeasurementMatrices(a0_cauchy_overcomplete2_filtered1, number_samples)
    csMM_cauchy_overcomplete31 = cssr.MeasurementMatrices(a0_cauchy_overcomplete3_filtered1, number_samples)

    csMM_cauchy_overcomplete12 = cssr.MeasurementMatrices(a0_cauchy_overcomplete1_filtered2, number_samples)
    csMM_cauchy_overcomplete22 = cssr.MeasurementMatrices(a0_cauchy_overcomplete2_filtered2, number_samples)
    csMM_cauchy_overcomplete32 = cssr.MeasurementMatrices(a0_cauchy_overcomplete3_filtered2, number_samples)

    ar_cauchy_overcomplete11_gauss = csMM_cauchy_overcomplete11.random_gauss_matrix()
    ar_cauchy_overcomplete21_gauss = csMM_cauchy_overcomplete21.random_gauss_matrix()
    ar_cauchy_overcomplete31_gauss = csMM_cauchy_overcomplete31.random_gauss_matrix()

    ar_cauchy_overcomplete11_bernoulli = csMM_cauchy_overcomplete11.random_bernoulli_matrix(probability=0.5)
    ar_cauchy_overcomplete21_bernoulli = csMM_cauchy_overcomplete21.random_bernoulli_matrix(probability=0.5)
    ar_cauchy_overcomplete31_bernoulli = csMM_cauchy_overcomplete31.random_bernoulli_matrix(probability=0.5)

    ar_cauchy_overcomplete11_partial_fourier = csMM_cauchy_overcomplete11.random_partial_fourier_matrix()
    ar_cauchy_overcomplete21_partial_fourier = csMM_cauchy_overcomplete21.random_partial_fourier_matrix()
    ar_cauchy_overcomplete31_partial_fourier = csMM_cauchy_overcomplete31.random_partial_fourier_matrix()

    ar_cauchy_overcomplete11_partial_dct = csMM_cauchy_overcomplete11.random_partial_dct_matrix()
    ar_cauchy_overcomplete21_partial_dct = csMM_cauchy_overcomplete21.random_partial_dct_matrix()
    ar_cauchy_overcomplete31_partial_dct = csMM_cauchy_overcomplete31.random_partial_dct_matrix()

    ar_cauchy_overcomplete11_toeplitz = csMM_cauchy_overcomplete11.random_toeplitz_matrix()
    ar_cauchy_overcomplete21_toeplitz = csMM_cauchy_overcomplete21.random_toeplitz_matrix()
    ar_cauchy_overcomplete31_toeplitz = csMM_cauchy_overcomplete31.random_toeplitz_matrix()

    ar_cauchy_overcomplete11_binary_block = csMM_cauchy_overcomplete11.binary_block()
    ar_cauchy_overcomplete21_binary_block = csMM_cauchy_overcomplete21.binary_block()
    ar_cauchy_overcomplete31_binary_block = csMM_cauchy_overcomplete31.binary_block()

    ar_cauchy_overcomplete11_sgn = csMM_cauchy_overcomplete11.random_sgn_matrix()
    ar_cauchy_overcomplete21_sgn = csMM_cauchy_overcomplete21.random_sgn_matrix()
    ar_cauchy_overcomplete31_sgn = csMM_cauchy_overcomplete31.random_sgn_matrix()

    ar_cauchy_overcomplete11_gdo = csMM_cauchy_overcomplete11.gdo_measurement_matrix(l=l, p=p)
    ar_cauchy_overcomplete21_gdo = csMM_cauchy_overcomplete21.gdo_measurement_matrix(l=l, p=p)
    ar_cauchy_overcomplete31_gdo = csMM_cauchy_overcomplete31.gdo_measurement_matrix(l=l, p=p)

    ar_cauchy_overcomplete12_gdo = csMM_cauchy_overcomplete12.gdo_measurement_matrix(l=l, p=p)
    ar_cauchy_overcomplete22_gdo = csMM_cauchy_overcomplete22.gdo_measurement_matrix(l=l, p=p)
    ar_cauchy_overcomplete32_gdo = csMM_cauchy_overcomplete32.gdo_measurement_matrix(l=l, p=p)

    ar_cauchy_overcomplete11_gdo_adaptive = csMM_cauchy_overcomplete11.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_cauchy_overcomplete21_gdo_adaptive = csMM_cauchy_overcomplete21.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_cauchy_overcomplete31_gdo_adaptive = csMM_cauchy_overcomplete31.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_cauchy_overcomplete12_gdo_adaptive = csMM_cauchy_overcomplete12.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_cauchy_overcomplete22_gdo_adaptive = csMM_cauchy_overcomplete22.gdo_measurement_matrix_adaptive(l=l, p=p)
    ar_cauchy_overcomplete32_gdo_adaptive = csMM_cauchy_overcomplete32.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_cauchy_overcomplete11_ajs = csMM_cauchy_overcomplete11.ajs(max_iter=max_iter)
    ar_cauchy_overcomplete21_ajs = csMM_cauchy_overcomplete21.ajs(max_iter=max_iter)
    ar_cauchy_overcomplete31_ajs = csMM_cauchy_overcomplete31.ajs(max_iter=max_iter)

    ar_cauchy_overcomplete11_afms = csMM_cauchy_overcomplete11.afms(max_iter=max_iter)
    ar_cauchy_overcomplete21_afms = csMM_cauchy_overcomplete21.afms(max_iter=max_iter)
    ar_cauchy_overcomplete31_afms = csMM_cauchy_overcomplete31.afms(max_iter=max_iter)

    ar_cauchy_overcomplete12_afms = csMM_cauchy_overcomplete12.afms(max_iter=max_iter)
    ar_cauchy_overcomplete22_afms = csMM_cauchy_overcomplete22.afms(max_iter=max_iter)
    ar_cauchy_overcomplete32_afms = csMM_cauchy_overcomplete32.afms(max_iter=max_iter)

    ar_cauchy_overcomplete11_hblz = csMM_cauchy_overcomplete11.hblz(l=l, p=p)
    ar_cauchy_overcomplete21_hblz = csMM_cauchy_overcomplete21.hblz(l=l, p=p)
    ar_cauchy_overcomplete31_hblz = csMM_cauchy_overcomplete31.hblz(l=l, p=p)

    ar_cauchy_overcomplete12_hblz = csMM_cauchy_overcomplete12.hblz(l=l, p=p)
    ar_cauchy_overcomplete22_hblz = csMM_cauchy_overcomplete22.hblz(l=l, p=p)
    ar_cauchy_overcomplete32_hblz = csMM_cauchy_overcomplete32.hblz(l=l, p=p)

    ar_cauchy_overcomplete11_ycwg = csMM_cauchy_overcomplete11.ycwg(max_iter=max_iter)
    ar_cauchy_overcomplete21_ycwg = csMM_cauchy_overcomplete21.ycwg(max_iter=max_iter)
    ar_cauchy_overcomplete31_ycwg = csMM_cauchy_overcomplete31.ycwg(max_iter=max_iter)

    ar_cauchy_overcomplete12_ycwg = csMM_cauchy_overcomplete12.ycwg(max_iter=max_iter)
    ar_cauchy_overcomplete22_ycwg = csMM_cauchy_overcomplete22.ycwg(max_iter=max_iter)
    ar_cauchy_overcomplete32_ycwg = csMM_cauchy_overcomplete32.ycwg(max_iter=max_iter)

    ar_cauchy_overcomplete11_xsfz = csMM_cauchy_overcomplete11.xsfz(max_iter=max_iter)
    ar_cauchy_overcomplete21_xsfz = csMM_cauchy_overcomplete21.xsfz(max_iter=max_iter)
    ar_cauchy_overcomplete31_xsfz = csMM_cauchy_overcomplete31.xsfz(max_iter=max_iter)

    ar_cauchy_overcomplete12_xsfz = csMM_cauchy_overcomplete12.xsfz(max_iter=max_iter)
    ar_cauchy_overcomplete22_xsfz = csMM_cauchy_overcomplete22.xsfz(max_iter=max_iter)
    ar_cauchy_overcomplete32_xsfz = csMM_cauchy_overcomplete32.xsfz(max_iter=max_iter)


if save_measurment_marices_cauchy and set_measurment_marices_cauchy and set_measurment_marices_cauchy_overcomplete:
    np.savez(dir_measurement_matrices + "cauchy",
             probability = 0.5,

             ar_cauchy11_gauss = ar_cauchy11_gauss,
             ar_cauchy21_gauss = ar_cauchy21_gauss,
             ar_cauchy31_gauss = ar_cauchy31_gauss,

             ar_cauchy11_bernoulli = ar_cauchy11_bernoulli,
             ar_cauchy21_bernoulli = ar_cauchy21_bernoulli,
             ar_cauchy31_bernoulli = ar_cauchy31_bernoulli,

             ar_cauchy11_partial_fourier = ar_cauchy11_partial_fourier,
             ar_cauchy21_partial_fourier = ar_cauchy21_partial_fourier,
             ar_cauchy31_partial_fourier = ar_cauchy31_partial_fourier,

             ar_cauchy11_partial_dct = ar_cauchy11_partial_dct,
             ar_cauchy21_partial_dct = ar_cauchy21_partial_dct,
             ar_cauchy31_partial_dct = ar_cauchy31_partial_dct,

             ar_cauchy11_toeplitz = ar_cauchy11_toeplitz,
             ar_cauchy21_toeplitz = ar_cauchy21_toeplitz,
             ar_cauchy31_toeplitz = ar_cauchy31_toeplitz,

             ar_cauchy11_binary_block = ar_cauchy11_binary_block,
             ar_cauchy21_binary_block = ar_cauchy21_binary_block,
             ar_cauchy31_binary_block = ar_cauchy31_binary_block,

             ar_cauchy11_sgn = ar_cauchy11_sgn,
             ar_cauchy21_sgn = ar_cauchy21_sgn,
             ar_cauchy31_sgn = ar_cauchy31_sgn,

             ar_cauchy11_gdo = ar_cauchy11_gdo,
             ar_cauchy21_gdo = ar_cauchy21_gdo,
             ar_cauchy31_gdo = ar_cauchy31_gdo,

             ar_cauchy12_gdo = ar_cauchy12_gdo,
             ar_cauchy22_gdo = ar_cauchy22_gdo,
             ar_cauchy32_gdo = ar_cauchy32_gdo,

             ar_cauchy11_gdo_adaptive = ar_cauchy11_gdo_adaptive,
             ar_cauchy21_gdo_adaptive = ar_cauchy21_gdo_adaptive,
             ar_cauchy31_gdo_adaptive = ar_cauchy31_gdo_adaptive,

             ar_cauchy12_gdo_adaptive = ar_cauchy12_gdo_adaptive,
             ar_cauchy22_gdo_adaptive = ar_cauchy22_gdo_adaptive,
             ar_cauchy32_gdo_adaptive = ar_cauchy32_gdo_adaptive,

             ar_cauchy11_ajs = ar_cauchy11_ajs,
             ar_cauchy21_ajs = ar_cauchy21_ajs,
             ar_cauchy31_ajs = ar_cauchy31_ajs,

             ar_cauchy11_afms = ar_cauchy11_afms,
             ar_cauchy21_afms = ar_cauchy21_afms,
             ar_cauchy31_afms = ar_cauchy31_afms,

             ar_cauchy12_afms = ar_cauchy12_afms,
             ar_cauchy22_afms = ar_cauchy22_afms,
             ar_cauchy32_afms = ar_cauchy32_afms,

             ar_cauchy11_hblz = ar_cauchy11_hblz,
             ar_cauchy21_hblz = ar_cauchy21_hblz,
             ar_cauchy31_hblz = ar_cauchy31_hblz,

             ar_cauchy12_hblz = ar_cauchy12_hblz,
             ar_cauchy22_hblz = ar_cauchy22_hblz,
             ar_cauchy32_hblz = ar_cauchy32_hblz,

             ar_cauchy11_ycwg = ar_cauchy11_ycwg,
             ar_cauchy21_ycwg = ar_cauchy21_ycwg,
             ar_cauchy31_ycwg = ar_cauchy31_ycwg,

             ar_cauchy12_ycwg = ar_cauchy12_ycwg,
             ar_cauchy22_ycwg = ar_cauchy22_ycwg,
             ar_cauchy32_ycwg = ar_cauchy32_ycwg,

             ar_cauchy11_xsfz = ar_cauchy11_xsfz,
             ar_cauchy21_xsfz = ar_cauchy21_xsfz,
             ar_cauchy31_xsfz = ar_cauchy31_xsfz,

             ar_cauchy12_xsfz = ar_cauchy12_xsfz,
             ar_cauchy22_xsfz = ar_cauchy22_xsfz,
             ar_cauchy32_xsfz = ar_cauchy32_xsfz,


             ar_cauchy_overcomplete11_gauss = ar_cauchy_overcomplete11_gauss,
             ar_cauchy_overcomplete21_gauss = ar_cauchy_overcomplete21_gauss,
             ar_cauchy_overcomplete31_gauss = ar_cauchy_overcomplete31_gauss,

             ar_cauchy_overcomplete11_bernoulli = ar_cauchy_overcomplete11_bernoulli,
             ar_cauchy_overcomplete21_bernoulli = ar_cauchy_overcomplete21_bernoulli,
             ar_cauchy_overcomplete31_bernoulli = ar_cauchy_overcomplete31_bernoulli,

             ar_cauchy_overcomplete11_partial_fourier = ar_cauchy_overcomplete11_partial_fourier,
             ar_cauchy_overcomplete21_partial_fourier = ar_cauchy_overcomplete21_partial_fourier,
             ar_cauchy_overcomplete31_partial_fourier = ar_cauchy_overcomplete31_partial_fourier,

             ar_cauchy_overcomplete11_partial_dct = ar_cauchy_overcomplete11_partial_dct,
             ar_cauchy_overcomplete21_partial_dct = ar_cauchy_overcomplete21_partial_dct,
             ar_cauchy_overcomplete31_partial_dct = ar_cauchy_overcomplete31_partial_dct,

             ar_cauchy_overcomplete11_toeplitz = ar_cauchy_overcomplete11_toeplitz,
             ar_cauchy_overcomplete21_toeplitz = ar_cauchy_overcomplete21_toeplitz,
             ar_cauchy_overcomplete31_toeplitz = ar_cauchy_overcomplete31_toeplitz,

             ar_cauchy_overcomplete11_binary_block = ar_cauchy_overcomplete11_binary_block,
             ar_cauchy_overcomplete21_binary_block = ar_cauchy_overcomplete21_binary_block,
             ar_cauchy_overcomplete31_binary_block = ar_cauchy_overcomplete31_binary_block,

             ar_cauchy_overcomplete11_sgn = ar_cauchy_overcomplete11_sgn,
             ar_cauchy_overcomplete21_sgn = ar_cauchy_overcomplete21_sgn,
             ar_cauchy_overcomplete31_sgn = ar_cauchy_overcomplete31_sgn,

             ar_cauchy_overcomplete11_gdo = ar_cauchy_overcomplete11_gdo,
             ar_cauchy_overcomplete21_gdo = ar_cauchy_overcomplete21_gdo,
             ar_cauchy_overcomplete31_gdo = ar_cauchy_overcomplete31_gdo,

             ar_cauchy_overcomplete12_gdo = ar_cauchy_overcomplete12_gdo,
             ar_cauchy_overcomplete22_gdo = ar_cauchy_overcomplete22_gdo,
             ar_cauchy_overcomplete32_gdo = ar_cauchy_overcomplete32_gdo,

             ar_cauchy_overcomplete11_gdo_adaptive = ar_cauchy_overcomplete11_gdo_adaptive,
             ar_cauchy_overcomplete21_gdo_adaptive = ar_cauchy_overcomplete21_gdo_adaptive,
             ar_cauchy_overcomplete31_gdo_adaptive = ar_cauchy_overcomplete31_gdo_adaptive,

             ar_cauchy_overcomplete12_gdo_adaptive = ar_cauchy_overcomplete12_gdo_adaptive,
             ar_cauchy_overcomplete22_gdo_adaptive = ar_cauchy_overcomplete22_gdo_adaptive,
             ar_cauchy_overcomplete32_gdo_adaptive = ar_cauchy_overcomplete32_gdo_adaptive,

             ar_cauchy_overcomplete11_ajs = ar_cauchy_overcomplete11_ajs,
             ar_cauchy_overcomplete21_ajs = ar_cauchy_overcomplete21_ajs,
             ar_cauchy_overcomplete31_ajs = ar_cauchy_overcomplete31_ajs,

             ar_cauchy_overcomplete11_afms = ar_cauchy_overcomplete11_afms,
             ar_cauchy_overcomplete21_afms = ar_cauchy_overcomplete21_afms,
             ar_cauchy_overcomplete31_afms = ar_cauchy_overcomplete31_afms,

             ar_cauchy_overcomplete12_afms = ar_cauchy_overcomplete12_afms,
             ar_cauchy_overcomplete22_afms = ar_cauchy_overcomplete22_afms,
             ar_cauchy_overcomplete32_afms = ar_cauchy_overcomplete32_afms,

             ar_cauchy_overcomplete11_hblz = ar_cauchy_overcomplete11_hblz,
             ar_cauchy_overcomplete21_hblz = ar_cauchy_overcomplete21_hblz,
             ar_cauchy_overcomplete31_hblz = ar_cauchy_overcomplete31_hblz,

             ar_cauchy_overcomplete12_hblz = ar_cauchy_overcomplete12_hblz,
             ar_cauchy_overcomplete22_hblz = ar_cauchy_overcomplete22_hblz,
             ar_cauchy_overcomplete32_hblz = ar_cauchy_overcomplete32_hblz,

             ar_cauchy_overcomplete11_ycwg = ar_cauchy_overcomplete11_ycwg,
             ar_cauchy_overcomplete21_ycwg = ar_cauchy_overcomplete21_ycwg,
             ar_cauchy_overcomplete31_ycwg = ar_cauchy_overcomplete31_ycwg,

             ar_cauchy_overcomplete12_ycwg = ar_cauchy_overcomplete12_ycwg,
             ar_cauchy_overcomplete22_ycwg = ar_cauchy_overcomplete22_ycwg,
             ar_cauchy_overcomplete32_ycwg = ar_cauchy_overcomplete32_ycwg,

             ar_cauchy_overcomplete11_xsfz = ar_cauchy_overcomplete11_xsfz,
             ar_cauchy_overcomplete21_xsfz = ar_cauchy_overcomplete21_xsfz,
             ar_cauchy_overcomplete31_xsfz = ar_cauchy_overcomplete31_xsfz,

             ar_cauchy_overcomplete12_xsfz = ar_cauchy_overcomplete12_xsfz,
             ar_cauchy_overcomplete22_xsfz = ar_cauchy_overcomplete22_xsfz,
             ar_cauchy_overcomplete32_xsfz = ar_cauchy_overcomplete32_xsfz

             )


# Fourier:
# --------

data_stub_fourier = load_stub_data(dir_stubs, "fourier")
shapes_fourier = data_stub_fourier["tracked_shapes_fourier"]
shapes_fourier = [tuple(item.tolist()) for item in shapes_fourier]

np.random.random = stub_random_random_from_data(
    {
     shapes_fourier[0] : data_stub_fourier["uniform_ns_m_fourier1"],
     shapes_fourier[1] : data_stub_fourier["uniform_ns_n_fourier1"]
     })


probability = 0.5
np.random.binomial = stub_random_binomial_from_data(
    {
     (1, probability, shapes_fourier[2]) : data_stub_fourier["bernoulli_ns_m_fourier1"]
     })


random.sample = stub_random_sample_from_data(
    {
     (range(shapes_fourier[3][0]), shapes_fourier[3][1]) : data_stub_fourier["samples_m_ns_fourier1"],

     (range(shapes_fourier[4][0]), shapes_fourier[4][1]) : data_stub_fourier["samples_n_m_fourier1"]
     })


choices = (-1,1)
np.random.choice = stub_random_choice_from_data(
    {
     (choices, shapes_fourier[5][0]) : data_stub_fourier["samples_binary_m_fourier1"]
     })


np.random.randn = stub_random_randn_from_data(
    {
     shapes_fourier[6] : data_stub_fourier["gaussian_ns_m_fourier1"]
     })


scipy.stats.unitary_group.rvs = stub_random_unitary_from_data(
    {
     shapes_fourier[7][0] : data_stub_fourier["random_unitary_ns_m_fourier"]
     })


# Dicionaries:
# ............
if set_measurment_marices_fourier:
    data_fourier = load_filter_data(dir_filters, "fourier_frame")

    a0_fourier1_filtered1 = data_fourier["a0_fourier1_filtered1"]

    a0_fourier1_filtered2 = data_fourier["a0_fourier1_filtered2"]

    csMM_fourier11 = cssr.MeasurementMatrices(a0_fourier1_filtered1, number_samples)

    csMM_fourier12 = cssr.MeasurementMatrices(a0_fourier1_filtered2, number_samples)


    ar_fourier11_gauss = csMM_fourier11.random_gauss_matrix()

    ar_fourier11_bernoulli = csMM_fourier11.random_bernoulli_matrix(probability=0.5)

    ar_fourier11_partial_fourier = csMM_fourier11.random_partial_fourier_matrix()

    ar_fourier11_partial_dct = csMM_fourier11.random_partial_dct_matrix()

    ar_fourier11_toeplitz = csMM_fourier11.random_toeplitz_matrix()

    ar_fourier11_binary_block = csMM_fourier11.binary_block()

    ar_fourier11_sgn = csMM_fourier11.random_sgn_matrix()

    ar_fourier11_gdo = csMM_fourier11.gdo_measurement_matrix(l=l, p=p)

    ar_fourier12_gdo = csMM_fourier12.gdo_measurement_matrix(l=l, p=p)

    ar_fourier11_gdo_adaptive = csMM_fourier11.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_fourier12_gdo_adaptive = csMM_fourier12.gdo_measurement_matrix_adaptive(l=l, p=p)

    ar_fourier11_ajs = csMM_fourier11.ajs(max_iter=max_iter)

    ar_fourier11_afms = csMM_fourier11.afms(max_iter=max_iter)

    ar_fourier12_afms = csMM_fourier12.afms(max_iter=max_iter)

    ar_fourier11_hblz = csMM_fourier11.hblz(l=l, p=p)

    ar_fourier12_hblz = csMM_fourier12.hblz(l=l, p=p)

    ar_fourier11_ycwg = csMM_fourier11.ycwg(max_iter=max_iter)

    ar_fourier12_ycwg = csMM_fourier12.ycwg(max_iter=max_iter)

    ar_fourier11_xsfz = csMM_fourier11.xsfz(max_iter=max_iter)

    ar_fourier12_xsfz = csMM_fourier12.xsfz(max_iter=max_iter)


if save_measurment_marices_fourier and set_measurment_marices_fourier:
    np.savez(dir_measurement_matrices + "fourier",
             probability = 0.5,

             ar_fourier11_gauss = ar_fourier11_gauss,

             ar_fourier11_bernoulli = ar_fourier11_bernoulli,

             ar_fourier11_partial_fourier = ar_fourier11_partial_fourier,

             ar_fourier11_partial_dct = ar_fourier11_partial_dct,

             ar_fourier11_toeplitz = ar_fourier11_toeplitz,

             ar_fourier11_binary_block = ar_fourier11_binary_block,

             ar_fourier11_sgn = ar_fourier11_sgn,

             ar_fourier11_gdo = ar_fourier11_gdo,

             ar_fourier12_gdo = ar_fourier12_gdo,

             ar_fourier11_gdo_adaptive = ar_fourier11_gdo_adaptive,

             ar_fourier12_gdo_adaptive = ar_fourier12_gdo_adaptive,

             ar_fourier11_ajs = ar_fourier11_ajs,

             ar_fourier11_afms = ar_fourier11_afms,

             ar_fourier12_afms = ar_fourier12_afms,

             ar_fourier11_hblz = ar_fourier11_hblz,

             ar_fourier12_hblz = ar_fourier12_hblz,

             ar_fourier11_ycwg = ar_fourier11_ycwg,

             ar_fourier12_ycwg = ar_fourier12_ycwg,

             ar_fourier11_xsfz = ar_fourier11_xsfz,

             ar_fourier12_xsfz = ar_fourier12_xsfz

             )





