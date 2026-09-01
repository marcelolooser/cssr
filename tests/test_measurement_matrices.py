"""
Test module for the measurement_matrices module of the cssr package.

@author: marcelo looser
"""

import scipy
import pytest
import random
import numpy as np

# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cssr import Frames
from cssr import Filters
from cssr import MeasurementMatrices

# =============================================================================
# Stubbing:
# =============================================================================

@pytest.fixture
def dir_stubs():
    return "data/data_stubs/"


@pytest.fixture
def data(request):
    return request.param


@pytest.fixture
def stub_random_random_from_data():
    def func(data):
        def stub(*, size=None):
            try:
                return data[size]
            except KeyError:
                raise ValueError(f"No stub data available for the shape {size}.")
        return stub
    return func


@pytest.fixture
def stub_random_binomial_from_data():
    def func(data):
        def stub(*, n=None, p=None, size=None):
            try:
                return data[(n, p, size)]
            except KeyError:
                raise ValueError(f"No stub data available for {(n, p, size)}.")
        return stub
    return func


@pytest.fixture
def stub_random_sample_from_data():
    def func(data):
        def stub(*args):
            try:
                return data[args]
            except KeyError:
                raise ValueError(f"No stub data available for the shape {args}.")
        return stub
    return func


@pytest.fixture
def stub_random_choice_from_data():
    def func(data):
        def stub(*args, size=None):
            try:
                return data[(args[0], size)]
            except KeyError:
                raise ValueError(f"No stub data available for {(args[0], size)}.")
        return stub
    return func


@pytest.fixture
def stub_random_randn_from_data():
    def func(data):
        def stub(*args):
            try:
                return data[args]
            except KeyError:
                raise ValueError(f"No stub data available for the shape {args}.")
        return stub
    return func


@pytest.fixture
def stub_random_unitary_from_data():
    def func(data):
        def stub(*args):
            try:
                return data[args[0]]
            except KeyError:
                raise ValueError(f"No stub data available for the shape {args}.")
        return stub
    return func



@pytest.fixture
def mock_random_heaviside(dir_stubs):
    data_stub_heaviside = np.load(dir_stubs + "heaviside" + ".npz")
    shapes_heaviside = data_stub_heaviside["tracked_shapes_heaviside"]
    shapes_heaviside = [tuple(item.tolist()) for item in shapes_heaviside]

    np_random_random = {
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
     }

    probability = float(data_stub_heaviside["probability"])
    np_random_binomial = {
         (1, probability, shapes_heaviside[12]) : data_stub_heaviside["bernoulli_ns_m_heaviside1"],
         (1, probability, shapes_heaviside[13]) : data_stub_heaviside["bernoulli_ns_m_heaviside2"],
         (1, probability, shapes_heaviside[14]) : data_stub_heaviside["bernoulli_ns_m_heaviside3"],

         (1, probability, shapes_heaviside[15]) : data_stub_heaviside["bernoulli_ns_m_heaviside_overcomplete1"],
         (1, probability, shapes_heaviside[16]) : data_stub_heaviside["bernoulli_ns_m_heaviside_overcomplete2"],
         (1, probability, shapes_heaviside[17]) : data_stub_heaviside["bernoulli_ns_m_heaviside_overcomplete3"]
         }


    random_sample = {
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
         }


    choices = (-1,1)
    np_random_choice = {
         (choices, shapes_heaviside[30][0]) : data_stub_heaviside["samples_binary_m_heaviside1"],
         (choices, shapes_heaviside[31][0]) : data_stub_heaviside["samples_binary_m_heaviside2"],
         (choices, shapes_heaviside[32][0]) : data_stub_heaviside["samples_binary_m_heaviside3"],

         (choices, shapes_heaviside[33][0]) : data_stub_heaviside["samples_binary_m_heaviside_overcomplete1"],
         (choices, shapes_heaviside[34][0]) : data_stub_heaviside["samples_binary_m_heaviside_overcomplete2"],
         (choices, shapes_heaviside[35][0]) : data_stub_heaviside["samples_binary_m_heaviside_overcomplete3"]
         }


    np_random_randn = {
         shapes_heaviside[36] : data_stub_heaviside["gaussian_ns_m_heaviside1"],
         shapes_heaviside[37] : data_stub_heaviside["gaussian_ns_m_heaviside2"],
         shapes_heaviside[38] : data_stub_heaviside["gaussian_ns_m_heaviside3"],

         shapes_heaviside[39] : data_stub_heaviside["gaussian_ns_m_heaviside_overcomplete1"],
         shapes_heaviside[40] : data_stub_heaviside["gaussian_ns_m_heaviside_overcomplete2"],
         shapes_heaviside[41] : data_stub_heaviside["gaussian_ns_m_heaviside_overcomplete3"]
         }

    scipy_stats_unitary_group_rvs = {
         shapes_heaviside[42][0] : data_stub_heaviside["random_unitary_ns_m_heaviside"]
         }

    return [np_random_random, np_random_binomial, random_sample, np_random_choice, np_random_randn, scipy_stats_unitary_group_rvs]


@pytest.fixture
def mock_random_gaussian(dir_stubs):
    data_stub_gaussian = np.load(dir_stubs + "gaussian" + ".npz")
    shapes_gaussian = data_stub_gaussian["tracked_shapes_gaussian"]
    shapes_gaussian = [tuple(item.tolist()) for item in shapes_gaussian]

    np_random_random = {
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
     }

    probability = float(data_stub_gaussian["probability"])
    np_random_binomial = {
         (1, probability, shapes_gaussian[12]) : data_stub_gaussian["bernoulli_ns_m_gaussian1"],
         (1, probability, shapes_gaussian[13]) : data_stub_gaussian["bernoulli_ns_m_gaussian2"],
         (1, probability, shapes_gaussian[14]) : data_stub_gaussian["bernoulli_ns_m_gaussian3"],

         (1, probability, shapes_gaussian[15]) : data_stub_gaussian["bernoulli_ns_m_gaussian_overcomplete1"],
         (1, probability, shapes_gaussian[16]) : data_stub_gaussian["bernoulli_ns_m_gaussian_overcomplete2"],
         (1, probability, shapes_gaussian[17]) : data_stub_gaussian["bernoulli_ns_m_gaussian_overcomplete3"]
         }


    random_sample = {
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
         }


    choices = (-1,1)
    np_random_choice = {
         (choices, shapes_gaussian[30][0]) : data_stub_gaussian["samples_binary_m_gaussian1"],
         (choices, shapes_gaussian[31][0]) : data_stub_gaussian["samples_binary_m_gaussian2"],
         (choices, shapes_gaussian[32][0]) : data_stub_gaussian["samples_binary_m_gaussian3"],

         (choices, shapes_gaussian[33][0]) : data_stub_gaussian["samples_binary_m_gaussian_overcomplete1"],
         (choices, shapes_gaussian[34][0]) : data_stub_gaussian["samples_binary_m_gaussian_overcomplete2"],
         (choices, shapes_gaussian[35][0]) : data_stub_gaussian["samples_binary_m_gaussian_overcomplete3"]
         }


    np_random_randn = {
         shapes_gaussian[36] : data_stub_gaussian["gaussian_ns_m_gaussian1"],
         shapes_gaussian[37] : data_stub_gaussian["gaussian_ns_m_gaussian2"],
         shapes_gaussian[38] : data_stub_gaussian["gaussian_ns_m_gaussian3"],

         shapes_gaussian[39] : data_stub_gaussian["gaussian_ns_m_gaussian_overcomplete1"],
         shapes_gaussian[40] : data_stub_gaussian["gaussian_ns_m_gaussian_overcomplete2"],
         shapes_gaussian[41] : data_stub_gaussian["gaussian_ns_m_gaussian_overcomplete3"]
         }

    scipy_stats_unitary_group_rvs = {
         shapes_gaussian[42][0] : data_stub_gaussian["random_unitary_ns_m_gaussian"]
         }

    return [np_random_random, np_random_binomial, random_sample, np_random_choice, np_random_randn, scipy_stats_unitary_group_rvs]


@pytest.fixture
def mock_random_cauchy(dir_stubs):
    data_stub_cauchy = np.load(dir_stubs + "cauchy" + ".npz")
    shapes_cauchy = data_stub_cauchy["tracked_shapes_cauchy"]
    shapes_cauchy = [tuple(item.tolist()) for item in shapes_cauchy]

    np_random_random = {
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
     }

    probability = float(data_stub_cauchy["probability"])
    np_random_binomial = {
         (1, probability, shapes_cauchy[12]) : data_stub_cauchy["bernoulli_ns_m_cauchy1"],
         (1, probability, shapes_cauchy[13]) : data_stub_cauchy["bernoulli_ns_m_cauchy2"],
         (1, probability, shapes_cauchy[14]) : data_stub_cauchy["bernoulli_ns_m_cauchy3"],

         (1, probability, shapes_cauchy[15]) : data_stub_cauchy["bernoulli_ns_m_cauchy_overcomplete1"],
         (1, probability, shapes_cauchy[16]) : data_stub_cauchy["bernoulli_ns_m_cauchy_overcomplete2"],
         (1, probability, shapes_cauchy[17]) : data_stub_cauchy["bernoulli_ns_m_cauchy_overcomplete3"]
         }


    random_sample = {
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
         }


    choices = (-1,1)
    np_random_choice = {
         (choices, shapes_cauchy[30][0]) : data_stub_cauchy["samples_binary_m_cauchy1"],
         (choices, shapes_cauchy[31][0]) : data_stub_cauchy["samples_binary_m_cauchy2"],
         (choices, shapes_cauchy[32][0]) : data_stub_cauchy["samples_binary_m_cauchy3"],

         (choices, shapes_cauchy[33][0]) : data_stub_cauchy["samples_binary_m_cauchy_overcomplete1"],
         (choices, shapes_cauchy[34][0]) : data_stub_cauchy["samples_binary_m_cauchy_overcomplete2"],
         (choices, shapes_cauchy[35][0]) : data_stub_cauchy["samples_binary_m_cauchy_overcomplete3"]
         }

    np_random_randn = {
         shapes_cauchy[36] : data_stub_cauchy["gaussian_ns_m_cauchy1"],
         shapes_cauchy[37] : data_stub_cauchy["gaussian_ns_m_cauchy2"],
         shapes_cauchy[38] : data_stub_cauchy["gaussian_ns_m_cauchy3"],

         shapes_cauchy[39] : data_stub_cauchy["gaussian_ns_m_cauchy_overcomplete1"],
         shapes_cauchy[40] : data_stub_cauchy["gaussian_ns_m_cauchy_overcomplete2"],
         shapes_cauchy[41] : data_stub_cauchy["gaussian_ns_m_cauchy_overcomplete3"]
         }

    scipy_stats_unitary_group_rvs = {
         shapes_cauchy[42][0] : data_stub_cauchy["random_unitary_ns_m_cauchy"]
         }

    return [np_random_random, np_random_binomial, random_sample, np_random_choice, np_random_randn, scipy_stats_unitary_group_rvs]


@pytest.fixture
def mock_random_fourier(dir_stubs):
    data_stub_fourier = np.load(dir_stubs + "fourier" + ".npz")
    shapes_fourier = data_stub_fourier["tracked_shapes_fourier"]
    shapes_fourier = [tuple(item.tolist()) for item in shapes_fourier]

    np_random_random = {
     shapes_fourier[0] : data_stub_fourier["uniform_ns_m_fourier1"],

     shapes_fourier[1] : data_stub_fourier["uniform_ns_n_fourier1"]
     }

    probability = float(data_stub_fourier["probability"])
    np_random_binomial = {
         (1, probability, shapes_fourier[2]) : data_stub_fourier["bernoulli_ns_m_fourier1"]
         }


    random_sample = {
         (range(shapes_fourier[3][0]), shapes_fourier[3][1]) : data_stub_fourier["samples_m_ns_fourier1"],

         (range(shapes_fourier[4][0]), shapes_fourier[4][1]) : data_stub_fourier["samples_n_m_fourier1"]
         }


    choices = (-1,1)
    np_random_choice = {
         (choices, shapes_fourier[5][0]) : data_stub_fourier["samples_binary_m_fourier1"]
         }


    np_random_randn = {
         shapes_fourier[6] : data_stub_fourier["gaussian_ns_m_fourier1"]
         }


    scipy_stats_unitary_group_rvs = {
         shapes_fourier[7][0] : data_stub_fourier["random_unitary_ns_m_fourier"]
         }

    return [np_random_random, np_random_binomial, random_sample, np_random_choice, np_random_randn, scipy_stats_unitary_group_rvs]


# =============================================================================
# General setup:
# =============================================================================

@pytest.fixture
def dir_measurement_matrices():
    return "data/data_measurement_matrices/"


@pytest.fixture
def frame_name(request):
    return request.param


@pytest.fixture
def load_measurement_matrices_data(dir_measurement_matrices, frame_name):
    return np.load(dir_measurement_matrices + frame_name + ".npz")


@pytest.fixture
def construct_test_signal_x_components():
    return np.arange(0, 20, 0.1)


@pytest.fixture
def measurement_matrices_configurations():
    number_samples = 25

    # Number of iterations:
    # ---------------------
    max_iter = 8
    l, p = 3, 3
    return [number_samples, max_iter, l, p]


@pytest.fixture
def construct_cutoffs():
    cutoffs = [[1, None], [0.6, 1.2], [70, None]] # where 70 is 70° celsius
    return cutoffs


@pytest.fixture
def construct_heaviside_frame(construct_test_signal_x_components):

    # Preliminaries:
    # --------------
    x = construct_test_signal_x_components
    dim = len(x)

    bw_heaviside = [1, dim//2, dim - 1]
    bw_heaviside_overcomplete = [[None, None], [dim//2 - 3, dim//2 + 3], [dim//2 - 3, dim//2 + 3]]
    ss_heaviside_overcomplete = [None, None, 3]

    # Construct frames:
    # -----------------

    csF = Frames(x)
    heaviside_frames = [
        csF.heaviside(bw_heaviside[0]),
        csF.heaviside(bw_heaviside[1]),
        csF.heaviside(bw_heaviside[2]),
        csF.heaviside_overcomplete(bw_heaviside_overcomplete[0][0], ss_heaviside_overcomplete[0]),
        csF.heaviside_overcomplete(bw_heaviside_overcomplete[1], ss_heaviside_overcomplete[1]),
        csF.heaviside_overcomplete(bw_heaviside_overcomplete[2], ss_heaviside_overcomplete[2]),
        ]

    return heaviside_frames


@pytest.fixture
def construct_gaussian_frame(construct_test_signal_x_components):

    # Preliminaries:
    # --------------
    x = construct_test_signal_x_components

    bw_gaussian = [abs(x[1] - x[0]), abs(x[-1] - x[0])/2, abs(x[-1] - x[0]) - abs(x[1] - x[0])]
    bw_gaussian_overcomplete = [[None, None], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])]]
    ss_gaussian_overcomplete = [None, None, 3*abs(x[1] - x[0])]

    # Construct frames:
    # -----------------

    csF = Frames(x)
    gaussian_frames = [
        csF.gaussian(bw_gaussian[0]),
        csF.gaussian(bw_gaussian[1]),
        csF.gaussian(bw_gaussian[2]),
        csF.gaussian_overcomplete(bw_gaussian_overcomplete[0][0], ss_gaussian_overcomplete[0]),
        csF.gaussian_overcomplete(bw_gaussian_overcomplete[1], ss_gaussian_overcomplete[1]),
        csF.gaussian_overcomplete(bw_gaussian_overcomplete[2], ss_gaussian_overcomplete[2]),
        ]

    return gaussian_frames


@pytest.fixture
def construct_cauchy_frame(construct_test_signal_x_components):

    # Preliminaries:
    # --------------
    x = construct_test_signal_x_components

    bw_cauchy = [abs(x[1] - x[0]), abs(x[-1] - x[0])/2, abs(x[-1] - x[0]) - abs(x[1] - x[0])]
    bw_cauchy_overcomplete = [[None, None], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])]]
    ss_cauchy_overcomplete = [None, None, 3*abs(x[1] - x[0])]

    # Construct frames:
    # -----------------

    csF = Frames(x)
    cauchy_frames = [
        csF.cauchy(bw_cauchy[0]),
        csF.cauchy(bw_cauchy[1]),
        csF.cauchy(bw_cauchy[2]),
        csF.cauchy_overcomplete(bw_cauchy_overcomplete[0][0], ss_cauchy_overcomplete[0]),
        csF.cauchy_overcomplete(bw_cauchy_overcomplete[1], ss_cauchy_overcomplete[1]),
        csF.cauchy_overcomplete(bw_cauchy_overcomplete[2], ss_cauchy_overcomplete[2]),
        ]

    return cauchy_frames


@pytest.fixture
def construct_fourier_frame(construct_test_signal_x_components):

    # Preliminaries:
    # --------------
    x = construct_test_signal_x_components

    # Construct frames:
    # -----------------

    csF = Frames(x)
    return csF.fourier()


@pytest.fixture
def construct_filtered_heaviside_frame(construct_test_signal_x_components, construct_heaviside_frame, construct_cutoffs):
    x = construct_test_signal_x_components
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    heaviside_frames = construct_heaviside_frame

    a0_heaviside1, a0_heaviside2, a0_heaviside3 = heaviside_frames[:3]
    a0_heaviside_overcomplete1, a0_heaviside_overcomplete2, a0_heaviside_overcomplete3 = heaviside_frames[3:]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............

    csFr_a01_heaviside = Filters(a0_heaviside1, x, cutoffs[0][0])
    csFr_a02_heaviside = Filters(a0_heaviside2, x, cutoffs[0][0])
    csFr_a03_heaviside = Filters(a0_heaviside3, x, cutoffs[0][0])

    csFr_a01_heaviside.heaviside_lowpass_filter(return_array=False)
    csFr_a02_heaviside.heaviside_lowpass_filter(return_array=False)
    csFr_a03_heaviside.heaviside_lowpass_filter(return_array=False)

    csFr_a01_heaviside.fir_filter(return_array=False)
    csFr_a02_heaviside.fir_filter(return_array=False)
    csFr_a03_heaviside.fir_filter(return_array=False)

    csFr_a01_heaviside.cutoff = cutoffs[1]
    csFr_a02_heaviside.cutoff = cutoffs[1]
    csFr_a03_heaviside.cutoff = cutoffs[1]

    csFr_a01_heaviside.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_heaviside.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a03_heaviside.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_a01_heaviside.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a02_heaviside.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a03_heaviside.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_heaviside1_filtered1 = csFr_a01_heaviside.truncated_a0
    a0_heaviside2_filtered1 = csFr_a02_heaviside.truncated_a0
    a0_heaviside3_filtered1 = csFr_a03_heaviside.truncated_a0

    csFr_a01_heaviside.reset()
    csFr_a02_heaviside.reset()
    csFr_a03_heaviside.reset()

    csFr_a01_heaviside.butter_filter(return_array=False)
    csFr_a02_heaviside.butter_filter(return_array=False)
    csFr_a03_heaviside.butter_filter(return_array=False)

    csFr_a01_heaviside.instrumental_lowpass_filter(return_array=False)
    csFr_a02_heaviside.instrumental_lowpass_filter(return_array=False)
    csFr_a03_heaviside.instrumental_lowpass_filter(return_array=False)

    csFr_a01_heaviside.cutoff = cutoffs[2][0]
    csFr_a02_heaviside.cutoff = cutoffs[2][0]
    csFr_a03_heaviside.cutoff = cutoffs[2][0]

    csFr_a01_heaviside.thermal_lowpass_filter(return_array=False)
    csFr_a02_heaviside.thermal_lowpass_filter(return_array=False)
    csFr_a03_heaviside.thermal_lowpass_filter(return_array=False)

    a0_heaviside1_filtered2 = csFr_a01_heaviside.truncated_a0
    a0_heaviside2_filtered2 = csFr_a02_heaviside.truncated_a0
    a0_heaviside3_filtered2 = csFr_a03_heaviside.truncated_a0


    # Overcomplete dictonaries:
    # .........................

    csFr_a01_heaviside_overcomplete = Filters(a0_heaviside_overcomplete1, x, cutoffs[0][0])
    csFr_a02_heaviside_overcomplete = Filters(a0_heaviside_overcomplete2, x, cutoffs[0][0])
    csFr_a03_heaviside_overcomplete = Filters(a0_heaviside_overcomplete3, x, cutoffs[0][0])

    csFr_a01_heaviside_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a02_heaviside_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a03_heaviside_overcomplete.heaviside_lowpass_filter(return_array=False)

    csFr_a01_heaviside_overcomplete.fir_filter(return_array=False)
    csFr_a02_heaviside_overcomplete.fir_filter(return_array=False)
    csFr_a03_heaviside_overcomplete.fir_filter(return_array=False)

    csFr_a01_heaviside_overcomplete.cutoff = cutoffs[1]
    csFr_a02_heaviside_overcomplete.cutoff = cutoffs[1]
    csFr_a03_heaviside_overcomplete.cutoff = cutoffs[1]

    csFr_a01_heaviside_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_heaviside_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a03_heaviside_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_a01_heaviside_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a02_heaviside_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a03_heaviside_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_heaviside_overcomplete1_filtered1 = csFr_a01_heaviside_overcomplete.truncated_a0
    a0_heaviside_overcomplete2_filtered1 = csFr_a02_heaviside_overcomplete.truncated_a0
    a0_heaviside_overcomplete3_filtered1 = csFr_a03_heaviside_overcomplete.truncated_a0

    csFr_a01_heaviside_overcomplete.reset()
    csFr_a02_heaviside_overcomplete.reset()
    csFr_a03_heaviside_overcomplete.reset()

    csFr_a01_heaviside_overcomplete.butter_filter(return_array=False)
    csFr_a02_heaviside_overcomplete.butter_filter(return_array=False)
    csFr_a03_heaviside_overcomplete.butter_filter(return_array=False)

    csFr_a01_heaviside_overcomplete.instrumental_lowpass_filter(return_array=False)
    csFr_a02_heaviside_overcomplete.instrumental_lowpass_filter(return_array=False)
    csFr_a03_heaviside_overcomplete.instrumental_lowpass_filter(return_array=False)

    csFr_a01_heaviside_overcomplete.cutoff = cutoffs[2][0]
    csFr_a02_heaviside_overcomplete.cutoff = cutoffs[2][0]
    csFr_a03_heaviside_overcomplete.cutoff = cutoffs[2][0]

    csFr_a01_heaviside_overcomplete.thermal_lowpass_filter(return_array=False)
    csFr_a02_heaviside_overcomplete.thermal_lowpass_filter(return_array=False)
    csFr_a03_heaviside_overcomplete.thermal_lowpass_filter(return_array=False)

    a0_heaviside_overcomplete1_filtered2 = csFr_a01_heaviside_overcomplete.truncated_a0
    a0_heaviside_overcomplete2_filtered2 = csFr_a02_heaviside_overcomplete.truncated_a0
    a0_heaviside_overcomplete3_filtered2 = csFr_a03_heaviside_overcomplete.truncated_a0


    heaviside_filtered_frames = [
        a0_heaviside1_filtered1,
        a0_heaviside2_filtered1,
        a0_heaviside3_filtered1,

        a0_heaviside1_filtered2,
        a0_heaviside2_filtered2,
        a0_heaviside3_filtered2,

        a0_heaviside_overcomplete1_filtered1,
        a0_heaviside_overcomplete2_filtered1,
        a0_heaviside_overcomplete3_filtered1,

        a0_heaviside_overcomplete1_filtered2,
        a0_heaviside_overcomplete2_filtered2,
        a0_heaviside_overcomplete3_filtered2
        ]

    return heaviside_filtered_frames


@pytest.fixture
def construct_filtered_gaussian_frame(construct_test_signal_x_components, construct_gaussian_frame, construct_cutoffs):
    x = construct_test_signal_x_components
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    gaussian_frames = construct_gaussian_frame

    a0_gaussian1, a0_gaussian2, a0_gaussian3 = gaussian_frames[:3]
    a0_gaussian_overcomplete1, a0_gaussian_overcomplete2, a0_gaussian_overcomplete3 = gaussian_frames[3:]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............

    csFr_a01_gaussian = Filters(a0_gaussian1, x, cutoffs[0][0])
    csFr_a02_gaussian = Filters(a0_gaussian2, x, cutoffs[0][0])
    csFr_a03_gaussian = Filters(a0_gaussian3, x, cutoffs[0][0])

    csFr_a01_gaussian.heaviside_lowpass_filter(return_array=False)
    csFr_a02_gaussian.heaviside_lowpass_filter(return_array=False)
    csFr_a03_gaussian.heaviside_lowpass_filter(return_array=False)

    csFr_a01_gaussian.fir_filter(return_array=False)
    csFr_a02_gaussian.fir_filter(return_array=False)
    csFr_a03_gaussian.fir_filter(return_array=False)

    csFr_a01_gaussian.cutoff = cutoffs[1]
    csFr_a02_gaussian.cutoff = cutoffs[1]
    csFr_a03_gaussian.cutoff = cutoffs[1]

    csFr_a01_gaussian.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_gaussian.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a03_gaussian.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_a01_gaussian.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a02_gaussian.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a03_gaussian.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_gaussian1_filtered1 = csFr_a01_gaussian.truncated_a0
    a0_gaussian2_filtered1 = csFr_a02_gaussian.truncated_a0
    a0_gaussian3_filtered1 = csFr_a03_gaussian.truncated_a0

    csFr_a01_gaussian.reset()
    csFr_a02_gaussian.reset()
    csFr_a03_gaussian.reset()

    csFr_a01_gaussian.butter_filter(return_array=False)
    csFr_a02_gaussian.butter_filter(return_array=False)
    csFr_a03_gaussian.butter_filter(return_array=False)

    csFr_a01_gaussian.instrumental_lowpass_filter(return_array=False)
    csFr_a02_gaussian.instrumental_lowpass_filter(return_array=False)
    csFr_a03_gaussian.instrumental_lowpass_filter(return_array=False)

    csFr_a01_gaussian.cutoff = cutoffs[2][0]
    csFr_a02_gaussian.cutoff = cutoffs[2][0]
    csFr_a03_gaussian.cutoff = cutoffs[2][0]

    csFr_a01_gaussian.thermal_lowpass_filter(return_array=False)
    csFr_a02_gaussian.thermal_lowpass_filter(return_array=False)
    csFr_a03_gaussian.thermal_lowpass_filter(return_array=False)

    a0_gaussian1_filtered2 = csFr_a01_gaussian.truncated_a0
    a0_gaussian2_filtered2 = csFr_a02_gaussian.truncated_a0
    a0_gaussian3_filtered2 = csFr_a03_gaussian.truncated_a0


    # Overcomplete dictonaries:
    # .........................

    csFr_a01_gaussian_overcomplete = Filters(a0_gaussian_overcomplete1, x, cutoffs[0][0])
    csFr_a02_gaussian_overcomplete = Filters(a0_gaussian_overcomplete2, x, cutoffs[0][0])
    csFr_a03_gaussian_overcomplete = Filters(a0_gaussian_overcomplete3, x, cutoffs[0][0])

    csFr_a01_gaussian_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a03_gaussian_overcomplete.heaviside_lowpass_filter(return_array=False)

    csFr_a01_gaussian_overcomplete.fir_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.fir_filter(return_array=False)
    csFr_a03_gaussian_overcomplete.fir_filter(return_array=False)

    csFr_a01_gaussian_overcomplete.cutoff = cutoffs[1]
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[1]
    csFr_a03_gaussian_overcomplete.cutoff = cutoffs[1]

    csFr_a01_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a03_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_a01_gaussian_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a02_gaussian_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a03_gaussian_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_gaussian_overcomplete1_filtered1 = csFr_a01_gaussian_overcomplete.truncated_a0
    a0_gaussian_overcomplete2_filtered1 = csFr_a02_gaussian_overcomplete.truncated_a0
    a0_gaussian_overcomplete3_filtered1 = csFr_a03_gaussian_overcomplete.truncated_a0

    csFr_a01_gaussian_overcomplete.reset()
    csFr_a02_gaussian_overcomplete.reset()
    csFr_a03_gaussian_overcomplete.reset()

    csFr_a01_gaussian_overcomplete.butter_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.butter_filter(return_array=False)
    csFr_a03_gaussian_overcomplete.butter_filter(return_array=False)

    csFr_a01_gaussian_overcomplete.instrumental_lowpass_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.instrumental_lowpass_filter(return_array=False)
    csFr_a03_gaussian_overcomplete.instrumental_lowpass_filter(return_array=False)

    csFr_a01_gaussian_overcomplete.cutoff = cutoffs[2][0]
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[2][0]
    csFr_a03_gaussian_overcomplete.cutoff = cutoffs[2][0]

    csFr_a01_gaussian_overcomplete.thermal_lowpass_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.thermal_lowpass_filter(return_array=False)
    csFr_a03_gaussian_overcomplete.thermal_lowpass_filter(return_array=False)

    a0_gaussian_overcomplete1_filtered2 = csFr_a01_gaussian_overcomplete.truncated_a0
    a0_gaussian_overcomplete2_filtered2 = csFr_a02_gaussian_overcomplete.truncated_a0
    a0_gaussian_overcomplete3_filtered2 = csFr_a03_gaussian_overcomplete.truncated_a0


    gaussian_filtered_frames = [
        a0_gaussian1_filtered1,
        a0_gaussian2_filtered1,
        a0_gaussian3_filtered1,

        a0_gaussian1_filtered2,
        a0_gaussian2_filtered2,
        a0_gaussian3_filtered2,

        a0_gaussian_overcomplete1_filtered1,
        a0_gaussian_overcomplete2_filtered1,
        a0_gaussian_overcomplete3_filtered1,

        a0_gaussian_overcomplete1_filtered2,
        a0_gaussian_overcomplete2_filtered2,
        a0_gaussian_overcomplete3_filtered2
        ]

    return gaussian_filtered_frames


@pytest.fixture
def construct_filtered_cauchy_frame(construct_test_signal_x_components, construct_cauchy_frame, construct_cutoffs):
    x = construct_test_signal_x_components
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    cauchy_frames = construct_cauchy_frame

    a0_cauchy1, a0_cauchy2, a0_cauchy3 = cauchy_frames[:3]
    a0_cauchy_overcomplete1, a0_cauchy_overcomplete2, a0_cauchy_overcomplete3 = cauchy_frames[3:]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............

    csFr_a01_cauchy = Filters(a0_cauchy1, x, cutoffs[0][0])
    csFr_a02_cauchy = Filters(a0_cauchy2, x, cutoffs[0][0])
    csFr_a03_cauchy = Filters(a0_cauchy3, x, cutoffs[0][0])

    csFr_a01_cauchy.heaviside_lowpass_filter(return_array=False)
    csFr_a02_cauchy.heaviside_lowpass_filter(return_array=False)
    csFr_a03_cauchy.heaviside_lowpass_filter(return_array=False)

    csFr_a01_cauchy.fir_filter(return_array=False)
    csFr_a02_cauchy.fir_filter(return_array=False)
    csFr_a03_cauchy.fir_filter(return_array=False)

    csFr_a01_cauchy.cutoff = cutoffs[1]
    csFr_a02_cauchy.cutoff = cutoffs[1]
    csFr_a03_cauchy.cutoff = cutoffs[1]

    csFr_a01_cauchy.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_cauchy.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a03_cauchy.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_a01_cauchy.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a02_cauchy.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a03_cauchy.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_cauchy1_filtered1 = csFr_a01_cauchy.truncated_a0
    a0_cauchy2_filtered1 = csFr_a02_cauchy.truncated_a0
    a0_cauchy3_filtered1 = csFr_a03_cauchy.truncated_a0

    csFr_a01_cauchy.reset()
    csFr_a02_cauchy.reset()
    csFr_a03_cauchy.reset()

    csFr_a01_cauchy.butter_filter(return_array=False)
    csFr_a02_cauchy.butter_filter(return_array=False)
    csFr_a03_cauchy.butter_filter(return_array=False)

    csFr_a01_cauchy.instrumental_lowpass_filter(return_array=False)
    csFr_a02_cauchy.instrumental_lowpass_filter(return_array=False)
    csFr_a03_cauchy.instrumental_lowpass_filter(return_array=False)

    csFr_a01_cauchy.cutoff = cutoffs[2][0]
    csFr_a02_cauchy.cutoff = cutoffs[2][0]
    csFr_a03_cauchy.cutoff = cutoffs[2][0]

    csFr_a01_cauchy.thermal_lowpass_filter(return_array=False)
    csFr_a02_cauchy.thermal_lowpass_filter(return_array=False)
    csFr_a03_cauchy.thermal_lowpass_filter(return_array=False)

    a0_cauchy1_filtered2 = csFr_a01_cauchy.truncated_a0
    a0_cauchy2_filtered2 = csFr_a02_cauchy.truncated_a0
    a0_cauchy3_filtered2 = csFr_a03_cauchy.truncated_a0


    # Overcomplete dictonaries:
    # .........................

    csFr_a01_cauchy_overcomplete = Filters(a0_cauchy_overcomplete1, x, cutoffs[0][0])
    csFr_a02_cauchy_overcomplete = Filters(a0_cauchy_overcomplete2, x, cutoffs[0][0])
    csFr_a03_cauchy_overcomplete = Filters(a0_cauchy_overcomplete3, x, cutoffs[0][0])

    csFr_a01_cauchy_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a02_cauchy_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a03_cauchy_overcomplete.heaviside_lowpass_filter(return_array=False)

    csFr_a01_cauchy_overcomplete.fir_filter(return_array=False)
    csFr_a02_cauchy_overcomplete.fir_filter(return_array=False)
    csFr_a03_cauchy_overcomplete.fir_filter(return_array=False)

    csFr_a01_cauchy_overcomplete.cutoff = cutoffs[1]
    csFr_a02_cauchy_overcomplete.cutoff = cutoffs[1]
    csFr_a03_cauchy_overcomplete.cutoff = cutoffs[1]

    csFr_a01_cauchy_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_cauchy_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a03_cauchy_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_a01_cauchy_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a02_cauchy_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_a03_cauchy_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_cauchy_overcomplete1_filtered1 = csFr_a01_cauchy_overcomplete.truncated_a0
    a0_cauchy_overcomplete2_filtered1 = csFr_a02_cauchy_overcomplete.truncated_a0
    a0_cauchy_overcomplete3_filtered1 = csFr_a03_cauchy_overcomplete.truncated_a0

    csFr_a01_cauchy_overcomplete.reset()
    csFr_a02_cauchy_overcomplete.reset()
    csFr_a03_cauchy_overcomplete.reset()

    csFr_a01_cauchy_overcomplete.butter_filter(return_array=False)
    csFr_a02_cauchy_overcomplete.butter_filter(return_array=False)
    csFr_a03_cauchy_overcomplete.butter_filter(return_array=False)

    csFr_a01_cauchy_overcomplete.instrumental_lowpass_filter(return_array=False)
    csFr_a02_cauchy_overcomplete.instrumental_lowpass_filter(return_array=False)
    csFr_a03_cauchy_overcomplete.instrumental_lowpass_filter(return_array=False)

    csFr_a01_cauchy_overcomplete.cutoff = cutoffs[2][0]
    csFr_a02_cauchy_overcomplete.cutoff = cutoffs[2][0]
    csFr_a03_cauchy_overcomplete.cutoff = cutoffs[2][0]

    csFr_a01_cauchy_overcomplete.thermal_lowpass_filter(return_array=False)
    csFr_a02_cauchy_overcomplete.thermal_lowpass_filter(return_array=False)
    csFr_a03_cauchy_overcomplete.thermal_lowpass_filter(return_array=False)

    a0_cauchy_overcomplete1_filtered2 = csFr_a01_cauchy_overcomplete.truncated_a0
    a0_cauchy_overcomplete2_filtered2 = csFr_a02_cauchy_overcomplete.truncated_a0
    a0_cauchy_overcomplete3_filtered2 = csFr_a03_cauchy_overcomplete.truncated_a0


    cauchy_filtered_frames = [
        a0_cauchy1_filtered1,
        a0_cauchy2_filtered1,
        a0_cauchy3_filtered1,

        a0_cauchy1_filtered2,
        a0_cauchy2_filtered2,
        a0_cauchy3_filtered2,

        a0_cauchy_overcomplete1_filtered1,
        a0_cauchy_overcomplete2_filtered1,
        a0_cauchy_overcomplete3_filtered1,

        a0_cauchy_overcomplete1_filtered2,
        a0_cauchy_overcomplete2_filtered2,
        a0_cauchy_overcomplete3_filtered2
        ]

    return cauchy_filtered_frames



@pytest.fixture
def construct_filtered_fourier_frame(construct_test_signal_x_components, construct_fourier_frame, construct_cutoffs):
    x = construct_test_signal_x_components
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    fourier_frames = construct_fourier_frame

    a0_fourier1 = fourier_frames


    # Main checks:
    # ------------

    # Dictonaries:
    # ............

    csFr_a01_fourier = Filters(a0_fourier1, x, cutoffs[0][0])

    csFr_a01_fourier.heaviside_lowpass_filter(return_array=False)

    csFr_a01_fourier.fir_filter(return_array=False)

    csFr_a01_fourier.cutoff = cutoffs[1]

    csFr_a01_fourier.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_a01_fourier.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_fourier1_filtered1 = csFr_a01_fourier.truncated_a0

    csFr_a01_fourier.reset()

    csFr_a01_fourier.butter_filter(return_array=False)

    csFr_a01_fourier.instrumental_lowpass_filter(return_array=False)

    csFr_a01_fourier.cutoff = cutoffs[2][0]

    csFr_a01_fourier.thermal_lowpass_filter(return_array=False)

    a0_fourier1_filtered2 = csFr_a01_fourier.truncated_a0


    fourier_filtered_frames = [
        a0_fourier1_filtered1,

        a0_fourier1_filtered2
        ]

    return fourier_filtered_frames


@pytest.mark.parametrize("frame_name", ["heaviside"], indirect=True)
def test_heaviside_based_measurement_matrices(load_measurement_matrices_data,
                                              construct_filtered_heaviside_frame,
                                              measurement_matrices_configurations,
                                              mock_random_heaviside,
                                              stub_random_random_from_data,
                                              stub_random_binomial_from_data,
                                              stub_random_sample_from_data,
                                              stub_random_choice_from_data,
                                              stub_random_randn_from_data,
                                              stub_random_unitary_from_data,
                                              monkeypatch):
    data = load_measurement_matrices_data
    number_samples, max_iter, l, p = measurement_matrices_configurations
    heaviside_filtered_frames = construct_filtered_heaviside_frame


    # Stubbing:
    # ---------

    dicts = mock_random_heaviside

    monkeypatch.setattr(np.random, "random", stub_random_random_from_data(dicts[0]))
    monkeypatch.setattr(np.random, "binomial", stub_random_binomial_from_data(dicts[1]))
    monkeypatch.setattr(random, "sample", stub_random_sample_from_data(dicts[2]))
    monkeypatch.setattr(np.random, "choice", stub_random_choice_from_data(dicts[3]))
    monkeypatch.setattr(np.random, "randn", stub_random_randn_from_data(dicts[4]))
    monkeypatch.setattr(scipy.stats.unitary_group, "rvs", stub_random_unitary_from_data(dicts[5]))


    # Preliminary constructions:
    # --------------------------

    a0_heaviside1_filtered1 = heaviside_filtered_frames[0]
    a0_heaviside2_filtered1 = heaviside_filtered_frames[1]
    a0_heaviside3_filtered1 = heaviside_filtered_frames[2]

    a0_heaviside1_filtered2 = heaviside_filtered_frames[3]
    a0_heaviside2_filtered2 = heaviside_filtered_frames[4]
    a0_heaviside3_filtered2 = heaviside_filtered_frames[5]


    a0_heaviside_overcomplete1_filtered1 = heaviside_filtered_frames[6]
    a0_heaviside_overcomplete2_filtered1 = heaviside_filtered_frames[7]
    a0_heaviside_overcomplete3_filtered1 = heaviside_filtered_frames[8]

    a0_heaviside_overcomplete1_filtered2 = heaviside_filtered_frames[9]
    a0_heaviside_overcomplete2_filtered2 = heaviside_filtered_frames[10]
    a0_heaviside_overcomplete3_filtered2 = heaviside_filtered_frames[11]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............


    csMM_heaviside11 = MeasurementMatrices(a0_heaviside1_filtered1, number_samples)
    csMM_heaviside21 = MeasurementMatrices(a0_heaviside2_filtered1, number_samples)
    csMM_heaviside31 = MeasurementMatrices(a0_heaviside3_filtered1, number_samples)

    csMM_heaviside12 = MeasurementMatrices(a0_heaviside1_filtered2, number_samples)
    csMM_heaviside22 = MeasurementMatrices(a0_heaviside2_filtered2, number_samples)
    csMM_heaviside32 = MeasurementMatrices(a0_heaviside3_filtered2, number_samples)


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

    csMM_heaviside_overcomplete11 = MeasurementMatrices(a0_heaviside_overcomplete1_filtered1, number_samples)
    csMM_heaviside_overcomplete21 = MeasurementMatrices(a0_heaviside_overcomplete2_filtered1, number_samples)
    csMM_heaviside_overcomplete31 = MeasurementMatrices(a0_heaviside_overcomplete3_filtered1, number_samples)

    csMM_heaviside_overcomplete12 = MeasurementMatrices(a0_heaviside_overcomplete1_filtered2, number_samples)
    csMM_heaviside_overcomplete22 = MeasurementMatrices(a0_heaviside_overcomplete2_filtered2, number_samples)
    csMM_heaviside_overcomplete32 = MeasurementMatrices(a0_heaviside_overcomplete3_filtered2, number_samples)

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


    assert np.array_equal(ar_heaviside11_gauss, data["ar_heaviside11_gauss"])
    assert np.array_equal(ar_heaviside21_gauss, data["ar_heaviside21_gauss"])
    assert np.array_equal(ar_heaviside31_gauss, data["ar_heaviside31_gauss"])

    assert np.array_equal(ar_heaviside11_bernoulli, data["ar_heaviside11_bernoulli"])
    assert np.array_equal(ar_heaviside21_bernoulli, data["ar_heaviside21_bernoulli"])
    assert np.array_equal(ar_heaviside31_bernoulli, data["ar_heaviside31_bernoulli"])

    assert np.array_equal(ar_heaviside11_partial_fourier, data["ar_heaviside11_partial_fourier"])
    assert np.array_equal(ar_heaviside21_partial_fourier, data["ar_heaviside21_partial_fourier"])
    assert np.array_equal(ar_heaviside31_partial_fourier, data["ar_heaviside31_partial_fourier"])

    assert np.array_equal(ar_heaviside11_partial_dct, data["ar_heaviside11_partial_dct"])
    assert np.array_equal(ar_heaviside21_partial_dct, data["ar_heaviside21_partial_dct"])
    assert np.array_equal(ar_heaviside31_partial_dct, data["ar_heaviside31_partial_dct"])

    assert np.array_equal(ar_heaviside11_toeplitz, data["ar_heaviside11_toeplitz"])
    assert np.array_equal(ar_heaviside21_toeplitz, data["ar_heaviside21_toeplitz"])
    assert np.array_equal(ar_heaviside31_toeplitz, data["ar_heaviside31_toeplitz"])

    assert np.array_equal(ar_heaviside11_binary_block, data["ar_heaviside11_binary_block"])
    assert np.array_equal(ar_heaviside21_binary_block, data["ar_heaviside21_binary_block"])
    assert np.array_equal(ar_heaviside31_binary_block, data["ar_heaviside31_binary_block"])

    assert np.array_equal(ar_heaviside11_sgn, data["ar_heaviside11_sgn"])
    assert np.array_equal(ar_heaviside21_sgn, data["ar_heaviside21_sgn"])
    assert np.array_equal(ar_heaviside31_sgn, data["ar_heaviside31_sgn"])

    assert np.array_equal(ar_heaviside11_gdo, data["ar_heaviside11_gdo"])
    assert np.array_equal(ar_heaviside21_gdo, data["ar_heaviside21_gdo"])
    assert np.array_equal(ar_heaviside31_gdo, data["ar_heaviside31_gdo"])

    assert np.array_equal(ar_heaviside12_gdo, data["ar_heaviside12_gdo"])
    assert np.array_equal(ar_heaviside22_gdo, data["ar_heaviside22_gdo"])
    assert np.array_equal(ar_heaviside32_gdo, data["ar_heaviside32_gdo"])

    assert np.array_equal(ar_heaviside11_gdo_adaptive, data["ar_heaviside11_gdo_adaptive"])
    assert np.array_equal(ar_heaviside21_gdo_adaptive, data["ar_heaviside21_gdo_adaptive"])
    assert np.array_equal(ar_heaviside31_gdo_adaptive, data["ar_heaviside31_gdo_adaptive"])

    assert np.array_equal(ar_heaviside12_gdo_adaptive, data["ar_heaviside12_gdo_adaptive"])
    assert np.array_equal(ar_heaviside22_gdo_adaptive, data["ar_heaviside22_gdo_adaptive"])
    assert np.array_equal(ar_heaviside32_gdo_adaptive, data["ar_heaviside32_gdo_adaptive"])

    assert np.array_equal(ar_heaviside11_ajs, data["ar_heaviside11_ajs"])
    assert np.array_equal(ar_heaviside21_ajs, data["ar_heaviside21_ajs"])
    assert np.array_equal(ar_heaviside31_ajs, data["ar_heaviside31_ajs"])

    assert np.array_equal(ar_heaviside11_afms, data["ar_heaviside11_afms"])
    assert np.array_equal(ar_heaviside21_afms, data["ar_heaviside21_afms"])
    assert np.array_equal(ar_heaviside31_afms, data["ar_heaviside31_afms"])

    assert np.array_equal(ar_heaviside12_afms, data["ar_heaviside12_afms"])
    assert np.array_equal(ar_heaviside22_afms, data["ar_heaviside22_afms"])
    assert np.array_equal(ar_heaviside32_afms, data["ar_heaviside32_afms"])

    assert np.array_equal(ar_heaviside11_hblz, data["ar_heaviside11_hblz"])
    assert np.array_equal(ar_heaviside21_hblz, data["ar_heaviside21_hblz"])
    assert np.array_equal(ar_heaviside31_hblz, data["ar_heaviside31_hblz"])

    assert np.array_equal(ar_heaviside12_hblz, data["ar_heaviside12_hblz"])
    assert np.array_equal(ar_heaviside22_hblz, data["ar_heaviside22_hblz"])
    assert np.array_equal(ar_heaviside32_hblz, data["ar_heaviside32_hblz"])

    assert np.array_equal(ar_heaviside11_ycwg, data["ar_heaviside11_ycwg"])
    assert np.array_equal(ar_heaviside21_ycwg, data["ar_heaviside21_ycwg"])
    assert np.array_equal(ar_heaviside31_ycwg, data["ar_heaviside31_ycwg"])

    assert np.array_equal(ar_heaviside12_ycwg, data["ar_heaviside12_ycwg"])
    assert np.array_equal(ar_heaviside22_ycwg, data["ar_heaviside22_ycwg"])
    assert np.array_equal(ar_heaviside32_ycwg, data["ar_heaviside32_ycwg"])

    assert np.array_equal(ar_heaviside11_xsfz, data["ar_heaviside11_xsfz"])
    assert np.array_equal(ar_heaviside21_xsfz, data["ar_heaviside21_xsfz"])
    assert np.array_equal(ar_heaviside31_xsfz, data["ar_heaviside31_xsfz"])

    assert np.array_equal(ar_heaviside12_xsfz, data["ar_heaviside12_xsfz"])
    assert np.array_equal(ar_heaviside22_xsfz, data["ar_heaviside22_xsfz"])
    assert np.array_equal(ar_heaviside32_xsfz, data["ar_heaviside32_xsfz"])


    assert np.array_equal(ar_heaviside_overcomplete11_gauss, data["ar_heaviside_overcomplete11_gauss"])
    assert np.array_equal(ar_heaviside_overcomplete21_gauss, data["ar_heaviside_overcomplete21_gauss"])
    assert np.array_equal(ar_heaviside_overcomplete31_gauss, data["ar_heaviside_overcomplete31_gauss"])

    assert np.array_equal(ar_heaviside_overcomplete11_bernoulli, data["ar_heaviside_overcomplete11_bernoulli"])
    assert np.array_equal(ar_heaviside_overcomplete21_bernoulli, data["ar_heaviside_overcomplete21_bernoulli"])
    assert np.array_equal(ar_heaviside_overcomplete31_bernoulli, data["ar_heaviside_overcomplete31_bernoulli"])

    assert np.array_equal(ar_heaviside_overcomplete11_partial_fourier, data["ar_heaviside_overcomplete11_partial_fourier"])
    assert np.array_equal(ar_heaviside_overcomplete21_partial_fourier, data["ar_heaviside_overcomplete21_partial_fourier"])
    assert np.array_equal(ar_heaviside_overcomplete31_partial_fourier, data["ar_heaviside_overcomplete31_partial_fourier"])

    assert np.array_equal(ar_heaviside_overcomplete11_partial_dct, data["ar_heaviside_overcomplete11_partial_dct"])
    assert np.array_equal(ar_heaviside_overcomplete21_partial_dct, data["ar_heaviside_overcomplete21_partial_dct"])
    assert np.array_equal(ar_heaviside_overcomplete31_partial_dct, data["ar_heaviside_overcomplete31_partial_dct"])

    assert np.array_equal(ar_heaviside_overcomplete11_toeplitz, data["ar_heaviside_overcomplete11_toeplitz"])
    assert np.array_equal(ar_heaviside_overcomplete21_toeplitz, data["ar_heaviside_overcomplete21_toeplitz"])
    assert np.array_equal(ar_heaviside_overcomplete31_toeplitz, data["ar_heaviside_overcomplete31_toeplitz"])

    assert np.array_equal(ar_heaviside_overcomplete11_binary_block, data["ar_heaviside_overcomplete11_binary_block"])
    assert np.array_equal(ar_heaviside_overcomplete21_binary_block, data["ar_heaviside_overcomplete21_binary_block"])
    assert np.array_equal(ar_heaviside_overcomplete31_binary_block, data["ar_heaviside_overcomplete31_binary_block"])

    assert np.array_equal(ar_heaviside_overcomplete11_sgn, data["ar_heaviside_overcomplete11_sgn"])
    assert np.array_equal(ar_heaviside_overcomplete21_sgn, data["ar_heaviside_overcomplete21_sgn"])
    assert np.array_equal(ar_heaviside_overcomplete31_sgn, data["ar_heaviside_overcomplete31_sgn"])

    assert np.array_equal(ar_heaviside_overcomplete11_gdo, data["ar_heaviside_overcomplete11_gdo"])
    assert np.array_equal(ar_heaviside_overcomplete21_gdo, data["ar_heaviside_overcomplete21_gdo"])
    assert np.array_equal(ar_heaviside_overcomplete31_gdo, data["ar_heaviside_overcomplete31_gdo"])

    assert np.array_equal(ar_heaviside_overcomplete12_gdo, data["ar_heaviside_overcomplete12_gdo"])
    assert np.array_equal(ar_heaviside_overcomplete22_gdo, data["ar_heaviside_overcomplete22_gdo"])
    assert np.array_equal(ar_heaviside_overcomplete32_gdo, data["ar_heaviside_overcomplete32_gdo"])

    assert np.array_equal(ar_heaviside_overcomplete11_gdo_adaptive, data["ar_heaviside_overcomplete11_gdo_adaptive"])
    assert np.array_equal(ar_heaviside_overcomplete21_gdo_adaptive, data["ar_heaviside_overcomplete21_gdo_adaptive"])
    assert np.array_equal(ar_heaviside_overcomplete31_gdo_adaptive, data["ar_heaviside_overcomplete31_gdo_adaptive"])

    assert np.array_equal(ar_heaviside_overcomplete12_gdo_adaptive, data["ar_heaviside_overcomplete12_gdo_adaptive"])
    assert np.array_equal(ar_heaviside_overcomplete22_gdo_adaptive, data["ar_heaviside_overcomplete22_gdo_adaptive"])
    assert np.array_equal(ar_heaviside_overcomplete32_gdo_adaptive, data["ar_heaviside_overcomplete32_gdo_adaptive"])

    assert np.array_equal(ar_heaviside_overcomplete11_ajs, data["ar_heaviside_overcomplete11_ajs"])
    assert np.array_equal(ar_heaviside_overcomplete21_ajs, data["ar_heaviside_overcomplete21_ajs"])
    assert np.array_equal(ar_heaviside_overcomplete31_ajs, data["ar_heaviside_overcomplete31_ajs"])

    assert np.array_equal(ar_heaviside_overcomplete11_afms, data["ar_heaviside_overcomplete11_afms"])
    assert np.array_equal(ar_heaviside_overcomplete21_afms, data["ar_heaviside_overcomplete21_afms"])
    assert np.array_equal(ar_heaviside_overcomplete31_afms, data["ar_heaviside_overcomplete31_afms"])

    assert np.array_equal(ar_heaviside_overcomplete12_afms, data["ar_heaviside_overcomplete12_afms"])
    assert np.array_equal(ar_heaviside_overcomplete22_afms, data["ar_heaviside_overcomplete22_afms"])
    assert np.array_equal(ar_heaviside_overcomplete32_afms, data["ar_heaviside_overcomplete32_afms"])

    assert np.array_equal(ar_heaviside_overcomplete11_hblz, data["ar_heaviside_overcomplete11_hblz"])
    assert np.array_equal(ar_heaviside_overcomplete21_hblz, data["ar_heaviside_overcomplete21_hblz"])
    assert np.array_equal(ar_heaviside_overcomplete31_hblz, data["ar_heaviside_overcomplete31_hblz"])

    assert np.array_equal(ar_heaviside_overcomplete12_hblz, data["ar_heaviside_overcomplete12_hblz"])
    assert np.array_equal(ar_heaviside_overcomplete22_hblz, data["ar_heaviside_overcomplete22_hblz"])
    assert np.array_equal(ar_heaviside_overcomplete32_hblz, data["ar_heaviside_overcomplete32_hblz"])

    assert np.array_equal(ar_heaviside_overcomplete11_ycwg, data["ar_heaviside_overcomplete11_ycwg"])
    assert np.array_equal(ar_heaviside_overcomplete21_ycwg, data["ar_heaviside_overcomplete21_ycwg"])
    assert np.array_equal(ar_heaviside_overcomplete31_ycwg, data["ar_heaviside_overcomplete31_ycwg"])

    assert np.array_equal(ar_heaviside_overcomplete12_ycwg, data["ar_heaviside_overcomplete12_ycwg"])
    assert np.array_equal(ar_heaviside_overcomplete22_ycwg, data["ar_heaviside_overcomplete22_ycwg"])
    assert np.array_equal(ar_heaviside_overcomplete32_ycwg, data["ar_heaviside_overcomplete32_ycwg"])

    assert np.array_equal(ar_heaviside_overcomplete11_xsfz, data["ar_heaviside_overcomplete11_xsfz"])
    assert np.array_equal(ar_heaviside_overcomplete21_xsfz, data["ar_heaviside_overcomplete21_xsfz"])
    assert np.array_equal(ar_heaviside_overcomplete31_xsfz, data["ar_heaviside_overcomplete31_xsfz"])

    assert np.array_equal(ar_heaviside_overcomplete12_xsfz, data["ar_heaviside_overcomplete12_xsfz"])
    assert np.array_equal(ar_heaviside_overcomplete22_xsfz, data["ar_heaviside_overcomplete22_xsfz"])
    assert np.array_equal(ar_heaviside_overcomplete32_xsfz, data["ar_heaviside_overcomplete32_xsfz"])


@pytest.mark.parametrize("frame_name", ["gaussian"], indirect=True)
def test_gaussian_based_measurement_matrices(load_measurement_matrices_data,
                                              construct_filtered_gaussian_frame,
                                              measurement_matrices_configurations,
                                              mock_random_gaussian,
                                              stub_random_random_from_data,
                                              stub_random_binomial_from_data,
                                              stub_random_sample_from_data,
                                              stub_random_choice_from_data,
                                              stub_random_randn_from_data,
                                              stub_random_unitary_from_data,
                                              monkeypatch):
    data = load_measurement_matrices_data
    number_samples, max_iter, l, p = measurement_matrices_configurations
    gaussian_filtered_frames = construct_filtered_gaussian_frame


    # Stubbing:
    # ---------

    dicts = mock_random_gaussian

    monkeypatch.setattr(np.random, "random", stub_random_random_from_data(dicts[0]))
    monkeypatch.setattr(np.random, "binomial", stub_random_binomial_from_data(dicts[1]))
    monkeypatch.setattr(random, "sample", stub_random_sample_from_data(dicts[2]))
    monkeypatch.setattr(np.random, "choice", stub_random_choice_from_data(dicts[3]))
    monkeypatch.setattr(np.random, "randn", stub_random_randn_from_data(dicts[4]))
    monkeypatch.setattr(scipy.stats.unitary_group, "rvs", stub_random_unitary_from_data(dicts[5]))


    # Preliminary constructions:
    # --------------------------

    a0_gaussian1_filtered1 = gaussian_filtered_frames[0]
    a0_gaussian2_filtered1 = gaussian_filtered_frames[1]
    a0_gaussian3_filtered1 = gaussian_filtered_frames[2]

    a0_gaussian1_filtered2 = gaussian_filtered_frames[3]
    a0_gaussian2_filtered2 = gaussian_filtered_frames[4]
    a0_gaussian3_filtered2 = gaussian_filtered_frames[5]


    a0_gaussian_overcomplete1_filtered1 = gaussian_filtered_frames[6]
    a0_gaussian_overcomplete2_filtered1 = gaussian_filtered_frames[7]
    a0_gaussian_overcomplete3_filtered1 = gaussian_filtered_frames[8]

    a0_gaussian_overcomplete1_filtered2 = gaussian_filtered_frames[9]
    a0_gaussian_overcomplete2_filtered2 = gaussian_filtered_frames[10]
    a0_gaussian_overcomplete3_filtered2 = gaussian_filtered_frames[11]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............


    csMM_gaussian11 = MeasurementMatrices(a0_gaussian1_filtered1, number_samples)
    csMM_gaussian21 = MeasurementMatrices(a0_gaussian2_filtered1, number_samples)
    csMM_gaussian31 = MeasurementMatrices(a0_gaussian3_filtered1, number_samples)

    csMM_gaussian12 = MeasurementMatrices(a0_gaussian1_filtered2, number_samples)
    csMM_gaussian22 = MeasurementMatrices(a0_gaussian2_filtered2, number_samples)
    csMM_gaussian32 = MeasurementMatrices(a0_gaussian3_filtered2, number_samples)


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

    csMM_gaussian_overcomplete11 = MeasurementMatrices(a0_gaussian_overcomplete1_filtered1, number_samples)
    csMM_gaussian_overcomplete21 = MeasurementMatrices(a0_gaussian_overcomplete2_filtered1, number_samples)
    csMM_gaussian_overcomplete31 = MeasurementMatrices(a0_gaussian_overcomplete3_filtered1, number_samples)

    csMM_gaussian_overcomplete12 = MeasurementMatrices(a0_gaussian_overcomplete1_filtered2, number_samples)
    csMM_gaussian_overcomplete22 = MeasurementMatrices(a0_gaussian_overcomplete2_filtered2, number_samples)
    csMM_gaussian_overcomplete32 = MeasurementMatrices(a0_gaussian_overcomplete3_filtered2, number_samples)

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


    assert np.array_equal(ar_gaussian11_gauss, data["ar_gaussian11_gauss"])
    assert np.array_equal(ar_gaussian21_gauss, data["ar_gaussian21_gauss"])
    assert np.array_equal(ar_gaussian31_gauss, data["ar_gaussian31_gauss"])

    assert np.array_equal(ar_gaussian11_bernoulli, data["ar_gaussian11_bernoulli"])
    assert np.array_equal(ar_gaussian21_bernoulli, data["ar_gaussian21_bernoulli"])
    assert np.array_equal(ar_gaussian31_bernoulli, data["ar_gaussian31_bernoulli"])

    assert np.array_equal(ar_gaussian11_partial_fourier, data["ar_gaussian11_partial_fourier"])
    assert np.array_equal(ar_gaussian21_partial_fourier, data["ar_gaussian21_partial_fourier"])
    assert np.array_equal(ar_gaussian31_partial_fourier, data["ar_gaussian31_partial_fourier"])

    assert np.array_equal(ar_gaussian11_partial_dct, data["ar_gaussian11_partial_dct"])
    assert np.array_equal(ar_gaussian21_partial_dct, data["ar_gaussian21_partial_dct"])
    assert np.array_equal(ar_gaussian31_partial_dct, data["ar_gaussian31_partial_dct"])

    assert np.array_equal(ar_gaussian11_toeplitz, data["ar_gaussian11_toeplitz"])
    assert np.array_equal(ar_gaussian21_toeplitz, data["ar_gaussian21_toeplitz"])
    assert np.array_equal(ar_gaussian31_toeplitz, data["ar_gaussian31_toeplitz"])

    assert np.array_equal(ar_gaussian11_binary_block, data["ar_gaussian11_binary_block"])
    assert np.array_equal(ar_gaussian21_binary_block, data["ar_gaussian21_binary_block"])
    assert np.array_equal(ar_gaussian31_binary_block, data["ar_gaussian31_binary_block"])

    assert np.array_equal(ar_gaussian11_sgn, data["ar_gaussian11_sgn"])
    assert np.array_equal(ar_gaussian21_sgn, data["ar_gaussian21_sgn"])
    assert np.array_equal(ar_gaussian31_sgn, data["ar_gaussian31_sgn"])

    assert np.array_equal(ar_gaussian11_gdo, data["ar_gaussian11_gdo"])
    assert np.array_equal(ar_gaussian21_gdo, data["ar_gaussian21_gdo"])
    assert np.array_equal(ar_gaussian31_gdo, data["ar_gaussian31_gdo"])

    assert np.array_equal(ar_gaussian12_gdo, data["ar_gaussian12_gdo"])
    assert np.array_equal(ar_gaussian22_gdo, data["ar_gaussian22_gdo"])
    assert np.array_equal(ar_gaussian32_gdo, data["ar_gaussian32_gdo"])

    assert np.array_equal(ar_gaussian11_gdo_adaptive, data["ar_gaussian11_gdo_adaptive"])
    assert np.array_equal(ar_gaussian21_gdo_adaptive, data["ar_gaussian21_gdo_adaptive"])
    assert np.array_equal(ar_gaussian31_gdo_adaptive, data["ar_gaussian31_gdo_adaptive"])

    assert np.array_equal(ar_gaussian12_gdo_adaptive, data["ar_gaussian12_gdo_adaptive"])
    assert np.array_equal(ar_gaussian22_gdo_adaptive, data["ar_gaussian22_gdo_adaptive"])
    assert np.array_equal(ar_gaussian32_gdo_adaptive, data["ar_gaussian32_gdo_adaptive"])

    assert np.array_equal(ar_gaussian11_ajs, data["ar_gaussian11_ajs"])
    assert np.array_equal(ar_gaussian21_ajs, data["ar_gaussian21_ajs"])
    assert np.array_equal(ar_gaussian31_ajs, data["ar_gaussian31_ajs"])

    assert np.array_equal(ar_gaussian11_afms, data["ar_gaussian11_afms"])
    assert np.array_equal(ar_gaussian21_afms, data["ar_gaussian21_afms"])
    assert np.array_equal(ar_gaussian31_afms, data["ar_gaussian31_afms"])

    assert np.array_equal(ar_gaussian12_afms, data["ar_gaussian12_afms"])
    assert np.array_equal(ar_gaussian22_afms, data["ar_gaussian22_afms"])
    assert np.array_equal(ar_gaussian32_afms, data["ar_gaussian32_afms"])

    assert np.array_equal(ar_gaussian11_hblz, data["ar_gaussian11_hblz"])
    assert np.array_equal(ar_gaussian21_hblz, data["ar_gaussian21_hblz"])
    assert np.array_equal(ar_gaussian31_hblz, data["ar_gaussian31_hblz"])

    assert np.array_equal(ar_gaussian12_hblz, data["ar_gaussian12_hblz"])
    assert np.array_equal(ar_gaussian22_hblz, data["ar_gaussian22_hblz"])
    assert np.array_equal(ar_gaussian32_hblz, data["ar_gaussian32_hblz"])

    assert np.array_equal(ar_gaussian11_ycwg, data["ar_gaussian11_ycwg"])
    assert np.array_equal(ar_gaussian21_ycwg, data["ar_gaussian21_ycwg"])
    assert np.array_equal(ar_gaussian31_ycwg, data["ar_gaussian31_ycwg"])

    assert np.array_equal(ar_gaussian12_ycwg, data["ar_gaussian12_ycwg"])
    assert np.array_equal(ar_gaussian22_ycwg, data["ar_gaussian22_ycwg"])
    assert np.array_equal(ar_gaussian32_ycwg, data["ar_gaussian32_ycwg"])

    assert np.array_equal(ar_gaussian11_xsfz, data["ar_gaussian11_xsfz"])
    assert np.array_equal(ar_gaussian21_xsfz, data["ar_gaussian21_xsfz"])
    assert np.array_equal(ar_gaussian31_xsfz, data["ar_gaussian31_xsfz"])

    assert np.array_equal(ar_gaussian12_xsfz, data["ar_gaussian12_xsfz"])
    assert np.array_equal(ar_gaussian22_xsfz, data["ar_gaussian22_xsfz"])
    assert np.array_equal(ar_gaussian32_xsfz, data["ar_gaussian32_xsfz"])


    assert np.array_equal(ar_gaussian_overcomplete11_gauss, data["ar_gaussian_overcomplete11_gauss"])
    assert np.array_equal(ar_gaussian_overcomplete21_gauss, data["ar_gaussian_overcomplete21_gauss"])
    assert np.array_equal(ar_gaussian_overcomplete31_gauss, data["ar_gaussian_overcomplete31_gauss"])

    assert np.array_equal(ar_gaussian_overcomplete11_bernoulli, data["ar_gaussian_overcomplete11_bernoulli"])
    assert np.array_equal(ar_gaussian_overcomplete21_bernoulli, data["ar_gaussian_overcomplete21_bernoulli"])
    assert np.array_equal(ar_gaussian_overcomplete31_bernoulli, data["ar_gaussian_overcomplete31_bernoulli"])

    assert np.array_equal(ar_gaussian_overcomplete11_partial_fourier, data["ar_gaussian_overcomplete11_partial_fourier"])
    assert np.array_equal(ar_gaussian_overcomplete21_partial_fourier, data["ar_gaussian_overcomplete21_partial_fourier"])
    assert np.array_equal(ar_gaussian_overcomplete31_partial_fourier, data["ar_gaussian_overcomplete31_partial_fourier"])

    assert np.array_equal(ar_gaussian_overcomplete11_partial_dct, data["ar_gaussian_overcomplete11_partial_dct"])
    assert np.array_equal(ar_gaussian_overcomplete21_partial_dct, data["ar_gaussian_overcomplete21_partial_dct"])
    assert np.array_equal(ar_gaussian_overcomplete31_partial_dct, data["ar_gaussian_overcomplete31_partial_dct"])

    assert np.array_equal(ar_gaussian_overcomplete11_toeplitz, data["ar_gaussian_overcomplete11_toeplitz"])
    assert np.array_equal(ar_gaussian_overcomplete21_toeplitz, data["ar_gaussian_overcomplete21_toeplitz"])
    assert np.array_equal(ar_gaussian_overcomplete31_toeplitz, data["ar_gaussian_overcomplete31_toeplitz"])

    assert np.array_equal(ar_gaussian_overcomplete11_binary_block, data["ar_gaussian_overcomplete11_binary_block"])
    assert np.array_equal(ar_gaussian_overcomplete21_binary_block, data["ar_gaussian_overcomplete21_binary_block"])
    assert np.array_equal(ar_gaussian_overcomplete31_binary_block, data["ar_gaussian_overcomplete31_binary_block"])

    assert np.array_equal(ar_gaussian_overcomplete11_sgn, data["ar_gaussian_overcomplete11_sgn"])
    assert np.array_equal(ar_gaussian_overcomplete21_sgn, data["ar_gaussian_overcomplete21_sgn"])
    assert np.array_equal(ar_gaussian_overcomplete31_sgn, data["ar_gaussian_overcomplete31_sgn"])

    assert np.array_equal(ar_gaussian_overcomplete11_gdo, data["ar_gaussian_overcomplete11_gdo"])
    assert np.array_equal(ar_gaussian_overcomplete21_gdo, data["ar_gaussian_overcomplete21_gdo"])
    assert np.array_equal(ar_gaussian_overcomplete31_gdo, data["ar_gaussian_overcomplete31_gdo"])

    assert np.array_equal(ar_gaussian_overcomplete12_gdo, data["ar_gaussian_overcomplete12_gdo"])
    assert np.array_equal(ar_gaussian_overcomplete22_gdo, data["ar_gaussian_overcomplete22_gdo"])
    assert np.array_equal(ar_gaussian_overcomplete32_gdo, data["ar_gaussian_overcomplete32_gdo"])

    assert np.array_equal(ar_gaussian_overcomplete11_gdo_adaptive, data["ar_gaussian_overcomplete11_gdo_adaptive"])
    assert np.array_equal(ar_gaussian_overcomplete21_gdo_adaptive, data["ar_gaussian_overcomplete21_gdo_adaptive"])
    assert np.array_equal(ar_gaussian_overcomplete31_gdo_adaptive, data["ar_gaussian_overcomplete31_gdo_adaptive"])

    assert np.array_equal(ar_gaussian_overcomplete12_gdo_adaptive, data["ar_gaussian_overcomplete12_gdo_adaptive"])
    assert np.array_equal(ar_gaussian_overcomplete22_gdo_adaptive, data["ar_gaussian_overcomplete22_gdo_adaptive"])
    assert np.array_equal(ar_gaussian_overcomplete32_gdo_adaptive, data["ar_gaussian_overcomplete32_gdo_adaptive"])

    assert np.array_equal(ar_gaussian_overcomplete11_ajs, data["ar_gaussian_overcomplete11_ajs"])
    assert np.array_equal(ar_gaussian_overcomplete21_ajs, data["ar_gaussian_overcomplete21_ajs"])
    assert np.array_equal(ar_gaussian_overcomplete31_ajs, data["ar_gaussian_overcomplete31_ajs"])

    assert np.array_equal(ar_gaussian_overcomplete11_afms, data["ar_gaussian_overcomplete11_afms"])
    assert np.array_equal(ar_gaussian_overcomplete21_afms, data["ar_gaussian_overcomplete21_afms"])
    assert np.array_equal(ar_gaussian_overcomplete31_afms, data["ar_gaussian_overcomplete31_afms"])

    assert np.array_equal(ar_gaussian_overcomplete12_afms, data["ar_gaussian_overcomplete12_afms"])
    assert np.array_equal(ar_gaussian_overcomplete22_afms, data["ar_gaussian_overcomplete22_afms"])
    assert np.array_equal(ar_gaussian_overcomplete32_afms, data["ar_gaussian_overcomplete32_afms"])

    assert np.array_equal(ar_gaussian_overcomplete11_hblz, data["ar_gaussian_overcomplete11_hblz"])
    assert np.array_equal(ar_gaussian_overcomplete21_hblz, data["ar_gaussian_overcomplete21_hblz"])
    assert np.array_equal(ar_gaussian_overcomplete31_hblz, data["ar_gaussian_overcomplete31_hblz"])

    assert np.array_equal(ar_gaussian_overcomplete12_hblz, data["ar_gaussian_overcomplete12_hblz"])
    assert np.array_equal(ar_gaussian_overcomplete22_hblz, data["ar_gaussian_overcomplete22_hblz"])
    assert np.array_equal(ar_gaussian_overcomplete32_hblz, data["ar_gaussian_overcomplete32_hblz"])

    assert np.array_equal(ar_gaussian_overcomplete11_ycwg, data["ar_gaussian_overcomplete11_ycwg"])
    assert np.array_equal(ar_gaussian_overcomplete21_ycwg, data["ar_gaussian_overcomplete21_ycwg"])
    assert np.array_equal(ar_gaussian_overcomplete31_ycwg, data["ar_gaussian_overcomplete31_ycwg"])

    assert np.array_equal(ar_gaussian_overcomplete12_ycwg, data["ar_gaussian_overcomplete12_ycwg"])
    assert np.array_equal(ar_gaussian_overcomplete22_ycwg, data["ar_gaussian_overcomplete22_ycwg"])
    assert np.array_equal(ar_gaussian_overcomplete32_ycwg, data["ar_gaussian_overcomplete32_ycwg"])

    assert np.array_equal(ar_gaussian_overcomplete11_xsfz, data["ar_gaussian_overcomplete11_xsfz"])
    assert np.array_equal(ar_gaussian_overcomplete21_xsfz, data["ar_gaussian_overcomplete21_xsfz"])
    assert np.array_equal(ar_gaussian_overcomplete31_xsfz, data["ar_gaussian_overcomplete31_xsfz"])

    assert np.array_equal(ar_gaussian_overcomplete12_xsfz, data["ar_gaussian_overcomplete12_xsfz"])
    assert np.array_equal(ar_gaussian_overcomplete22_xsfz, data["ar_gaussian_overcomplete22_xsfz"])
    assert np.array_equal(ar_gaussian_overcomplete32_xsfz, data["ar_gaussian_overcomplete32_xsfz"])


@pytest.mark.parametrize("frame_name", ["cauchy"], indirect=True)
def test_cauchy_based_measurement_matrices(load_measurement_matrices_data,
                                              construct_filtered_cauchy_frame,
                                              measurement_matrices_configurations,
                                              mock_random_cauchy,
                                              stub_random_random_from_data,
                                              stub_random_binomial_from_data,
                                              stub_random_sample_from_data,
                                              stub_random_choice_from_data,
                                              stub_random_randn_from_data,
                                              stub_random_unitary_from_data,
                                              monkeypatch):
    data = load_measurement_matrices_data
    number_samples, max_iter, l, p = measurement_matrices_configurations
    cauchy_filtered_frames = construct_filtered_cauchy_frame


    # Stubbing:
    # ---------

    dicts = mock_random_cauchy

    monkeypatch.setattr(np.random, "random", stub_random_random_from_data(dicts[0]))
    monkeypatch.setattr(np.random, "binomial", stub_random_binomial_from_data(dicts[1]))
    monkeypatch.setattr(random, "sample", stub_random_sample_from_data(dicts[2]))
    monkeypatch.setattr(np.random, "choice", stub_random_choice_from_data(dicts[3]))
    monkeypatch.setattr(np.random, "randn", stub_random_randn_from_data(dicts[4]))
    monkeypatch.setattr(scipy.stats.unitary_group, "rvs", stub_random_unitary_from_data(dicts[5]))


    # Preliminary constructions:
    # --------------------------

    a0_cauchy1_filtered1 = cauchy_filtered_frames[0]
    a0_cauchy2_filtered1 = cauchy_filtered_frames[1]
    a0_cauchy3_filtered1 = cauchy_filtered_frames[2]

    a0_cauchy1_filtered2 = cauchy_filtered_frames[3]
    a0_cauchy2_filtered2 = cauchy_filtered_frames[4]
    a0_cauchy3_filtered2 = cauchy_filtered_frames[5]


    a0_cauchy_overcomplete1_filtered1 = cauchy_filtered_frames[6]
    a0_cauchy_overcomplete2_filtered1 = cauchy_filtered_frames[7]
    a0_cauchy_overcomplete3_filtered1 = cauchy_filtered_frames[8]

    a0_cauchy_overcomplete1_filtered2 = cauchy_filtered_frames[9]
    a0_cauchy_overcomplete2_filtered2 = cauchy_filtered_frames[10]
    a0_cauchy_overcomplete3_filtered2 = cauchy_filtered_frames[11]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............


    csMM_cauchy11 = MeasurementMatrices(a0_cauchy1_filtered1, number_samples)
    csMM_cauchy21 = MeasurementMatrices(a0_cauchy2_filtered1, number_samples)
    csMM_cauchy31 = MeasurementMatrices(a0_cauchy3_filtered1, number_samples)

    csMM_cauchy12 = MeasurementMatrices(a0_cauchy1_filtered2, number_samples)
    csMM_cauchy22 = MeasurementMatrices(a0_cauchy2_filtered2, number_samples)
    csMM_cauchy32 = MeasurementMatrices(a0_cauchy3_filtered2, number_samples)


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

    csMM_cauchy_overcomplete11 = MeasurementMatrices(a0_cauchy_overcomplete1_filtered1, number_samples)
    csMM_cauchy_overcomplete21 = MeasurementMatrices(a0_cauchy_overcomplete2_filtered1, number_samples)
    csMM_cauchy_overcomplete31 = MeasurementMatrices(a0_cauchy_overcomplete3_filtered1, number_samples)

    csMM_cauchy_overcomplete12 = MeasurementMatrices(a0_cauchy_overcomplete1_filtered2, number_samples)
    csMM_cauchy_overcomplete22 = MeasurementMatrices(a0_cauchy_overcomplete2_filtered2, number_samples)
    csMM_cauchy_overcomplete32 = MeasurementMatrices(a0_cauchy_overcomplete3_filtered2, number_samples)

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


    assert np.array_equal(ar_cauchy11_gauss, data["ar_cauchy11_gauss"])
    assert np.array_equal(ar_cauchy21_gauss, data["ar_cauchy21_gauss"])
    assert np.array_equal(ar_cauchy31_gauss, data["ar_cauchy31_gauss"])

    assert np.array_equal(ar_cauchy11_bernoulli, data["ar_cauchy11_bernoulli"])
    assert np.array_equal(ar_cauchy21_bernoulli, data["ar_cauchy21_bernoulli"])
    assert np.array_equal(ar_cauchy31_bernoulli, data["ar_cauchy31_bernoulli"])

    assert np.array_equal(ar_cauchy11_partial_fourier, data["ar_cauchy11_partial_fourier"])
    assert np.array_equal(ar_cauchy21_partial_fourier, data["ar_cauchy21_partial_fourier"])
    assert np.array_equal(ar_cauchy31_partial_fourier, data["ar_cauchy31_partial_fourier"])

    assert np.array_equal(ar_cauchy11_partial_dct, data["ar_cauchy11_partial_dct"])
    assert np.array_equal(ar_cauchy21_partial_dct, data["ar_cauchy21_partial_dct"])
    assert np.array_equal(ar_cauchy31_partial_dct, data["ar_cauchy31_partial_dct"])

    assert np.array_equal(ar_cauchy11_toeplitz, data["ar_cauchy11_toeplitz"])
    assert np.array_equal(ar_cauchy21_toeplitz, data["ar_cauchy21_toeplitz"])
    assert np.array_equal(ar_cauchy31_toeplitz, data["ar_cauchy31_toeplitz"])

    assert np.array_equal(ar_cauchy11_binary_block, data["ar_cauchy11_binary_block"])
    assert np.array_equal(ar_cauchy21_binary_block, data["ar_cauchy21_binary_block"])
    assert np.array_equal(ar_cauchy31_binary_block, data["ar_cauchy31_binary_block"])

    assert np.array_equal(ar_cauchy11_sgn, data["ar_cauchy11_sgn"])
    assert np.array_equal(ar_cauchy21_sgn, data["ar_cauchy21_sgn"])
    assert np.array_equal(ar_cauchy31_sgn, data["ar_cauchy31_sgn"])

    assert np.array_equal(ar_cauchy11_gdo, data["ar_cauchy11_gdo"])
    assert np.array_equal(ar_cauchy21_gdo, data["ar_cauchy21_gdo"])
    assert np.array_equal(ar_cauchy31_gdo, data["ar_cauchy31_gdo"])

    assert np.array_equal(ar_cauchy12_gdo, data["ar_cauchy12_gdo"])
    assert np.array_equal(ar_cauchy22_gdo, data["ar_cauchy22_gdo"])
    assert np.array_equal(ar_cauchy32_gdo, data["ar_cauchy32_gdo"])

    assert np.array_equal(ar_cauchy11_gdo_adaptive, data["ar_cauchy11_gdo_adaptive"])
    assert np.array_equal(ar_cauchy21_gdo_adaptive, data["ar_cauchy21_gdo_adaptive"])
    assert np.array_equal(ar_cauchy31_gdo_adaptive, data["ar_cauchy31_gdo_adaptive"])

    assert np.array_equal(ar_cauchy12_gdo_adaptive, data["ar_cauchy12_gdo_adaptive"])
    assert np.array_equal(ar_cauchy22_gdo_adaptive, data["ar_cauchy22_gdo_adaptive"])
    assert np.array_equal(ar_cauchy32_gdo_adaptive, data["ar_cauchy32_gdo_adaptive"])

    assert np.array_equal(ar_cauchy11_ajs, data["ar_cauchy11_ajs"])
    assert np.array_equal(ar_cauchy21_ajs, data["ar_cauchy21_ajs"])
    assert np.array_equal(ar_cauchy31_ajs, data["ar_cauchy31_ajs"])

    assert np.array_equal(ar_cauchy11_afms, data["ar_cauchy11_afms"])
    assert np.array_equal(ar_cauchy21_afms, data["ar_cauchy21_afms"])
    assert np.array_equal(ar_cauchy31_afms, data["ar_cauchy31_afms"])

    assert np.array_equal(ar_cauchy12_afms, data["ar_cauchy12_afms"])
    assert np.array_equal(ar_cauchy22_afms, data["ar_cauchy22_afms"])
    assert np.array_equal(ar_cauchy32_afms, data["ar_cauchy32_afms"])

    assert np.array_equal(ar_cauchy11_hblz, data["ar_cauchy11_hblz"])
    assert np.array_equal(ar_cauchy21_hblz, data["ar_cauchy21_hblz"])
    assert np.array_equal(ar_cauchy31_hblz, data["ar_cauchy31_hblz"])

    assert np.array_equal(ar_cauchy12_hblz, data["ar_cauchy12_hblz"])
    assert np.array_equal(ar_cauchy22_hblz, data["ar_cauchy22_hblz"])
    assert np.array_equal(ar_cauchy32_hblz, data["ar_cauchy32_hblz"])

    assert np.array_equal(ar_cauchy11_ycwg, data["ar_cauchy11_ycwg"])
    assert np.array_equal(ar_cauchy21_ycwg, data["ar_cauchy21_ycwg"])
    assert np.array_equal(ar_cauchy31_ycwg, data["ar_cauchy31_ycwg"])

    assert np.array_equal(ar_cauchy12_ycwg, data["ar_cauchy12_ycwg"])
    assert np.array_equal(ar_cauchy22_ycwg, data["ar_cauchy22_ycwg"])
    assert np.array_equal(ar_cauchy32_ycwg, data["ar_cauchy32_ycwg"])

    assert np.array_equal(ar_cauchy11_xsfz, data["ar_cauchy11_xsfz"])
    assert np.array_equal(ar_cauchy21_xsfz, data["ar_cauchy21_xsfz"])
    assert np.array_equal(ar_cauchy31_xsfz, data["ar_cauchy31_xsfz"])

    assert np.array_equal(ar_cauchy12_xsfz, data["ar_cauchy12_xsfz"])
    assert np.array_equal(ar_cauchy22_xsfz, data["ar_cauchy22_xsfz"])
    assert np.array_equal(ar_cauchy32_xsfz, data["ar_cauchy32_xsfz"])


    assert np.array_equal(ar_cauchy_overcomplete11_gauss, data["ar_cauchy_overcomplete11_gauss"])
    assert np.array_equal(ar_cauchy_overcomplete21_gauss, data["ar_cauchy_overcomplete21_gauss"])
    assert np.array_equal(ar_cauchy_overcomplete31_gauss, data["ar_cauchy_overcomplete31_gauss"])

    assert np.array_equal(ar_cauchy_overcomplete11_bernoulli, data["ar_cauchy_overcomplete11_bernoulli"])
    assert np.array_equal(ar_cauchy_overcomplete21_bernoulli, data["ar_cauchy_overcomplete21_bernoulli"])
    assert np.array_equal(ar_cauchy_overcomplete31_bernoulli, data["ar_cauchy_overcomplete31_bernoulli"])

    assert np.array_equal(ar_cauchy_overcomplete11_partial_fourier, data["ar_cauchy_overcomplete11_partial_fourier"])
    assert np.array_equal(ar_cauchy_overcomplete21_partial_fourier, data["ar_cauchy_overcomplete21_partial_fourier"])
    assert np.array_equal(ar_cauchy_overcomplete31_partial_fourier, data["ar_cauchy_overcomplete31_partial_fourier"])

    assert np.array_equal(ar_cauchy_overcomplete11_partial_dct, data["ar_cauchy_overcomplete11_partial_dct"])
    assert np.array_equal(ar_cauchy_overcomplete21_partial_dct, data["ar_cauchy_overcomplete21_partial_dct"])
    assert np.array_equal(ar_cauchy_overcomplete31_partial_dct, data["ar_cauchy_overcomplete31_partial_dct"])

    assert np.array_equal(ar_cauchy_overcomplete11_toeplitz, data["ar_cauchy_overcomplete11_toeplitz"])
    assert np.array_equal(ar_cauchy_overcomplete21_toeplitz, data["ar_cauchy_overcomplete21_toeplitz"])
    assert np.array_equal(ar_cauchy_overcomplete31_toeplitz, data["ar_cauchy_overcomplete31_toeplitz"])

    assert np.array_equal(ar_cauchy_overcomplete11_binary_block, data["ar_cauchy_overcomplete11_binary_block"])
    assert np.array_equal(ar_cauchy_overcomplete21_binary_block, data["ar_cauchy_overcomplete21_binary_block"])
    assert np.array_equal(ar_cauchy_overcomplete31_binary_block, data["ar_cauchy_overcomplete31_binary_block"])

    assert np.array_equal(ar_cauchy_overcomplete11_sgn, data["ar_cauchy_overcomplete11_sgn"])
    assert np.array_equal(ar_cauchy_overcomplete21_sgn, data["ar_cauchy_overcomplete21_sgn"])
    assert np.array_equal(ar_cauchy_overcomplete31_sgn, data["ar_cauchy_overcomplete31_sgn"])

    assert np.array_equal(ar_cauchy_overcomplete11_gdo, data["ar_cauchy_overcomplete11_gdo"])
    assert np.array_equal(ar_cauchy_overcomplete21_gdo, data["ar_cauchy_overcomplete21_gdo"])
    assert np.array_equal(ar_cauchy_overcomplete31_gdo, data["ar_cauchy_overcomplete31_gdo"])

    assert np.array_equal(ar_cauchy_overcomplete12_gdo, data["ar_cauchy_overcomplete12_gdo"])
    assert np.array_equal(ar_cauchy_overcomplete22_gdo, data["ar_cauchy_overcomplete22_gdo"])
    assert np.array_equal(ar_cauchy_overcomplete32_gdo, data["ar_cauchy_overcomplete32_gdo"])

    assert np.array_equal(ar_cauchy_overcomplete11_gdo_adaptive, data["ar_cauchy_overcomplete11_gdo_adaptive"])
    assert np.array_equal(ar_cauchy_overcomplete21_gdo_adaptive, data["ar_cauchy_overcomplete21_gdo_adaptive"])
    assert np.array_equal(ar_cauchy_overcomplete31_gdo_adaptive, data["ar_cauchy_overcomplete31_gdo_adaptive"])

    assert np.array_equal(ar_cauchy_overcomplete12_gdo_adaptive, data["ar_cauchy_overcomplete12_gdo_adaptive"])
    assert np.array_equal(ar_cauchy_overcomplete22_gdo_adaptive, data["ar_cauchy_overcomplete22_gdo_adaptive"])
    assert np.array_equal(ar_cauchy_overcomplete32_gdo_adaptive, data["ar_cauchy_overcomplete32_gdo_adaptive"])

    assert np.array_equal(ar_cauchy_overcomplete11_ajs, data["ar_cauchy_overcomplete11_ajs"])
    assert np.array_equal(ar_cauchy_overcomplete21_ajs, data["ar_cauchy_overcomplete21_ajs"])
    assert np.array_equal(ar_cauchy_overcomplete31_ajs, data["ar_cauchy_overcomplete31_ajs"])

    assert np.array_equal(ar_cauchy_overcomplete11_afms, data["ar_cauchy_overcomplete11_afms"])
    assert np.array_equal(ar_cauchy_overcomplete21_afms, data["ar_cauchy_overcomplete21_afms"])
    assert np.array_equal(ar_cauchy_overcomplete31_afms, data["ar_cauchy_overcomplete31_afms"])

    assert np.array_equal(ar_cauchy_overcomplete12_afms, data["ar_cauchy_overcomplete12_afms"])
    assert np.array_equal(ar_cauchy_overcomplete22_afms, data["ar_cauchy_overcomplete22_afms"])
    assert np.array_equal(ar_cauchy_overcomplete32_afms, data["ar_cauchy_overcomplete32_afms"])

    assert np.array_equal(ar_cauchy_overcomplete11_hblz, data["ar_cauchy_overcomplete11_hblz"])
    assert np.array_equal(ar_cauchy_overcomplete21_hblz, data["ar_cauchy_overcomplete21_hblz"])
    assert np.array_equal(ar_cauchy_overcomplete31_hblz, data["ar_cauchy_overcomplete31_hblz"])

    assert np.array_equal(ar_cauchy_overcomplete12_hblz, data["ar_cauchy_overcomplete12_hblz"])
    assert np.array_equal(ar_cauchy_overcomplete22_hblz, data["ar_cauchy_overcomplete22_hblz"])
    assert np.array_equal(ar_cauchy_overcomplete32_hblz, data["ar_cauchy_overcomplete32_hblz"])

    assert np.array_equal(ar_cauchy_overcomplete11_ycwg, data["ar_cauchy_overcomplete11_ycwg"])
    assert np.array_equal(ar_cauchy_overcomplete21_ycwg, data["ar_cauchy_overcomplete21_ycwg"])
    assert np.array_equal(ar_cauchy_overcomplete31_ycwg, data["ar_cauchy_overcomplete31_ycwg"])

    assert np.array_equal(ar_cauchy_overcomplete12_ycwg, data["ar_cauchy_overcomplete12_ycwg"])
    assert np.array_equal(ar_cauchy_overcomplete22_ycwg, data["ar_cauchy_overcomplete22_ycwg"])
    assert np.array_equal(ar_cauchy_overcomplete32_ycwg, data["ar_cauchy_overcomplete32_ycwg"])

    assert np.array_equal(ar_cauchy_overcomplete11_xsfz, data["ar_cauchy_overcomplete11_xsfz"])
    assert np.array_equal(ar_cauchy_overcomplete21_xsfz, data["ar_cauchy_overcomplete21_xsfz"])
    assert np.array_equal(ar_cauchy_overcomplete31_xsfz, data["ar_cauchy_overcomplete31_xsfz"])

    assert np.array_equal(ar_cauchy_overcomplete12_xsfz, data["ar_cauchy_overcomplete12_xsfz"])
    assert np.array_equal(ar_cauchy_overcomplete22_xsfz, data["ar_cauchy_overcomplete22_xsfz"])
    assert np.array_equal(ar_cauchy_overcomplete32_xsfz, data["ar_cauchy_overcomplete32_xsfz"])


@pytest.mark.parametrize("frame_name", ["fourier"], indirect=True)
def test_fourier_based_measurement_matrices(load_measurement_matrices_data,
                                              construct_filtered_fourier_frame,
                                              measurement_matrices_configurations,
                                              mock_random_fourier,
                                              stub_random_random_from_data,
                                              stub_random_binomial_from_data,
                                              stub_random_sample_from_data,
                                              stub_random_choice_from_data,
                                              stub_random_randn_from_data,
                                              stub_random_unitary_from_data,
                                              monkeypatch):
    data = load_measurement_matrices_data
    number_samples, max_iter, l, p = measurement_matrices_configurations
    fourier_filtered_frames = construct_filtered_fourier_frame


    # Stubbing:
    # ---------

    dicts = mock_random_fourier

    monkeypatch.setattr(np.random, "random", stub_random_random_from_data(dicts[0]))
    monkeypatch.setattr(np.random, "binomial", stub_random_binomial_from_data(dicts[1]))
    monkeypatch.setattr(random, "sample", stub_random_sample_from_data(dicts[2]))
    monkeypatch.setattr(np.random, "choice", stub_random_choice_from_data(dicts[3]))
    monkeypatch.setattr(np.random, "randn", stub_random_randn_from_data(dicts[4]))
    monkeypatch.setattr(scipy.stats.unitary_group, "rvs", stub_random_unitary_from_data(dicts[5]))


    # Preliminary constructions:
    # --------------------------

    a0_fourier1_filtered1 = fourier_filtered_frames[0]

    a0_fourier1_filtered2 = fourier_filtered_frames[1]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............


    csMM_fourier11 = MeasurementMatrices(a0_fourier1_filtered1, number_samples)

    csMM_fourier12 = MeasurementMatrices(a0_fourier1_filtered2, number_samples)


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


    assert np.array_equal(ar_fourier11_gauss, data["ar_fourier11_gauss"])

    assert np.array_equal(ar_fourier11_bernoulli, data["ar_fourier11_bernoulli"])

    assert np.array_equal(ar_fourier11_partial_fourier, data["ar_fourier11_partial_fourier"])

    assert np.array_equal(ar_fourier11_partial_dct, data["ar_fourier11_partial_dct"])

    assert np.array_equal(ar_fourier11_toeplitz, data["ar_fourier11_toeplitz"])

    assert np.array_equal(ar_fourier11_binary_block, data["ar_fourier11_binary_block"])

    assert np.array_equal(ar_fourier11_sgn, data["ar_fourier11_sgn"])

    assert np.array_equal(ar_fourier11_gdo, data["ar_fourier11_gdo"])

    assert np.array_equal(ar_fourier12_gdo, data["ar_fourier12_gdo"])

    assert np.array_equal(ar_fourier11_gdo_adaptive, data["ar_fourier11_gdo_adaptive"])

    assert np.array_equal(ar_fourier12_gdo_adaptive, data["ar_fourier12_gdo_adaptive"])

    assert np.array_equal(ar_fourier11_ajs, data["ar_fourier11_ajs"])

    assert np.array_equal(ar_fourier11_afms, data["ar_fourier11_afms"])

    assert np.array_equal(ar_fourier12_afms, data["ar_fourier12_afms"])

    assert np.array_equal(ar_fourier11_hblz, data["ar_fourier11_hblz"])

    assert np.array_equal(ar_fourier12_hblz, data["ar_fourier12_hblz"])

    assert np.array_equal(ar_fourier11_ycwg, data["ar_fourier11_ycwg"])

    assert np.array_equal(ar_fourier12_ycwg, data["ar_fourier12_ycwg"])

    assert np.array_equal(ar_fourier11_xsfz, data["ar_fourier11_xsfz"])

    assert np.array_equal(ar_fourier12_xsfz, data["ar_fourier12_xsfz"])




if __name__ == "__main__":
    pytest.main([__file__])