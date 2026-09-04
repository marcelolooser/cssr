"""
Test module for the measurement_matrices module of the cssr package.

@author: marcelo looser
"""

import scipy
import pytest
import random
import numpy as np

from cssr import Frames
from cssr import Filters
from cssr import MeasurementMatrices

# =============================================================================
# Stubbing:
# =============================================================================

@pytest.fixture
def dir_stubs():
    return "tests/data/data_stubs/"


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
def mock_random_gaussian(dir_stubs):
    data_stub_gaussian = np.load(dir_stubs + "gaussian" + ".npz")
    shapes_gaussian = data_stub_gaussian["tracked_shapes_gaussian"]
    shapes_gaussian = [tuple(item.tolist()) for item in shapes_gaussian]

    np_random_random = {
     shapes_gaussian[0] : data_stub_gaussian["uniform_ns_m_gaussian2"],
     shapes_gaussian[1] : data_stub_gaussian["uniform_ns_n_gaussian2"],

     shapes_gaussian[2] : data_stub_gaussian["uniform_ns_m_gaussian_overcomplete2"],
     shapes_gaussian[3] : data_stub_gaussian["uniform_ns_n_gaussian_overcomplete2"]
     }

    probability = float(data_stub_gaussian["probability"])
    np_random_binomial = {
         (1, probability, shapes_gaussian[4]) : data_stub_gaussian["bernoulli_ns_m_gaussian2"],

         (1, probability, shapes_gaussian[5]) : data_stub_gaussian["bernoulli_ns_m_gaussian_overcomplete2"]
         }


    random_sample = {
         (range(shapes_gaussian[6][0]), shapes_gaussian[6][1]) : data_stub_gaussian["samples_m_ns_gaussian2"],
         (range(shapes_gaussian[7][0]), shapes_gaussian[7][1]) : data_stub_gaussian["samples_n_m_gaussian2"],

         (range(shapes_gaussian[8][0]), shapes_gaussian[8][1]) : data_stub_gaussian["samples_m_ns_gaussian_overcomplete2"],
         (range(shapes_gaussian[9][0]), shapes_gaussian[9][1]) : data_stub_gaussian["samples_n_m_gaussian_overcomplete2"]
         }


    choices = (-1,1)
    np_random_choice = {
         (choices, shapes_gaussian[10][0]) : data_stub_gaussian["samples_binary_m_gaussian2"],

         (choices, shapes_gaussian[11][0]) : data_stub_gaussian["samples_binary_m_gaussian_overcomplete2"]
         }


    np_random_randn = {
         shapes_gaussian[12] : data_stub_gaussian["gaussian_ns_m_gaussian2"],

         shapes_gaussian[13] : data_stub_gaussian["gaussian_ns_m_gaussian_overcomplete2"]
         }

    scipy_stats_unitary_group_rvs = {
         shapes_gaussian[14][0] : data_stub_gaussian["random_unitary_ns_m_gaussian"]
         }

    return [np_random_random, np_random_binomial, random_sample, np_random_choice, np_random_randn, scipy_stats_unitary_group_rvs]


# =============================================================================
# General setup:
# =============================================================================

@pytest.fixture
def dir_measurement_matrices():
    return "tests/data/data_measurement_matrices/"


@pytest.fixture
def frame_name(request):
    return request.param


@pytest.fixture
def load_measurement_matrix_data(dir_measurement_matrices, frame_name):
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
def construct_gaussian_frame(construct_test_signal_x_components):

    # Preliminaries:
    # --------------
    x = construct_test_signal_x_components

    bw_gaussian = abs(x[-1] - x[0])/20
    bw_gaussian_overcomplete = [abs(x[-1] - x[0])/20 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/20 + 3*abs(x[1] - x[0])]
    ss_gaussian_overcomplete = None

    # Construct frames:
    # -----------------

    csF = Frames(x)
    gaussian_frames = [
        csF.gaussian(bw_gaussian),
        csF.gaussian_overcomplete(bw_gaussian_overcomplete, ss_gaussian_overcomplete)
        ]

    return gaussian_frames


@pytest.fixture
def construct_filtered_gaussian_frame(construct_test_signal_x_components, construct_gaussian_frame, construct_cutoffs):
    x = construct_test_signal_x_components
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    gaussian_frames = construct_gaussian_frame

    a0_gaussian2 = gaussian_frames[0]
    a0_gaussian_overcomplete2= gaussian_frames[1]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............

    csFr_a02_gaussian = Filters(a0_gaussian2, x, cutoffs[0][0])

    csFr_a02_gaussian.heaviside_lowpass_filter(return_array=False)
    csFr_a02_gaussian.fir_filter(return_array=False)
    csFr_a02_gaussian.cutoff = cutoffs[1]
    csFr_a02_gaussian.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_gaussian.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_gaussian2_filtered1 = csFr_a02_gaussian.truncated_a0

    csFr_a02_gaussian.reset()

    csFr_a02_gaussian.butter_filter(return_array=False)
    csFr_a02_gaussian.instrumental_lowpass_filter(return_array=False)
    csFr_a02_gaussian.cutoff = cutoffs[2][0]
    csFr_a02_gaussian.thermal_lowpass_filter(return_array=False)

    a0_gaussian2_filtered2 = csFr_a02_gaussian.truncated_a0


    # Overcomplete dictonaries:
    # .........................

    csFr_a02_gaussian_overcomplete = Filters(a0_gaussian_overcomplete2, x, cutoffs[0][0])

    csFr_a02_gaussian_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.fir_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[1]
    csFr_a02_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_gaussian_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_gaussian_overcomplete2_filtered1 = csFr_a02_gaussian_overcomplete.truncated_a0

    csFr_a02_gaussian_overcomplete.reset()
    csFr_a02_gaussian_overcomplete.butter_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.instrumental_lowpass_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[2][0]
    csFr_a02_gaussian_overcomplete.thermal_lowpass_filter(return_array=False)

    a0_gaussian_overcomplete2_filtered2 = csFr_a02_gaussian_overcomplete.truncated_a0


    gaussian_filtered_frames = [
        a0_gaussian2_filtered1,
        a0_gaussian2_filtered2,

        a0_gaussian_overcomplete2_filtered1,
        a0_gaussian_overcomplete2_filtered2
        ]

    return gaussian_filtered_frames



@pytest.mark.parametrize("frame_name", ["gaussian"], indirect=True)
def test_gaussian_based_measurement_matrices(load_measurement_matrix_data,
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
    data = load_measurement_matrix_data
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

    a0_gaussian2_filtered1 = gaussian_filtered_frames[0]
    a0_gaussian2_filtered2 = gaussian_filtered_frames[1]

    a0_gaussian_overcomplete2_filtered1 = gaussian_filtered_frames[2]
    a0_gaussian_overcomplete2_filtered2 = gaussian_filtered_frames[3]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............


    csMM_gaussian21 = MeasurementMatrices(a0_gaussian2_filtered1, number_samples)

    csMM_gaussian22 = MeasurementMatrices(a0_gaussian2_filtered2, number_samples)


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
    ar_gaussian21_ajs = csMM_gaussian21.ajs(max_iter=max_iter)
    ar_gaussian21_afms = csMM_gaussian21.afms(max_iter=max_iter)
    ar_gaussian22_afms = csMM_gaussian22.afms(max_iter=max_iter)
    ar_gaussian21_hblz = csMM_gaussian21.hblz(l=l, p=p)
    ar_gaussian22_hblz = csMM_gaussian22.hblz(l=l, p=p)
    ar_gaussian21_ycwg = csMM_gaussian21.ycwg(max_iter=max_iter)
    ar_gaussian22_ycwg = csMM_gaussian22.ycwg(max_iter=max_iter)
    ar_gaussian21_xsfz = csMM_gaussian21.xsfz(max_iter=max_iter)
    ar_gaussian22_xsfz = csMM_gaussian22.xsfz(max_iter=max_iter)



    # Overcomplete dicionaries:
    # .........................

    csMM_gaussian_overcomplete21 = MeasurementMatrices(a0_gaussian_overcomplete2_filtered1, number_samples)

    csMM_gaussian_overcomplete22 = MeasurementMatrices(a0_gaussian_overcomplete2_filtered2, number_samples)

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
    ar_gaussian_overcomplete21_ajs = csMM_gaussian_overcomplete21.ajs(max_iter=max_iter)
    ar_gaussian_overcomplete21_afms = csMM_gaussian_overcomplete21.afms(max_iter=max_iter)
    ar_gaussian_overcomplete22_afms = csMM_gaussian_overcomplete22.afms(max_iter=max_iter)
    ar_gaussian_overcomplete21_hblz = csMM_gaussian_overcomplete21.hblz(l=l, p=p)
    ar_gaussian_overcomplete22_hblz = csMM_gaussian_overcomplete22.hblz(l=l, p=p)
    ar_gaussian_overcomplete21_ycwg = csMM_gaussian_overcomplete21.ycwg(max_iter=max_iter)
    ar_gaussian_overcomplete22_ycwg = csMM_gaussian_overcomplete22.ycwg(max_iter=max_iter)
    ar_gaussian_overcomplete21_xsfz = csMM_gaussian_overcomplete21.xsfz(max_iter=max_iter)
    ar_gaussian_overcomplete22_xsfz = csMM_gaussian_overcomplete22.xsfz(max_iter=max_iter)


    assert np.allclose(ar_gaussian21_gauss, data["ar_gaussian21_gauss"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_bernoulli, data["ar_gaussian21_bernoulli"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_partial_fourier, data["ar_gaussian21_partial_fourier"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_partial_dct, data["ar_gaussian21_partial_dct"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_toeplitz, data["ar_gaussian21_toeplitz"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_binary_block, data["ar_gaussian21_binary_block"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_sgn, data["ar_gaussian21_sgn"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_gdo, data["ar_gaussian21_gdo"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian22_gdo, data["ar_gaussian22_gdo"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_gdo_adaptive, data["ar_gaussian21_gdo_adaptive"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian22_gdo_adaptive, data["ar_gaussian22_gdo_adaptive"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_ajs, data["ar_gaussian21_ajs"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_afms, data["ar_gaussian21_afms"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian22_afms, data["ar_gaussian22_afms"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_hblz, data["ar_gaussian21_hblz"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian22_hblz, data["ar_gaussian22_hblz"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_ycwg, data["ar_gaussian21_ycwg"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian22_ycwg, data["ar_gaussian22_ycwg"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian21_xsfz, data["ar_gaussian21_xsfz"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian22_xsfz, data["ar_gaussian22_xsfz"], rtol=1e-9, atol=1e-9)


    assert np.allclose(ar_gaussian_overcomplete21_gauss, data["ar_gaussian_overcomplete21_gauss"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_bernoulli, data["ar_gaussian_overcomplete21_bernoulli"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_partial_fourier, data["ar_gaussian_overcomplete21_partial_fourier"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_partial_dct, data["ar_gaussian_overcomplete21_partial_dct"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_toeplitz, data["ar_gaussian_overcomplete21_toeplitz"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_binary_block, data["ar_gaussian_overcomplete21_binary_block"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_sgn, data["ar_gaussian_overcomplete21_sgn"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_gdo, data["ar_gaussian_overcomplete21_gdo"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete22_gdo, data["ar_gaussian_overcomplete22_gdo"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_gdo_adaptive, data["ar_gaussian_overcomplete21_gdo_adaptive"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete22_gdo_adaptive, data["ar_gaussian_overcomplete22_gdo_adaptive"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_ajs, data["ar_gaussian_overcomplete21_ajs"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_afms, data["ar_gaussian_overcomplete21_afms"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete22_afms, data["ar_gaussian_overcomplete22_afms"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_hblz, data["ar_gaussian_overcomplete21_hblz"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete22_hblz, data["ar_gaussian_overcomplete22_hblz"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_ycwg, data["ar_gaussian_overcomplete21_ycwg"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete22_ycwg, data["ar_gaussian_overcomplete22_ycwg"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete21_xsfz, data["ar_gaussian_overcomplete21_xsfz"], rtol=1e-9, atol=1e-9)
    assert np.allclose(ar_gaussian_overcomplete22_xsfz, data["ar_gaussian_overcomplete22_xsfz"], rtol=1e-9, atol=1e-9)


# =============================================================================
