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
from cssr import Superresolvers


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
def dir_filters():
    return "data/data_filters/"


@pytest.fixture
def dir_superresolvers():
    return "data/data_superresolvers/"


@pytest.fixture
def frame_name(request):
    return request.param


@pytest.fixture
def load_signal_data(dir_filters, frame_name):
    return np.load(dir_filters + frame_name + "_signal.npz")


@pytest.fixture
def load_superresolver_data(dir_superresolvers, frame_name):
    return np.load(dir_superresolvers + frame_name + ".npz")


@pytest.fixture
def construct_test_signal_x_components():
    return np.arange(0, 20, 0.1)


@pytest.fixture
def superresolvers_configurations():
    number_samples = 25
    max_iter = 8
    noise_level = 1e-8
    return [number_samples, max_iter, noise_level]


@pytest.fixture
def construct_cutoffs():
    cutoffs = [[1, None], [0.6, 1.2], [70, None]] # where 70 is 70° celsius
    return cutoffs



@pytest.fixture
def construct_sensing_matrix_components_gauss_gaussian(superresolvers_configurations,
                                                       construct_test_signal_x_components,
                                                       construct_cutoffs,
                                                       mock_random_gaussian,
                                                       stub_random_random_from_data,
                                                       stub_random_binomial_from_data,
                                                       stub_random_sample_from_data,
                                                       stub_random_choice_from_data,
                                                       stub_random_randn_from_data,
                                                       stub_random_unitary_from_data,
                                                       monkeypatch):

    # Preliminaries:
    # --------------
    number_samples, max_iter, noise_level = superresolvers_configurations
    x = construct_test_signal_x_components
    cutoffs = construct_cutoffs

    bw_gaussian = abs(x[-1] - x[0])/20
    bw_gaussian_overcomplete = [abs(x[-1] - x[0])/20 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/20 + 3*abs(x[1] - x[0])]
    ss_gaussian_overcomplete = None

    # Construct frames:
    # -----------------

    csF = Frames(x)

    a0_gaussian2 = csF.gaussian(bw_gaussian)
    a0_gaussian_overcomplete2 = csF.gaussian_overcomplete(bw_gaussian_overcomplete, ss_gaussian_overcomplete)

    # Apply filters:
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



    # Overcomplete dictonaries:
    # .........................

    csFr_a02_gaussian_overcomplete = Filters(a0_gaussian_overcomplete2, x, cutoffs[0][0])
    csFr_a02_gaussian_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.fir_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[1]
    csFr_a02_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_gaussian_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_gaussian_overcomplete2_filtered1 = csFr_a02_gaussian_overcomplete.truncated_a0


    # Construct measurement matrices:
    # -------------------------------

    # Stubbing:
    # .........

    dicts = mock_random_gaussian

    monkeypatch.setattr(np.random, "random", stub_random_random_from_data(dicts[0]))
    monkeypatch.setattr(np.random, "binomial", stub_random_binomial_from_data(dicts[1]))
    monkeypatch.setattr(random, "sample", stub_random_sample_from_data(dicts[2]))
    monkeypatch.setattr(np.random, "choice", stub_random_choice_from_data(dicts[3]))
    monkeypatch.setattr(np.random, "randn", stub_random_randn_from_data(dicts[4]))
    monkeypatch.setattr(scipy.stats.unitary_group, "rvs", stub_random_unitary_from_data(dicts[5]))

    # Dictonaries:
    # ............

    csMM_gaussian21 = MeasurementMatrices(a0_gaussian2_filtered1, number_samples)

    ar_gaussian21_gauss = csMM_gaussian21.random_gauss_matrix()


    # Overcomplete dictonaries:
    # .........................

    csMM_gaussian_overcomplete21 = MeasurementMatrices(a0_gaussian_overcomplete2_filtered1, number_samples)

    ar_gaussian_overcomplete21_gauss = csMM_gaussian_overcomplete21.random_gauss_matrix()

    components = [
        [ a0_gaussian2,
          a0_gaussian2_filtered1,
          ar_gaussian21_gauss ],
        [ a0_gaussian_overcomplete2,
          a0_gaussian_overcomplete2_filtered1,
          ar_gaussian_overcomplete21_gauss ]
        ]

    return components



@pytest.mark.parametrize("frame_name", ["gaussian"], indirect=True)
def test_superresolvers_guassian_based_sensing_matrices(load_superresolver_data,
                                                construct_sensing_matrix_components_gauss_gaussian,
                                                load_signal_data, superresolvers_configurations):
    data = load_superresolver_data

    # Preliminary constructions:
    # --------------------------

    _, max_iter, noise_level = superresolvers_configurations
    components_dictionary, components_overcomplete_dictionary = construct_sensing_matrix_components_gauss_gaussian

    data_y = load_signal_data
    y = data_y["y_gaussian2_filtered1"]


    # Main checks:
    # ------------

    # Dictonaries:
    # ............

    csS_frame21_gaussian_gauss = Superresolvers(components_dictionary[0],
                                                components_dictionary[1],
                                                components_dictionary[2])

    y_gaussian21_gauss_sr_bp = csS_frame21_gaussian_gauss.bp(y)
    y_gaussian21_gauss_sr_bpd = csS_frame21_gaussian_gauss.bpd(y, noise_level)
    y_gaussian21_gauss_sr_ic = csS_frame21_gaussian_gauss.ic(y, max_iter=max_iter)
    y_gaussian21_gauss_sr_nlht = csS_frame21_gaussian_gauss.nlht(y, noise_level)
    y_gaussian21_gauss_sr_nlht_lasso = csS_frame21_gaussian_gauss.nlht_lasso(y, max_iter=max_iter)


    # Overcomplete dicionaries:
    # .........................

    csS_frame21_gaussian_overcomplete_gauss = Superresolvers(components_overcomplete_dictionary[0],
                                                              components_overcomplete_dictionary[1],
                                                              components_overcomplete_dictionary[2])

    y_gaussian_overcomplete21_gauss_sr_bp = csS_frame21_gaussian_overcomplete_gauss.bp(y)
    y_gaussian_overcomplete21_gauss_sr_bpd = csS_frame21_gaussian_overcomplete_gauss.bpd(y, noise_level)
    y_gaussian_overcomplete21_gauss_sr_ic = csS_frame21_gaussian_overcomplete_gauss.ic(y, max_iter=max_iter)
    y_gaussian_overcomplete21_gauss_sr_nlht = csS_frame21_gaussian_overcomplete_gauss.nlht(y, noise_level)
    y_gaussian_overcomplete21_gauss_sr_nlht_lasso = csS_frame21_gaussian_overcomplete_gauss.nlht_lasso(y, max_iter=max_iter)

    assert np.allclose(y_gaussian21_gauss_sr_bp, data["y_gaussian21_gauss_sr_bp"])
    assert np.allclose(y_gaussian21_gauss_sr_bpd, data["y_gaussian21_gauss_sr_bpd"])
    assert np.allclose(y_gaussian21_gauss_sr_ic, data["y_gaussian21_gauss_sr_ic"])
    assert np.allclose(y_gaussian21_gauss_sr_nlht, data["y_gaussian21_gauss_sr_nlht"])
    assert np.allclose(y_gaussian21_gauss_sr_nlht_lasso, data["y_gaussian21_gauss_sr_nlht_lasso"])

    assert np.allclose(y_gaussian_overcomplete21_gauss_sr_bp, data["y_gaussian_overcomplete21_gauss_sr_bp"])
    assert np.allclose(y_gaussian_overcomplete21_gauss_sr_bpd, data["y_gaussian_overcomplete21_gauss_sr_bpd"])
    assert np.allclose(y_gaussian_overcomplete21_gauss_sr_ic, data["y_gaussian_overcomplete21_gauss_sr_ic"])
    assert np.allclose(y_gaussian_overcomplete21_gauss_sr_nlht, data["y_gaussian_overcomplete21_gauss_sr_nlht"])
    assert np.allclose(y_gaussian_overcomplete21_gauss_sr_nlht_lasso, data["y_gaussian_overcomplete21_gauss_sr_nlht_lasso"])


# =============================================================================
