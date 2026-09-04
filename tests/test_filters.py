"""
Test module for the filters module of the cssr package.

@author: marcelo looser
"""


import pytest
import numpy as np
from cssr import Frames
from cssr import Filters

# =============================================================================


@pytest.fixture
def dir_filters():
    return "data/data_filters/"


@pytest.fixture
def frame_name(request):
    return request.param


@pytest.fixture
def load_filter_data(dir_filters, frame_name):
    return np.load(dir_filters + frame_name + ".npz")


@pytest.fixture
def construct_test_signal_x_components():
    return np.arange(0, 20, 0.1)


@pytest.fixture
def construct_test_signal(construct_test_signal_x_components):
    x = construct_test_signal_x_components
    dim = len(x)

    n_peaks = 16 # np.mod(8*dim//10, n_peaks) == 0
    amps = np.arange(0.01, 1, 1/n_peaks).reshape((-1,1))
    peaks = np.arange(dim//10, 9*dim//10, 8*dim//10//n_peaks)

    y_sparse = np.zeros((dim, 1))
    y_sparse[peaks] = amps
    return [x, dim, n_peaks, amps, peaks, y_sparse]


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


@pytest.mark.parametrize("frame_name", ["gaussian_signal"], indirect=True)
def test_filtered_gaussian_signal(load_filter_data, construct_test_signal, construct_gaussian_frame, construct_cutoffs):
    data = load_filter_data
    x, dim, n_peaks, amps, peaks, y_sparse = construct_test_signal
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    gaussian_frames = construct_gaussian_frame
    a0_gaussian2 = gaussian_frames[0]


    # Main checks:
    # ------------

    y_gaussian2 = a0_gaussian2.dot(y_sparse)

    csFr_y2_gaussian = Filters(y_gaussian2, x, cutoffs[0][0], filter_signal=True)

    csFr_y2_gaussian.heaviside_lowpass_filter(return_array=False)
    csFr_y2_gaussian.fir_filter(return_array=False)
    csFr_y2_gaussian.cutoff = cutoffs[1]
    csFr_y2_gaussian.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_y2_gaussian.butter_filter(order=5, btype="bandstop", return_array=False)

    y_gaussian2_filtered1 = csFr_y2_gaussian.truncated_a0
    y_gaussian_filter2_record1 = csFr_y2_gaussian.filter_record(name_only=True)

    csFr_y2_gaussian.reset()

    csFr_y2_gaussian.butter_filter(return_array=False)
    csFr_y2_gaussian.instrumental_lowpass_filter(return_array=False)
    csFr_y2_gaussian.cutoff = cutoffs[2][0]
    csFr_y2_gaussian.thermal_lowpass_filter(return_array=False)

    y_gaussian2_filtered2 = csFr_y2_gaussian.truncated_a0
    y_gaussian_filter2_record2 = csFr_y2_gaussian.filter_record(name_only=True)


    assert np.array_equal(y_gaussian2, data["y_gaussian2"])
    assert np.array_equal(y_gaussian2_filtered1, data["y_gaussian2_filtered1"])
    assert np.array_equal(y_gaussian_filter2_record1, data["y_gaussian_filter2_record1"])
    assert np.array_equal(y_gaussian2_filtered2, data["y_gaussian2_filtered2"])
    assert np.array_equal(y_gaussian_filter2_record2, data["y_gaussian_filter2_record2"])


@pytest.mark.parametrize("frame_name", ["gaussian_frame"], indirect=True)
def test_filtered_gaussian_frame(load_filter_data, construct_test_signal_x_components, construct_gaussian_frame, construct_cutoffs):
    data = load_filter_data
    x = construct_test_signal_x_components
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------
    gaussian_frames = construct_gaussian_frame

    a0_gaussian2 = gaussian_frames[0]
    a0_gaussian_overcomplete2 = gaussian_frames[1]

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
    a0_gaussian_filter2_record1 = csFr_a02_gaussian.filter_record(name_only=True)

    csFr_a02_gaussian.reset()

    csFr_a02_gaussian.butter_filter(return_array=False)
    csFr_a02_gaussian.instrumental_lowpass_filter(return_array=False)
    csFr_a02_gaussian.cutoff = cutoffs[2][0]
    csFr_a02_gaussian.thermal_lowpass_filter(return_array=False)

    a0_gaussian2_filtered2 = csFr_a02_gaussian.truncated_a0
    a0_gaussian_filter2_record2 = csFr_a02_gaussian.filter_record(name_only=True)


    assert np.array_equal(a0_gaussian2_filtered1, data["a0_gaussian2_filtered1"])
    assert np.array_equal(a0_gaussian_filter2_record1, data["a0_gaussian_filter2_record1"])
    assert np.array_equal(a0_gaussian2_filtered2, data["a0_gaussian2_filtered2"])
    assert np.array_equal(a0_gaussian_filter2_record2, data["a0_gaussian_filter2_record2"])


    # Overcomplete dictonaries:
    # .........................

    csFr_a02_gaussian_overcomplete = Filters(a0_gaussian_overcomplete2, x, cutoffs[0][0])

    csFr_a02_gaussian_overcomplete.heaviside_lowpass_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.fir_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[1]
    csFr_a02_gaussian_overcomplete.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a02_gaussian_overcomplete.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_gaussian_overcomplete2_filtered1 = csFr_a02_gaussian_overcomplete.truncated_a0
    a0_gaussian_overcomplete_filter2_record1 = csFr_a02_gaussian_overcomplete.filter_record(name_only=True)

    csFr_a02_gaussian_overcomplete.reset()

    csFr_a02_gaussian_overcomplete.butter_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.instrumental_lowpass_filter(return_array=False)
    csFr_a02_gaussian_overcomplete.cutoff = cutoffs[2][0]
    csFr_a02_gaussian_overcomplete.thermal_lowpass_filter(return_array=False)

    a0_gaussian_overcomplete2_filtered2 = csFr_a02_gaussian_overcomplete.truncated_a0
    a0_gaussian_overcomplete_filter2_record2 = csFr_a02_gaussian_overcomplete.filter_record(name_only=True)


    assert np.array_equal(a0_gaussian_overcomplete2_filtered1, data["a0_gaussian_overcomplete2_filtered1"])
    assert np.array_equal(a0_gaussian_overcomplete_filter2_record1, data["a0_gaussian_overcomplete_filter2_record1"])
    assert np.array_equal(a0_gaussian_overcomplete2_filtered2, data["a0_gaussian_overcomplete2_filtered2"])
    assert np.array_equal(a0_gaussian_overcomplete_filter2_record2, data["a0_gaussian_overcomplete_filter2_record2"])

# =============================================================================
