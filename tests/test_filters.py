"""
Test module for the filters module of the cssr package.

@author: marcelo looser
"""


import pytest
import numpy as np

# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


@pytest.mark.parametrize("frame_name", ["heaviside_signal"], indirect=True)
def test_filtered_heaviside_signal(load_filter_data, construct_test_signal, construct_heaviside_frame, construct_cutoffs):
    data = load_filter_data
    x, dim, n_peaks, amps, peaks, y_sparse = construct_test_signal
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    heaviside_frames = construct_heaviside_frame
    a0_heaviside1, a0_heaviside2, a0_heaviside3 = heaviside_frames[:3]


    # Main checks:
    # ------------

    y_heaviside1 = a0_heaviside1.dot(y_sparse)
    y_heaviside2 = a0_heaviside2.dot(y_sparse)
    y_heaviside3 = a0_heaviside3.dot(y_sparse)

    csFr_y1_heaviside = Filters(y_heaviside1, x, cutoffs[0][0], filter_signal=True)
    csFr_y2_heaviside = Filters(y_heaviside2, x, cutoffs[0][0], filter_signal=True)
    csFr_y3_heaviside = Filters(y_heaviside3, x, cutoffs[0][0], filter_signal=True)

    csFr_y1_heaviside.heaviside_lowpass_filter(return_array=False)
    csFr_y2_heaviside.heaviside_lowpass_filter(return_array=False)
    csFr_y3_heaviside.heaviside_lowpass_filter(return_array=False)

    csFr_y1_heaviside.fir_filter(return_array=False)
    csFr_y2_heaviside.fir_filter(return_array=False)
    csFr_y3_heaviside.fir_filter(return_array=False)

    csFr_y1_heaviside.cutoff = cutoffs[1]
    csFr_y2_heaviside.cutoff = cutoffs[1]
    csFr_y3_heaviside.cutoff = cutoffs[1]

    csFr_y1_heaviside.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_y2_heaviside.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_y3_heaviside.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_y1_heaviside.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_y2_heaviside.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_y3_heaviside.butter_filter(order=5, btype="bandstop", return_array=False)

    y_heaviside1_filtered1 = csFr_y1_heaviside.truncated_a0
    y_heaviside2_filtered1 = csFr_y2_heaviside.truncated_a0
    y_heaviside3_filtered1 = csFr_y3_heaviside.truncated_a0

    y_heaviside_filter1_record1 = csFr_y1_heaviside.filter_record(name_only=True)
    y_heaviside_filter2_record1 = csFr_y2_heaviside.filter_record(name_only=True)
    y_heaviside_filter3_record1 = csFr_y3_heaviside.filter_record(name_only=True)

    csFr_y1_heaviside.reset()
    csFr_y2_heaviside.reset()
    csFr_y3_heaviside.reset()

    csFr_y1_heaviside.butter_filter(return_array=False)
    csFr_y2_heaviside.butter_filter(return_array=False)
    csFr_y3_heaviside.butter_filter(return_array=False)

    csFr_y1_heaviside.instrumental_lowpass_filter(return_array=False)
    csFr_y2_heaviside.instrumental_lowpass_filter(return_array=False)
    csFr_y3_heaviside.instrumental_lowpass_filter(return_array=False)

    csFr_y1_heaviside.cutoff = cutoffs[2][0]
    csFr_y2_heaviside.cutoff = cutoffs[2][0]
    csFr_y3_heaviside.cutoff = cutoffs[2][0]

    csFr_y1_heaviside.thermal_lowpass_filter(return_array=False)
    csFr_y2_heaviside.thermal_lowpass_filter(return_array=False)
    csFr_y3_heaviside.thermal_lowpass_filter(return_array=False)

    y_heaviside1_filtered2 = csFr_y1_heaviside.truncated_a0
    y_heaviside2_filtered2 = csFr_y2_heaviside.truncated_a0
    y_heaviside3_filtered2 = csFr_y3_heaviside.truncated_a0

    y_heaviside_filter1_record2 = csFr_y1_heaviside.filter_record(name_only=True)
    y_heaviside_filter2_record2 = csFr_y2_heaviside.filter_record(name_only=True)
    y_heaviside_filter3_record2 = csFr_y3_heaviside.filter_record(name_only=True)


    assert np.array_equal(y_heaviside1, data["y_heaviside1"])
    assert np.array_equal(y_heaviside2, data["y_heaviside2"])
    assert np.array_equal(y_heaviside3, data["y_heaviside3"])

    assert np.array_equal(y_heaviside1_filtered1, data["y_heaviside1_filtered1"])
    assert np.array_equal(y_heaviside2_filtered1, data["y_heaviside2_filtered1"])
    assert np.array_equal(y_heaviside3_filtered1, data["y_heaviside3_filtered1"])

    assert np.array_equal(y_heaviside_filter1_record1, data["y_heaviside_filter1_record1"])
    assert np.array_equal(y_heaviside_filter2_record1, data["y_heaviside_filter2_record1"])
    assert np.array_equal(y_heaviside_filter3_record1, data["y_heaviside_filter3_record1"])

    assert np.array_equal(y_heaviside1_filtered2, data["y_heaviside1_filtered2"])
    assert np.array_equal(y_heaviside2_filtered2, data["y_heaviside2_filtered2"])
    assert np.array_equal(y_heaviside3_filtered2, data["y_heaviside3_filtered2"])

    assert np.array_equal(y_heaviside_filter1_record2, data["y_heaviside_filter1_record2"])
    assert np.array_equal(y_heaviside_filter2_record2, data["y_heaviside_filter2_record2"])
    assert np.array_equal(y_heaviside_filter3_record2, data["y_heaviside_filter3_record2"])


@pytest.mark.parametrize("frame_name", ["gaussian_signal"], indirect=True)
def test_filtered_gaussian_signal(load_filter_data, construct_test_signal, construct_gaussian_frame, construct_cutoffs):
    data = load_filter_data
    x, dim, n_peaks, amps, peaks, y_sparse = construct_test_signal
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    gaussian_frames = construct_gaussian_frame
    a0_gaussian1, a0_gaussian2, a0_gaussian3 = gaussian_frames[:3]


    # Main checks:
    # ------------

    y_gaussian1 = a0_gaussian1.dot(y_sparse)
    y_gaussian2 = a0_gaussian2.dot(y_sparse)
    y_gaussian3 = a0_gaussian3.dot(y_sparse)

    csFr_y1_gaussian = Filters(y_gaussian1, x, cutoffs[0][0], filter_signal=True)
    csFr_y2_gaussian = Filters(y_gaussian2, x, cutoffs[0][0], filter_signal=True)
    csFr_y3_gaussian = Filters(y_gaussian3, x, cutoffs[0][0], filter_signal=True)

    csFr_y1_gaussian.heaviside_lowpass_filter(return_array=False)
    csFr_y2_gaussian.heaviside_lowpass_filter(return_array=False)
    csFr_y3_gaussian.heaviside_lowpass_filter(return_array=False)

    csFr_y1_gaussian.fir_filter(return_array=False)
    csFr_y2_gaussian.fir_filter(return_array=False)
    csFr_y3_gaussian.fir_filter(return_array=False)

    csFr_y1_gaussian.cutoff = cutoffs[1]
    csFr_y2_gaussian.cutoff = cutoffs[1]
    csFr_y3_gaussian.cutoff = cutoffs[1]

    csFr_y1_gaussian.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_y2_gaussian.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_y3_gaussian.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_y1_gaussian.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_y2_gaussian.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_y3_gaussian.butter_filter(order=5, btype="bandstop", return_array=False)

    y_gaussian1_filtered1 = csFr_y1_gaussian.truncated_a0
    y_gaussian2_filtered1 = csFr_y2_gaussian.truncated_a0
    y_gaussian3_filtered1 = csFr_y3_gaussian.truncated_a0

    y_gaussian_filter1_record1 = csFr_y1_gaussian.filter_record(name_only=True)
    y_gaussian_filter2_record1 = csFr_y2_gaussian.filter_record(name_only=True)
    y_gaussian_filter3_record1 = csFr_y3_gaussian.filter_record(name_only=True)

    csFr_y1_gaussian.reset()
    csFr_y2_gaussian.reset()
    csFr_y3_gaussian.reset()

    csFr_y1_gaussian.butter_filter(return_array=False)
    csFr_y2_gaussian.butter_filter(return_array=False)
    csFr_y3_gaussian.butter_filter(return_array=False)

    csFr_y1_gaussian.instrumental_lowpass_filter(return_array=False)
    csFr_y2_gaussian.instrumental_lowpass_filter(return_array=False)
    csFr_y3_gaussian.instrumental_lowpass_filter(return_array=False)

    csFr_y1_gaussian.cutoff = cutoffs[2][0]
    csFr_y2_gaussian.cutoff = cutoffs[2][0]
    csFr_y3_gaussian.cutoff = cutoffs[2][0]

    csFr_y1_gaussian.thermal_lowpass_filter(return_array=False)
    csFr_y2_gaussian.thermal_lowpass_filter(return_array=False)
    csFr_y3_gaussian.thermal_lowpass_filter(return_array=False)

    y_gaussian1_filtered2 = csFr_y1_gaussian.truncated_a0
    y_gaussian2_filtered2 = csFr_y2_gaussian.truncated_a0
    y_gaussian3_filtered2 = csFr_y3_gaussian.truncated_a0

    y_gaussian_filter1_record2 = csFr_y1_gaussian.filter_record(name_only=True)
    y_gaussian_filter2_record2 = csFr_y2_gaussian.filter_record(name_only=True)
    y_gaussian_filter3_record2 = csFr_y3_gaussian.filter_record(name_only=True)


    assert np.array_equal(y_gaussian1, data["y_gaussian1"])
    assert np.array_equal(y_gaussian2, data["y_gaussian2"])
    assert np.array_equal(y_gaussian3, data["y_gaussian3"])

    assert np.array_equal(y_gaussian1_filtered1, data["y_gaussian1_filtered1"])
    assert np.array_equal(y_gaussian2_filtered1, data["y_gaussian2_filtered1"])
    assert np.array_equal(y_gaussian3_filtered1, data["y_gaussian3_filtered1"])

    assert np.array_equal(y_gaussian_filter1_record1, data["y_gaussian_filter1_record1"])
    assert np.array_equal(y_gaussian_filter2_record1, data["y_gaussian_filter2_record1"])
    assert np.array_equal(y_gaussian_filter3_record1, data["y_gaussian_filter3_record1"])

    assert np.array_equal(y_gaussian1_filtered2, data["y_gaussian1_filtered2"])
    assert np.array_equal(y_gaussian2_filtered2, data["y_gaussian2_filtered2"])
    assert np.array_equal(y_gaussian3_filtered2, data["y_gaussian3_filtered2"])

    assert np.array_equal(y_gaussian_filter1_record2, data["y_gaussian_filter1_record2"])
    assert np.array_equal(y_gaussian_filter2_record2, data["y_gaussian_filter2_record2"])
    assert np.array_equal(y_gaussian_filter3_record2, data["y_gaussian_filter3_record2"])


@pytest.mark.parametrize("frame_name", ["cauchy_signal"], indirect=True)
def test_filtered_cauchy_signal(load_filter_data, construct_test_signal, construct_cauchy_frame, construct_cutoffs):
    data = load_filter_data
    x, dim, n_peaks, amps, peaks, y_sparse = construct_test_signal
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    cauchy_frames = construct_cauchy_frame
    a0_cauchy1, a0_cauchy2, a0_cauchy3 = cauchy_frames[:3]


    # Main checks:
    # ------------

    y_cauchy1 = a0_cauchy1.dot(y_sparse)
    y_cauchy2 = a0_cauchy2.dot(y_sparse)
    y_cauchy3 = a0_cauchy3.dot(y_sparse)

    csFr_y1_cauchy = Filters(y_cauchy1, x, cutoffs[0][0], filter_signal=True)
    csFr_y2_cauchy = Filters(y_cauchy2, x, cutoffs[0][0], filter_signal=True)
    csFr_y3_cauchy = Filters(y_cauchy3, x, cutoffs[0][0], filter_signal=True)

    csFr_y1_cauchy.heaviside_lowpass_filter(return_array=False)
    csFr_y2_cauchy.heaviside_lowpass_filter(return_array=False)
    csFr_y3_cauchy.heaviside_lowpass_filter(return_array=False)

    csFr_y1_cauchy.fir_filter(return_array=False)
    csFr_y2_cauchy.fir_filter(return_array=False)
    csFr_y3_cauchy.fir_filter(return_array=False)

    csFr_y1_cauchy.cutoff = cutoffs[1]
    csFr_y2_cauchy.cutoff = cutoffs[1]
    csFr_y3_cauchy.cutoff = cutoffs[1]

    csFr_y1_cauchy.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_y2_cauchy.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_y3_cauchy.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)

    csFr_y1_cauchy.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_y2_cauchy.butter_filter(order=5, btype="bandstop", return_array=False)
    csFr_y3_cauchy.butter_filter(order=5, btype="bandstop", return_array=False)

    y_cauchy1_filtered1 = csFr_y1_cauchy.truncated_a0
    y_cauchy2_filtered1 = csFr_y2_cauchy.truncated_a0
    y_cauchy3_filtered1 = csFr_y3_cauchy.truncated_a0

    y_cauchy_filter1_record1 = csFr_y1_cauchy.filter_record(name_only=True)
    y_cauchy_filter2_record1 = csFr_y2_cauchy.filter_record(name_only=True)
    y_cauchy_filter3_record1 = csFr_y3_cauchy.filter_record(name_only=True)

    csFr_y1_cauchy.reset()
    csFr_y2_cauchy.reset()
    csFr_y3_cauchy.reset()

    csFr_y1_cauchy.butter_filter(return_array=False)
    csFr_y2_cauchy.butter_filter(return_array=False)
    csFr_y3_cauchy.butter_filter(return_array=False)

    csFr_y1_cauchy.instrumental_lowpass_filter(return_array=False)
    csFr_y2_cauchy.instrumental_lowpass_filter(return_array=False)
    csFr_y3_cauchy.instrumental_lowpass_filter(return_array=False)

    csFr_y1_cauchy.cutoff = cutoffs[2][0]
    csFr_y2_cauchy.cutoff = cutoffs[2][0]
    csFr_y3_cauchy.cutoff = cutoffs[2][0]

    csFr_y1_cauchy.thermal_lowpass_filter(return_array=False)
    csFr_y2_cauchy.thermal_lowpass_filter(return_array=False)
    csFr_y3_cauchy.thermal_lowpass_filter(return_array=False)

    y_cauchy1_filtered2 = csFr_y1_cauchy.truncated_a0
    y_cauchy2_filtered2 = csFr_y2_cauchy.truncated_a0
    y_cauchy3_filtered2 = csFr_y3_cauchy.truncated_a0

    y_cauchy_filter1_record2 = csFr_y1_cauchy.filter_record(name_only=True)
    y_cauchy_filter2_record2 = csFr_y2_cauchy.filter_record(name_only=True)
    y_cauchy_filter3_record2 = csFr_y3_cauchy.filter_record(name_only=True)


    assert np.array_equal(y_cauchy1, data["y_cauchy1"])
    assert np.array_equal(y_cauchy2, data["y_cauchy2"])
    assert np.array_equal(y_cauchy3, data["y_cauchy3"])

    assert np.array_equal(y_cauchy1_filtered1, data["y_cauchy1_filtered1"])
    assert np.array_equal(y_cauchy2_filtered1, data["y_cauchy2_filtered1"])
    assert np.array_equal(y_cauchy3_filtered1, data["y_cauchy3_filtered1"])

    assert np.array_equal(y_cauchy_filter1_record1, data["y_cauchy_filter1_record1"])
    assert np.array_equal(y_cauchy_filter2_record1, data["y_cauchy_filter2_record1"])
    assert np.array_equal(y_cauchy_filter3_record1, data["y_cauchy_filter3_record1"])

    assert np.array_equal(y_cauchy1_filtered2, data["y_cauchy1_filtered2"])
    assert np.array_equal(y_cauchy2_filtered2, data["y_cauchy2_filtered2"])
    assert np.array_equal(y_cauchy3_filtered2, data["y_cauchy3_filtered2"])

    assert np.array_equal(y_cauchy_filter1_record2, data["y_cauchy_filter1_record2"])
    assert np.array_equal(y_cauchy_filter2_record2, data["y_cauchy_filter2_record2"])
    assert np.array_equal(y_cauchy_filter3_record2, data["y_cauchy_filter3_record2"])


@pytest.mark.parametrize("frame_name", ["fourier_signal"], indirect=True)
def test_filtered_fourier_signal(load_filter_data, construct_test_signal, construct_fourier_frame, construct_cutoffs):
    data = load_filter_data
    x, dim, n_peaks, amps, peaks, y_sparse = construct_test_signal
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    a0_fourier1 = construct_fourier_frame

    # Main checks:
    # ------------

    y_fourier1 = a0_fourier1.dot(y_sparse)
    csFr_y1_fourier = Filters(y_fourier1, x, cutoffs[0][0], filter_signal=True)
    csFr_y1_fourier.heaviside_lowpass_filter(return_array=False)
    csFr_y1_fourier.fir_filter(return_array=False)

    csFr_y1_fourier.cutoff = cutoffs[1]

    csFr_y1_fourier.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_y1_fourier.butter_filter(order=5, btype="bandstop", return_array=False)

    y_fourier1_filtered1 = csFr_y1_fourier.truncated_a0
    y_fourier_filter1_record1 = csFr_y1_fourier.filter_record(name_only=True)

    csFr_y1_fourier.reset()
    csFr_y1_fourier.butter_filter(return_array=False)
    csFr_y1_fourier.instrumental_lowpass_filter(return_array=False)

    csFr_y1_fourier.cutoff = cutoffs[2][0]

    csFr_y1_fourier.thermal_lowpass_filter(return_array=False)

    y_fourier1_filtered2 = csFr_y1_fourier.truncated_a0
    y_fourier_filter1_record2 = csFr_y1_fourier.filter_record(name_only=True)


    assert np.array_equal(y_fourier1, data["y_fourier1"])
    assert np.array_equal(y_fourier1_filtered1, data["y_fourier1_filtered1"])
    assert np.array_equal(y_fourier_filter1_record1, data["y_fourier_filter1_record1"])
    assert np.array_equal(y_fourier1_filtered2, data["y_fourier1_filtered2"])
    assert np.array_equal(y_fourier_filter1_record2, data["y_fourier_filter1_record2"])



@pytest.mark.parametrize("frame_name", ["heaviside_frame"], indirect=True)
def test_filtered_heaviside_frame(load_filter_data, construct_test_signal_x_components, construct_heaviside_frame, construct_cutoffs):
    data = load_filter_data
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

    a0_heaviside_filter1_record1 = csFr_a01_heaviside.filter_record(name_only=True)
    a0_heaviside_filter2_record1 = csFr_a02_heaviside.filter_record(name_only=True)
    a0_heaviside_filter3_record1 = csFr_a03_heaviside.filter_record(name_only=True)

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

    a0_heaviside_filter1_record2 = csFr_a01_heaviside.filter_record(name_only=True)
    a0_heaviside_filter2_record2 = csFr_a02_heaviside.filter_record(name_only=True)
    a0_heaviside_filter3_record2 = csFr_a03_heaviside.filter_record(name_only=True)


    assert np.array_equal(a0_heaviside1_filtered1, data["a0_heaviside1_filtered1"])
    assert np.array_equal(a0_heaviside2_filtered1, data["a0_heaviside2_filtered1"])
    assert np.array_equal(a0_heaviside3_filtered1, data["a0_heaviside3_filtered1"])

    assert np.array_equal(a0_heaviside_filter1_record1, data["a0_heaviside_filter1_record1"])
    assert np.array_equal(a0_heaviside_filter2_record1, data["a0_heaviside_filter2_record1"])
    assert np.array_equal(a0_heaviside_filter3_record1, data["a0_heaviside_filter3_record1"])

    assert np.array_equal(a0_heaviside1_filtered2, data["a0_heaviside1_filtered2"])
    assert np.array_equal(a0_heaviside2_filtered2, data["a0_heaviside2_filtered2"])
    assert np.array_equal(a0_heaviside3_filtered2, data["a0_heaviside3_filtered2"])

    assert np.array_equal(a0_heaviside_filter1_record2, data["a0_heaviside_filter1_record2"])
    assert np.array_equal(a0_heaviside_filter2_record2, data["a0_heaviside_filter2_record2"])
    assert np.array_equal(a0_heaviside_filter3_record2, data["a0_heaviside_filter3_record2"])


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

    a0_heaviside_overcomplete_filter1_record1 = csFr_a01_heaviside_overcomplete.filter_record(name_only=True)
    a0_heaviside_overcomplete_filter2_record1 = csFr_a02_heaviside_overcomplete.filter_record(name_only=True)
    a0_heaviside_overcomplete_filter3_record1 = csFr_a03_heaviside_overcomplete.filter_record(name_only=True)

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

    a0_heaviside_overcomplete_filter1_record2 = csFr_a01_heaviside_overcomplete.filter_record(name_only=True)
    a0_heaviside_overcomplete_filter2_record2 = csFr_a02_heaviside_overcomplete.filter_record(name_only=True)
    a0_heaviside_overcomplete_filter3_record2 = csFr_a03_heaviside_overcomplete.filter_record(name_only=True)


    assert np.array_equal(a0_heaviside_overcomplete1_filtered1, data["a0_heaviside_overcomplete1_filtered1"])
    assert np.array_equal(a0_heaviside_overcomplete2_filtered1, data["a0_heaviside_overcomplete2_filtered1"])
    assert np.array_equal(a0_heaviside_overcomplete3_filtered1, data["a0_heaviside_overcomplete3_filtered1"])

    assert np.array_equal(a0_heaviside_overcomplete_filter1_record1, data["a0_heaviside_overcomplete_filter1_record1"])
    assert np.array_equal(a0_heaviside_overcomplete_filter2_record1, data["a0_heaviside_overcomplete_filter2_record1"])
    assert np.array_equal(a0_heaviside_overcomplete_filter3_record1, data["a0_heaviside_overcomplete_filter3_record1"])

    assert np.array_equal(a0_heaviside_overcomplete1_filtered2, data["a0_heaviside_overcomplete1_filtered2"])
    assert np.array_equal(a0_heaviside_overcomplete2_filtered2, data["a0_heaviside_overcomplete2_filtered2"])
    assert np.array_equal(a0_heaviside_overcomplete3_filtered2, data["a0_heaviside_overcomplete3_filtered2"])

    assert np.array_equal(a0_heaviside_overcomplete_filter1_record2, data["a0_heaviside_overcomplete_filter1_record2"])
    assert np.array_equal(a0_heaviside_overcomplete_filter2_record2, data["a0_heaviside_overcomplete_filter2_record2"])
    assert np.array_equal(a0_heaviside_overcomplete_filter3_record2, data["a0_heaviside_overcomplete_filter3_record2"])


@pytest.mark.parametrize("frame_name", ["gaussian_frame"], indirect=True)
def test_filtered_gaussian_frame(load_filter_data, construct_test_signal_x_components, construct_gaussian_frame, construct_cutoffs):
    data = load_filter_data
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

    a0_gaussian_filter1_record1 = csFr_a01_gaussian.filter_record(name_only=True)
    a0_gaussian_filter2_record1 = csFr_a02_gaussian.filter_record(name_only=True)
    a0_gaussian_filter3_record1 = csFr_a03_gaussian.filter_record(name_only=True)

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

    a0_gaussian_filter1_record2 = csFr_a01_gaussian.filter_record(name_only=True)
    a0_gaussian_filter2_record2 = csFr_a02_gaussian.filter_record(name_only=True)
    a0_gaussian_filter3_record2 = csFr_a03_gaussian.filter_record(name_only=True)


    assert np.array_equal(a0_gaussian1_filtered1, data["a0_gaussian1_filtered1"])
    assert np.array_equal(a0_gaussian2_filtered1, data["a0_gaussian2_filtered1"])
    assert np.array_equal(a0_gaussian3_filtered1, data["a0_gaussian3_filtered1"])

    assert np.array_equal(a0_gaussian_filter1_record1, data["a0_gaussian_filter1_record1"])
    assert np.array_equal(a0_gaussian_filter2_record1, data["a0_gaussian_filter2_record1"])
    assert np.array_equal(a0_gaussian_filter3_record1, data["a0_gaussian_filter3_record1"])

    assert np.array_equal(a0_gaussian1_filtered2, data["a0_gaussian1_filtered2"])
    assert np.array_equal(a0_gaussian2_filtered2, data["a0_gaussian2_filtered2"])
    assert np.array_equal(a0_gaussian3_filtered2, data["a0_gaussian3_filtered2"])

    assert np.array_equal(a0_gaussian_filter1_record2, data["a0_gaussian_filter1_record2"])
    assert np.array_equal(a0_gaussian_filter2_record2, data["a0_gaussian_filter2_record2"])
    assert np.array_equal(a0_gaussian_filter3_record2, data["a0_gaussian_filter3_record2"])


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

    a0_gaussian_overcomplete_filter1_record1 = csFr_a01_gaussian_overcomplete.filter_record(name_only=True)
    a0_gaussian_overcomplete_filter2_record1 = csFr_a02_gaussian_overcomplete.filter_record(name_only=True)
    a0_gaussian_overcomplete_filter3_record1 = csFr_a03_gaussian_overcomplete.filter_record(name_only=True)

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

    a0_gaussian_overcomplete_filter1_record2 = csFr_a01_gaussian_overcomplete.filter_record(name_only=True)
    a0_gaussian_overcomplete_filter2_record2 = csFr_a02_gaussian_overcomplete.filter_record(name_only=True)
    a0_gaussian_overcomplete_filter3_record2 = csFr_a03_gaussian_overcomplete.filter_record(name_only=True)


    assert np.array_equal(a0_gaussian_overcomplete1_filtered1, data["a0_gaussian_overcomplete1_filtered1"])
    assert np.array_equal(a0_gaussian_overcomplete2_filtered1, data["a0_gaussian_overcomplete2_filtered1"])
    assert np.array_equal(a0_gaussian_overcomplete3_filtered1, data["a0_gaussian_overcomplete3_filtered1"])

    assert np.array_equal(a0_gaussian_overcomplete_filter1_record1, data["a0_gaussian_overcomplete_filter1_record1"])
    assert np.array_equal(a0_gaussian_overcomplete_filter2_record1, data["a0_gaussian_overcomplete_filter2_record1"])
    assert np.array_equal(a0_gaussian_overcomplete_filter3_record1, data["a0_gaussian_overcomplete_filter3_record1"])

    assert np.array_equal(a0_gaussian_overcomplete1_filtered2, data["a0_gaussian_overcomplete1_filtered2"])
    assert np.array_equal(a0_gaussian_overcomplete2_filtered2, data["a0_gaussian_overcomplete2_filtered2"])
    assert np.array_equal(a0_gaussian_overcomplete3_filtered2, data["a0_gaussian_overcomplete3_filtered2"])

    assert np.array_equal(a0_gaussian_overcomplete_filter1_record2, data["a0_gaussian_overcomplete_filter1_record2"])
    assert np.array_equal(a0_gaussian_overcomplete_filter2_record2, data["a0_gaussian_overcomplete_filter2_record2"])
    assert np.array_equal(a0_gaussian_overcomplete_filter3_record2, data["a0_gaussian_overcomplete_filter3_record2"])


@pytest.mark.parametrize("frame_name", ["cauchy_frame"], indirect=True)
def test_filtered_cauchy_frame(load_filter_data, construct_test_signal_x_components, construct_cauchy_frame, construct_cutoffs):
    data = load_filter_data
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

    a0_cauchy_filter1_record1 = csFr_a01_cauchy.filter_record(name_only=True)
    a0_cauchy_filter2_record1 = csFr_a02_cauchy.filter_record(name_only=True)
    a0_cauchy_filter3_record1 = csFr_a03_cauchy.filter_record(name_only=True)

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

    a0_cauchy_filter1_record2 = csFr_a01_cauchy.filter_record(name_only=True)
    a0_cauchy_filter2_record2 = csFr_a02_cauchy.filter_record(name_only=True)
    a0_cauchy_filter3_record2 = csFr_a03_cauchy.filter_record(name_only=True)


    assert np.array_equal(a0_cauchy1_filtered1, data["a0_cauchy1_filtered1"])
    assert np.array_equal(a0_cauchy2_filtered1, data["a0_cauchy2_filtered1"])
    assert np.array_equal(a0_cauchy3_filtered1, data["a0_cauchy3_filtered1"])

    assert np.array_equal(a0_cauchy_filter1_record1, data["a0_cauchy_filter1_record1"])
    assert np.array_equal(a0_cauchy_filter2_record1, data["a0_cauchy_filter2_record1"])
    assert np.array_equal(a0_cauchy_filter3_record1, data["a0_cauchy_filter3_record1"])

    assert np.array_equal(a0_cauchy1_filtered2, data["a0_cauchy1_filtered2"])
    assert np.array_equal(a0_cauchy2_filtered2, data["a0_cauchy2_filtered2"])
    assert np.array_equal(a0_cauchy3_filtered2, data["a0_cauchy3_filtered2"])

    assert np.array_equal(a0_cauchy_filter1_record2, data["a0_cauchy_filter1_record2"])
    assert np.array_equal(a0_cauchy_filter2_record2, data["a0_cauchy_filter2_record2"])
    assert np.array_equal(a0_cauchy_filter3_record2, data["a0_cauchy_filter3_record2"])


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

    a0_cauchy_overcomplete_filter1_record1 = csFr_a01_cauchy_overcomplete.filter_record(name_only=True)
    a0_cauchy_overcomplete_filter2_record1 = csFr_a02_cauchy_overcomplete.filter_record(name_only=True)
    a0_cauchy_overcomplete_filter3_record1 = csFr_a03_cauchy_overcomplete.filter_record(name_only=True)

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

    a0_cauchy_overcomplete_filter1_record2 = csFr_a01_cauchy_overcomplete.filter_record(name_only=True)
    a0_cauchy_overcomplete_filter2_record2 = csFr_a02_cauchy_overcomplete.filter_record(name_only=True)
    a0_cauchy_overcomplete_filter3_record2 = csFr_a03_cauchy_overcomplete.filter_record(name_only=True)


    assert np.array_equal(a0_cauchy_overcomplete1_filtered1, data["a0_cauchy_overcomplete1_filtered1"])
    assert np.array_equal(a0_cauchy_overcomplete2_filtered1, data["a0_cauchy_overcomplete2_filtered1"])
    assert np.array_equal(a0_cauchy_overcomplete3_filtered1, data["a0_cauchy_overcomplete3_filtered1"])

    assert np.array_equal(a0_cauchy_overcomplete_filter1_record1, data["a0_cauchy_overcomplete_filter1_record1"])
    assert np.array_equal(a0_cauchy_overcomplete_filter2_record1, data["a0_cauchy_overcomplete_filter2_record1"])
    assert np.array_equal(a0_cauchy_overcomplete_filter3_record1, data["a0_cauchy_overcomplete_filter3_record1"])

    assert np.array_equal(a0_cauchy_overcomplete1_filtered2, data["a0_cauchy_overcomplete1_filtered2"])
    assert np.array_equal(a0_cauchy_overcomplete2_filtered2, data["a0_cauchy_overcomplete2_filtered2"])
    assert np.array_equal(a0_cauchy_overcomplete3_filtered2, data["a0_cauchy_overcomplete3_filtered2"])

    assert np.array_equal(a0_cauchy_overcomplete_filter1_record2, data["a0_cauchy_overcomplete_filter1_record2"])
    assert np.array_equal(a0_cauchy_overcomplete_filter2_record2, data["a0_cauchy_overcomplete_filter2_record2"])
    assert np.array_equal(a0_cauchy_overcomplete_filter3_record2, data["a0_cauchy_overcomplete_filter3_record2"])


@pytest.mark.parametrize("frame_name", ["fourier_frame"], indirect=True)
def test_filtered_fourier_frame(load_filter_data, construct_test_signal_x_components, construct_fourier_frame, construct_cutoffs):
    data = load_filter_data
    x = construct_test_signal_x_components
    cutoffs = construct_cutoffs


    # Preliminary constructions:
    # --------------------------

    a0_fourier1 = construct_fourier_frame


    # Main checks:
    # ------------

    csFr_a01_fourier = Filters(a0_fourier1, x, cutoffs[0][0])
    csFr_a01_fourier.heaviside_lowpass_filter(return_array=False)
    csFr_a01_fourier.fir_filter(return_array=False)

    csFr_a01_fourier.cutoff = cutoffs[1]

    csFr_a01_fourier.fir_filter(numtabs=5, pass_zero="bandstop", return_array=False)
    csFr_a01_fourier.butter_filter(order=5, btype="bandstop", return_array=False)

    a0_fourier1_filtered1 = csFr_a01_fourier.truncated_a0
    a0_fourier_filter1_record1 = csFr_a01_fourier.filter_record(name_only=True)

    csFr_a01_fourier.reset()
    csFr_a01_fourier.butter_filter(return_array=False)
    csFr_a01_fourier.instrumental_lowpass_filter(return_array=False)

    csFr_a01_fourier.cutoff = cutoffs[2][0]

    csFr_a01_fourier.thermal_lowpass_filter(return_array=False)

    a0_fourier1_filtered2 = csFr_a01_fourier.truncated_a0
    a0_fourier_filter1_record2 = csFr_a01_fourier.filter_record(name_only=True)


    assert np.array_equal(a0_fourier1_filtered1, data["a0_fourier1_filtered1"])
    assert np.array_equal(a0_fourier_filter1_record1, data["a0_fourier_filter1_record1"])
    assert np.array_equal(a0_fourier1_filtered2, data["a0_fourier1_filtered2"])
    assert np.array_equal(a0_fourier_filter1_record2, data["a0_fourier_filter1_record2"])


if __name__ == "__main__":
    pytest.main([__file__])