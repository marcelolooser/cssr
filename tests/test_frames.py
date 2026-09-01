"""
Test module for the frames module of the cssr package.

@author: marcelo looser
"""


import pytest
import numpy as np

# =============================================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cssr import Frames

# =============================================================================

@pytest.fixture
def dir_frames():
    return "data/data_frames/"


@pytest.fixture
def frame_name(request):
    return request.param


@pytest.fixture
def load_frame_data(dir_frames, frame_name):
    return np.load(dir_frames + frame_name + ".npz")


@pytest.fixture
def construct_test_signal_x_components():
    return np.arange(0, 20, 0.1)


@pytest.mark.parametrize("frame_name", ["heaviside"], indirect=True)
def test_heaviside(load_frame_data, construct_test_signal_x_components):
    data = load_frame_data
    x = construct_test_signal_x_components
    csF = Frames(x)

    # Preliminary checks:
    # -------------------

    dim = len(x)
    bw_heaviside = [1, dim//2, dim - 1]
    bw_heaviside_overcomplete = [[None, None], [dim//2 - 3, dim//2 + 3], [dim//2 - 3, dim//2 + 3]]
    ss_heaviside_overcomplete = [None, None, 3]

    assert np.array_equal(x, data["x"])
    assert np.array_equal(bw_heaviside, data["bw_heaviside"])
    assert np.array_equal(np.array(bw_heaviside_overcomplete, dtype=float), data["bw_heaviside_overcomplete"], equal_nan=True)
    assert np.array_equal(np.array(ss_heaviside_overcomplete, dtype=float), data["ss_heaviside_overcomplete"], equal_nan=True)

    # Main checks:
    # ------------

    a0_heaviside1 = csF.heaviside(bw_heaviside[0])
    a0_heaviside2 = csF.heaviside(bw_heaviside[1])
    a0_heaviside3 = csF.heaviside(bw_heaviside[2])

    a0_heaviside_overcomplete1 = csF.heaviside_overcomplete(
        bw_heaviside_overcomplete[0][0], ss_heaviside_overcomplete[0]
        )
    a0_heaviside_overcomplete2 = csF.heaviside_overcomplete(
        bw_heaviside_overcomplete[1], ss_heaviside_overcomplete[1]
        )
    a0_heaviside_overcomplete3 = csF.heaviside_overcomplete(
        bw_heaviside_overcomplete[2], ss_heaviside_overcomplete[2]
        )

    assert np.array_equal(a0_heaviside1, data["a0_heaviside1"])
    assert np.array_equal(a0_heaviside2, data["a0_heaviside2"])
    assert np.array_equal(a0_heaviside3, data["a0_heaviside3"])

    assert np.array_equal(a0_heaviside_overcomplete1, data["a0_heaviside_overcomplete1"])
    assert np.array_equal(a0_heaviside_overcomplete2, data["a0_heaviside_overcomplete2"])
    assert np.array_equal(a0_heaviside_overcomplete3, data["a0_heaviside_overcomplete3"])


@pytest.mark.parametrize("frame_name", ["gaussian"], indirect=True)
def test_gaussian(load_frame_data, construct_test_signal_x_components):
    data = load_frame_data
    x = construct_test_signal_x_components
    csF = Frames(x)

    # Preliminary checks:
    # -------------------

    bw_gaussian = [abs(x[1] - x[0]), abs(x[-1] - x[0])/2, abs(x[-1] - x[0]) - abs(x[1] - x[0])]
    bw_gaussian_overcomplete = [[None, None], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])]]
    ss_gaussian_overcomplete = [None, None, 3*abs(x[1] - x[0])]

    assert np.array_equal(x, data["x"])
    assert np.array_equal(bw_gaussian, data["bw_gaussian"])
    assert np.array_equal(np.array(bw_gaussian_overcomplete, dtype=float), data["bw_gaussian_overcomplete"], equal_nan=True)
    assert np.array_equal(np.array(ss_gaussian_overcomplete, dtype=float), data["ss_gaussian_overcomplete"], equal_nan=True)

    # Main checks:
    # ------------

    a0_gaussian1 = csF.gaussian(bw_gaussian[0])
    a0_gaussian2 = csF.gaussian(bw_gaussian[1])
    a0_gaussian3 = csF.gaussian(bw_gaussian[2])

    a0_gaussian_overcomplete1 = csF.gaussian_overcomplete(
        bw_gaussian_overcomplete[0][0], ss_gaussian_overcomplete[0]
        )
    a0_gaussian_overcomplete2 = csF.gaussian_overcomplete(
        bw_gaussian_overcomplete[1], ss_gaussian_overcomplete[1]
        )
    a0_gaussian_overcomplete3 = csF.gaussian_overcomplete(
        bw_gaussian_overcomplete[2], ss_gaussian_overcomplete[2]
        )

    assert np.array_equal(a0_gaussian1, data["a0_gaussian1"])
    assert np.array_equal(a0_gaussian2, data["a0_gaussian2"])
    assert np.array_equal(a0_gaussian3, data["a0_gaussian3"])

    assert np.array_equal(a0_gaussian_overcomplete1, data["a0_gaussian_overcomplete1"])
    assert np.array_equal(a0_gaussian_overcomplete2, data["a0_gaussian_overcomplete2"])
    assert np.array_equal(a0_gaussian_overcomplete3, data["a0_gaussian_overcomplete3"])


@pytest.mark.parametrize("frame_name", ["cauchy"], indirect=True)
def test_cauchy(load_frame_data, construct_test_signal_x_components):
    data = load_frame_data
    x = construct_test_signal_x_components
    csF = Frames(x)

    # Preliminary checks:
    # -------------------

    bw_cauchy = [abs(x[1] - x[0]), abs(x[-1] - x[0])/2, abs(x[-1] - x[0]) - abs(x[1] - x[0])]
    bw_cauchy_overcomplete = [[None, None], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])], [abs(x[-1] - x[0])/2 - 3*abs(x[1] - x[0]), abs(x[-1] - x[0])/2 + 3*abs(x[1] - x[0])]]
    ss_cauchy_overcomplete = [None, None, 3*abs(x[1] - x[0])]

    assert np.array_equal(x, data["x"])
    assert np.array_equal(bw_cauchy, data["bw_cauchy"])
    assert np.array_equal(np.array(bw_cauchy_overcomplete, dtype=float), data["bw_cauchy_overcomplete"], equal_nan=True)
    assert np.array_equal(np.array(ss_cauchy_overcomplete, dtype=float), data["ss_cauchy_overcomplete"], equal_nan=True)

    # Main checks:
    # ------------

    a0_cauchy1 = csF.cauchy(bw_cauchy[0])
    a0_cauchy2 = csF.cauchy(bw_cauchy[1])
    a0_cauchy3 = csF.cauchy(bw_cauchy[2])

    a0_cauchy_overcomplete1 = csF.cauchy_overcomplete(
        bw_cauchy_overcomplete[0][0], ss_cauchy_overcomplete[0]
        )
    a0_cauchy_overcomplete2 = csF.cauchy_overcomplete(
        bw_cauchy_overcomplete[1], ss_cauchy_overcomplete[1]
        )
    a0_cauchy_overcomplete3 = csF.cauchy_overcomplete(
        bw_cauchy_overcomplete[2], ss_cauchy_overcomplete[2]
        )

    assert np.array_equal(a0_cauchy1, data["a0_cauchy1"])
    assert np.array_equal(a0_cauchy2, data["a0_cauchy2"])
    assert np.array_equal(a0_cauchy3, data["a0_cauchy3"])

    assert np.array_equal(a0_cauchy_overcomplete1, data["a0_cauchy_overcomplete1"])
    assert np.array_equal(a0_cauchy_overcomplete2, data["a0_cauchy_overcomplete2"])
    assert np.array_equal(a0_cauchy_overcomplete3, data["a0_cauchy_overcomplete3"])


@pytest.mark.parametrize("frame_name", ["fourier"], indirect=True)
def test_fourier(load_frame_data, construct_test_signal_x_components):
    data = load_frame_data
    x = construct_test_signal_x_components
    csF = Frames(x)

    # Preliminary checks:
    # -------------------

    assert np.array_equal(x, data["x"])

    # Main checks:
    # ------------

    a0_fourier1 = csF.fourier()

    assert np.array_equal(a0_fourier1 , data["a0_fourier1"])


if __name__ == "__main__":
    pytest.main([__file__])