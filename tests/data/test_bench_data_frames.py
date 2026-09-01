"""
Module to construct test bench data for the frames module of the cssr package.

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


# =============================================================================
# Trial test data
# =============================================================================

x = np.arange(0, 20, 0.1)
dim = len(x)

# =============================================================================
# Start test data construction
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
