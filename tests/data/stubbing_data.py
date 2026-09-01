"""
Module to construct stubbing data for the cssr package test bench.

@author: marcelo looser
"""

import scipy
import random
import numpy as np


# =============================================================================
# Helper functions:
# =============================================================================


def load_filter_data(dir_filters, frame_name):
    return np.load(dir_filters + frame_name + ".npz")

def load_measurement_matrcix_data(dir_measurement_matrcices, frame_name):
    return np.load(dir_measurement_matrcices + frame_name + ".npz")


# =============================================================================
# Preliminaries:
# =============================================================================


dir_stubs = "data_stubs/"
dir_filters = "data_filters/"
dir_measurement_matrices = "data_measurement_matrices/"


set_frame_stubs_heaviside = True
save_frame_stubs_heaviside = True

set_frame_stubs_gaussian = True
save_frame_stubs_gaussian = True

set_frame_stubs_cauchy = True
save_frame_stubs_cauchy = True

set_frame_stubs_fourier = True
save_frame_stubs_fourier = True


# =============================================================================
# Main:
# =============================================================================

if set_frame_stubs_heaviside:

    np.random.seed(9)
    random.seed(9)

    number_samples_heaviside = 25
    data = load_filter_data(dir_filters, "heaviside_frame")

    # Dictonaries:
    # ------------

    a0_heaviside1_filtered1 = data["a0_heaviside1_filtered1"]
    a0_heaviside2_filtered1 = data["a0_heaviside2_filtered1"]
    a0_heaviside3_filtered1 = data["a0_heaviside3_filtered1"]

    m_heaviside1, n_heaviside1 = a0_heaviside1_filtered1.shape
    m_heaviside2, n_heaviside2 = a0_heaviside2_filtered1.shape
    m_heaviside3, n_heaviside3 = a0_heaviside3_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_heaviside1 = np.random.random(size=(number_samples_heaviside, m_heaviside1))
    uniform_ns_m_heaviside2 = np.random.random(size=(number_samples_heaviside, m_heaviside2))
    uniform_ns_m_heaviside3 = np.random.random(size=(number_samples_heaviside, m_heaviside3))

    uniform_ns_n_heaviside1 = np.random.random(size=(number_samples_heaviside, n_heaviside1))
    uniform_ns_n_heaviside2 = np.random.random(size=(number_samples_heaviside, n_heaviside2))
    uniform_ns_n_heaviside3 = np.random.random(size=(number_samples_heaviside, n_heaviside3))

    probability = 0.5
    bernoulli_ns_m_heaviside1 = np.random.binomial(n=1, p=probability, size=(number_samples_heaviside, m_heaviside1))
    bernoulli_ns_m_heaviside2 = np.random.binomial(n=1, p=probability, size=(number_samples_heaviside, m_heaviside2))
    bernoulli_ns_m_heaviside3 = np.random.binomial(n=1, p=probability, size=(number_samples_heaviside, m_heaviside3))

    samples_m_ns_heaviside1 = random.sample(range(m_heaviside1), number_samples_heaviside)
    samples_m_ns_heaviside2 = random.sample(range(m_heaviside2), number_samples_heaviside)
    samples_m_ns_heaviside3 = random.sample(range(m_heaviside3), number_samples_heaviside)

    samples_n_m_heaviside1 = random.sample(range(n_heaviside1), m_heaviside1)
    samples_n_m_heaviside2 = random.sample(range(n_heaviside2), m_heaviside2)
    samples_n_m_heaviside3 = random.sample(range(n_heaviside3), m_heaviside3)

    samples_binary_m_heaviside1 = np.random.choice([-1,1], size=m_heaviside1)
    samples_binary_m_heaviside2 = np.random.choice([-1,1], size=m_heaviside2)
    samples_binary_m_heaviside3 = np.random.choice([-1,1], size=m_heaviside3)

    gaussian_ns_m_heaviside1 = np.random.randn(number_samples_heaviside, m_heaviside1)
    gaussian_ns_m_heaviside2 = np.random.randn(number_samples_heaviside, m_heaviside2)
    gaussian_ns_m_heaviside3 = np.random.randn(number_samples_heaviside, m_heaviside3)



    # Overcomplete dictonaries:
    # -------------------------

    a0_heaviside_overcomplete1_filtered1 = data["a0_heaviside_overcomplete1_filtered1"]
    a0_heaviside_overcomplete2_filtered1 = data["a0_heaviside_overcomplete2_filtered1"]
    a0_heaviside_overcomplete3_filtered1 = data["a0_heaviside_overcomplete3_filtered1"]

    m_heaviside_overcomplete1, n_heaviside_overcomplete1 = a0_heaviside_overcomplete1_filtered1.shape
    m_heaviside_overcomplete2, n_heaviside_overcomplete2 = a0_heaviside_overcomplete2_filtered1.shape
    m_heaviside_overcomplete3, n_heaviside_overcomplete3 = a0_heaviside_overcomplete3_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_heaviside_overcomplete1 = np.random.random(size=(number_samples_heaviside, m_heaviside_overcomplete1))
    uniform_ns_m_heaviside_overcomplete2 = np.random.random(size=(number_samples_heaviside, m_heaviside_overcomplete2))
    uniform_ns_m_heaviside_overcomplete3 = np.random.random(size=(number_samples_heaviside, m_heaviside_overcomplete3))

    uniform_ns_n_heaviside_overcomplete1 = np.random.random(size=(number_samples_heaviside, n_heaviside_overcomplete1))
    uniform_ns_n_heaviside_overcomplete2 = np.random.random(size=(number_samples_heaviside, n_heaviside_overcomplete2))
    uniform_ns_n_heaviside_overcomplete3 = np.random.random(size=(number_samples_heaviside, n_heaviside_overcomplete3))

    probability = 0.5
    bernoulli_ns_m_heaviside_overcomplete1 = np.random.binomial(n=1, p=probability, size=(number_samples_heaviside, m_heaviside_overcomplete1))
    bernoulli_ns_m_heaviside_overcomplete2 = np.random.binomial(n=1, p=probability, size=(number_samples_heaviside, m_heaviside_overcomplete2))
    bernoulli_ns_m_heaviside_overcomplete3 = np.random.binomial(n=1, p=probability, size=(number_samples_heaviside, m_heaviside_overcomplete3))

    samples_m_ns_heaviside_overcomplete1 = random.sample(range(m_heaviside_overcomplete1), number_samples_heaviside)
    samples_m_ns_heaviside_overcomplete2 = random.sample(range(m_heaviside_overcomplete2), number_samples_heaviside)
    samples_m_ns_heaviside_overcomplete3 = random.sample(range(m_heaviside_overcomplete3), number_samples_heaviside)

    if n_heaviside_overcomplete1 >= m_heaviside_overcomplete1:
        samples_n_m_heaviside_overcomplete1 = random.sample(range(n_heaviside_overcomplete1), m_heaviside_overcomplete1)
    else:
        samples_n_m_heaviside_overcomplete1 = random.sample(range(m_heaviside_overcomplete1), m_heaviside_overcomplete1)

    if n_heaviside_overcomplete2 >= m_heaviside_overcomplete2:
        samples_n_m_heaviside_overcomplete2 = random.sample(range(n_heaviside_overcomplete2), m_heaviside_overcomplete2)
    else:
        samples_n_m_heaviside_overcomplete2 = random.sample(range(m_heaviside_overcomplete2), m_heaviside_overcomplete2)

    if n_heaviside_overcomplete3 >= m_heaviside_overcomplete3:
        samples_n_m_heaviside_overcomplete3 = random.sample(range(n_heaviside_overcomplete3), m_heaviside_overcomplete3)
    else:
        samples_n_m_heaviside_overcomplete3 = random.sample(range(m_heaviside_overcomplete3), m_heaviside_overcomplete3)

    samples_binary_m_heaviside_overcomplete1 = np.random.choice([-1,1], size=m_heaviside_overcomplete1)
    samples_binary_m_heaviside_overcomplete2 = np.random.choice([-1,1], size=m_heaviside_overcomplete2)
    samples_binary_m_heaviside_overcomplete3 = np.random.choice([-1,1], size=m_heaviside_overcomplete3)

    gaussian_ns_m_heaviside_overcomplete1 = np.random.randn(number_samples_heaviside, m_heaviside_overcomplete1)
    gaussian_ns_m_heaviside_overcomplete2 = np.random.randn(number_samples_heaviside, m_heaviside_overcomplete2)
    gaussian_ns_m_heaviside_overcomplete3 = np.random.randn(number_samples_heaviside, m_heaviside_overcomplete3)

    random_unitary_ns_m_heaviside = scipy.stats.unitary_group.rvs(number_samples_heaviside)

    tracked_shapes_heaviside = np.array((
        (number_samples_heaviside, m_heaviside1),
        (number_samples_heaviside, m_heaviside2),
        (number_samples_heaviside, m_heaviside3),

        (number_samples_heaviside, n_heaviside1),
        (number_samples_heaviside, n_heaviside2),
        (number_samples_heaviside, n_heaviside3),

        (number_samples_heaviside, m_heaviside_overcomplete1),
        (number_samples_heaviside, m_heaviside_overcomplete2),
        (number_samples_heaviside, m_heaviside_overcomplete3),

        (number_samples_heaviside, n_heaviside_overcomplete1),
        (number_samples_heaviside, n_heaviside_overcomplete2),
        (number_samples_heaviside, n_heaviside_overcomplete3),


        (number_samples_heaviside, m_heaviside1),
        (number_samples_heaviside, m_heaviside2),
        (number_samples_heaviside, m_heaviside3),

        (number_samples_heaviside, m_heaviside_overcomplete1),
        (number_samples_heaviside, m_heaviside_overcomplete2),
        (number_samples_heaviside, m_heaviside_overcomplete3),


        (m_heaviside1, number_samples_heaviside),
        (m_heaviside2, number_samples_heaviside),
        (m_heaviside3, number_samples_heaviside),

        (n_heaviside1, m_heaviside1),
        (n_heaviside2, m_heaviside2),
        (n_heaviside3, m_heaviside3),

        (m_heaviside_overcomplete1, number_samples_heaviside),
        (m_heaviside_overcomplete2, number_samples_heaviside),
        (m_heaviside_overcomplete3, number_samples_heaviside),

        (n_heaviside_overcomplete1 if n_heaviside_overcomplete1 >= m_heaviside_overcomplete1 else m_heaviside_overcomplete1, m_heaviside_overcomplete1),
        (n_heaviside_overcomplete2 if n_heaviside_overcomplete2 >= m_heaviside_overcomplete2 else m_heaviside_overcomplete2, m_heaviside_overcomplete2),
        (n_heaviside_overcomplete3 if n_heaviside_overcomplete3 >= m_heaviside_overcomplete3 else m_heaviside_overcomplete3, m_heaviside_overcomplete3),


        (m_heaviside1, -1),
        (m_heaviside2, -1),
        (m_heaviside3, -1),

        (m_heaviside_overcomplete1, -1),
        (m_heaviside_overcomplete2, -1),
        (m_heaviside_overcomplete3, -1),

        (number_samples_heaviside, m_heaviside1),
        (number_samples_heaviside, m_heaviside2),
        (number_samples_heaviside, m_heaviside3),

        (number_samples_heaviside, m_heaviside_overcomplete1),
        (number_samples_heaviside, m_heaviside_overcomplete2),
        (number_samples_heaviside, m_heaviside_overcomplete3),

        (number_samples_heaviside, -1)

        ), dtype=int)


if save_frame_stubs_heaviside:
    np.savez(dir_stubs + "heaviside",
             probability = 0.5,
             tracked_shapes_heaviside = tracked_shapes_heaviside,
             uniform_ns_m_heaviside1 = uniform_ns_m_heaviside1,
             uniform_ns_m_heaviside2 = uniform_ns_m_heaviside2,
             uniform_ns_m_heaviside3 = uniform_ns_m_heaviside3,

             uniform_ns_n_heaviside1 = uniform_ns_n_heaviside1,
             uniform_ns_n_heaviside2 = uniform_ns_n_heaviside2,
             uniform_ns_n_heaviside3 = uniform_ns_n_heaviside3,

             bernoulli_ns_m_heaviside1 = bernoulli_ns_m_heaviside1,
             bernoulli_ns_m_heaviside2 = bernoulli_ns_m_heaviside2,
             bernoulli_ns_m_heaviside3 = bernoulli_ns_m_heaviside3,

             samples_m_ns_heaviside1 = samples_m_ns_heaviside1,
             samples_m_ns_heaviside2 = samples_m_ns_heaviside2,
             samples_m_ns_heaviside3 = samples_m_ns_heaviside3,

             samples_n_m_heaviside1 = samples_n_m_heaviside1,
             samples_n_m_heaviside2 = samples_n_m_heaviside2,
             samples_n_m_heaviside3 = samples_n_m_heaviside3,

             samples_binary_m_heaviside1 = samples_binary_m_heaviside1,
             samples_binary_m_heaviside2 = samples_binary_m_heaviside2,
             samples_binary_m_heaviside3 = samples_binary_m_heaviside3,

             gaussian_ns_m_heaviside1 = gaussian_ns_m_heaviside1,
             gaussian_ns_m_heaviside2 = gaussian_ns_m_heaviside2,
             gaussian_ns_m_heaviside3 = gaussian_ns_m_heaviside3,


             uniform_ns_m_heaviside_overcomplete1 = uniform_ns_m_heaviside_overcomplete1,
             uniform_ns_m_heaviside_overcomplete2 = uniform_ns_m_heaviside_overcomplete2,
             uniform_ns_m_heaviside_overcomplete3 = uniform_ns_m_heaviside_overcomplete3,

             uniform_ns_n_heaviside_overcomplete1 = uniform_ns_n_heaviside_overcomplete1,
             uniform_ns_n_heaviside_overcomplete2 = uniform_ns_n_heaviside_overcomplete2,
             uniform_ns_n_heaviside_overcomplete3 = uniform_ns_n_heaviside_overcomplete3,

             bernoulli_ns_m_heaviside_overcomplete1 = bernoulli_ns_m_heaviside_overcomplete1,
             bernoulli_ns_m_heaviside_overcomplete2 = bernoulli_ns_m_heaviside_overcomplete2,
             bernoulli_ns_m_heaviside_overcomplete3 = bernoulli_ns_m_heaviside_overcomplete3,

             samples_m_ns_heaviside_overcomplete1 = samples_m_ns_heaviside_overcomplete1,
             samples_m_ns_heaviside_overcomplete2 = samples_m_ns_heaviside_overcomplete2,
             samples_m_ns_heaviside_overcomplete3 = samples_m_ns_heaviside_overcomplete3,

             samples_n_m_heaviside_overcomplete1 = samples_n_m_heaviside_overcomplete1,
             samples_n_m_heaviside_overcomplete2 = samples_n_m_heaviside_overcomplete2,
             samples_n_m_heaviside_overcomplete3 = samples_n_m_heaviside_overcomplete3,

             samples_binary_m_heaviside_overcomplete1 = samples_binary_m_heaviside_overcomplete1,
             samples_binary_m_heaviside_overcomplete2 = samples_binary_m_heaviside_overcomplete2,
             samples_binary_m_heaviside_overcomplete3 = samples_binary_m_heaviside_overcomplete3,

             gaussian_ns_m_heaviside_overcomplete1 = gaussian_ns_m_heaviside_overcomplete1,
             gaussian_ns_m_heaviside_overcomplete2 = gaussian_ns_m_heaviside_overcomplete2,
             gaussian_ns_m_heaviside_overcomplete3 = gaussian_ns_m_heaviside_overcomplete3,

             random_unitary_ns_m_heaviside = random_unitary_ns_m_heaviside
             )



if set_frame_stubs_gaussian:

    np.random.seed(9)
    random.seed(9)

    number_samples_gaussian = 25
    data = load_filter_data(dir_filters, "gaussian_frame")

    # Dictonaries:
    # ------------

    a0_gaussian1_filtered1 = data["a0_gaussian1_filtered1"]
    a0_gaussian2_filtered1 = data["a0_gaussian2_filtered1"]
    a0_gaussian3_filtered1 = data["a0_gaussian3_filtered1"]

    m_gaussian1, n_gaussian1 = a0_gaussian1_filtered1.shape
    m_gaussian2, n_gaussian2 = a0_gaussian2_filtered1.shape
    m_gaussian3, n_gaussian3 = a0_gaussian3_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_gaussian1 = np.random.random(size=(number_samples_gaussian, m_gaussian1))
    uniform_ns_m_gaussian2 = np.random.random(size=(number_samples_gaussian, m_gaussian2))
    uniform_ns_m_gaussian3 = np.random.random(size=(number_samples_gaussian, m_gaussian3))

    uniform_ns_n_gaussian1 = np.random.random(size=(number_samples_gaussian, n_gaussian1))
    uniform_ns_n_gaussian2 = np.random.random(size=(number_samples_gaussian, n_gaussian2))
    uniform_ns_n_gaussian3 = np.random.random(size=(number_samples_gaussian, n_gaussian3))

    probability = 0.5
    bernoulli_ns_m_gaussian1 = np.random.binomial(n=1, p=probability, size=(number_samples_gaussian, m_gaussian1))
    bernoulli_ns_m_gaussian2 = np.random.binomial(n=1, p=probability, size=(number_samples_gaussian, m_gaussian2))
    bernoulli_ns_m_gaussian3 = np.random.binomial(n=1, p=probability, size=(number_samples_gaussian, m_gaussian3))

    samples_m_ns_gaussian1 = random.sample(range(m_gaussian1), number_samples_gaussian)
    samples_m_ns_gaussian2 = random.sample(range(m_gaussian2), number_samples_gaussian)
    samples_m_ns_gaussian3 = random.sample(range(m_gaussian3), number_samples_gaussian)

    samples_n_m_gaussian1 = random.sample(range(n_gaussian1), m_gaussian1)
    samples_n_m_gaussian2 = random.sample(range(n_gaussian2), m_gaussian2)
    samples_n_m_gaussian3 = random.sample(range(n_gaussian3), m_gaussian3)

    samples_binary_m_gaussian1 = np.random.choice([-1,1], size=m_gaussian1)
    samples_binary_m_gaussian2 = np.random.choice([-1,1], size=m_gaussian2)
    samples_binary_m_gaussian3 = np.random.choice([-1,1], size=m_gaussian3)

    gaussian_ns_m_gaussian1 = np.random.randn(number_samples_gaussian, m_gaussian1)
    gaussian_ns_m_gaussian2 = np.random.randn(number_samples_gaussian, m_gaussian2)
    gaussian_ns_m_gaussian3 = np.random.randn(number_samples_gaussian, m_gaussian3)


    # Overcomplete dictonaries:
    # -------------------------

    a0_gaussian_overcomplete1_filtered1 = data["a0_gaussian_overcomplete1_filtered1"]
    a0_gaussian_overcomplete2_filtered1 = data["a0_gaussian_overcomplete2_filtered1"]
    a0_gaussian_overcomplete3_filtered1 = data["a0_gaussian_overcomplete3_filtered1"]

    m_gaussian_overcomplete1, n_gaussian_overcomplete1 = a0_gaussian_overcomplete1_filtered1.shape
    m_gaussian_overcomplete2, n_gaussian_overcomplete2 = a0_gaussian_overcomplete2_filtered1.shape
    m_gaussian_overcomplete3, n_gaussian_overcomplete3 = a0_gaussian_overcomplete3_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_gaussian_overcomplete1 = np.random.random(size=(number_samples_gaussian, m_gaussian_overcomplete1))
    uniform_ns_m_gaussian_overcomplete2 = np.random.random(size=(number_samples_gaussian, m_gaussian_overcomplete2))
    uniform_ns_m_gaussian_overcomplete3 = np.random.random(size=(number_samples_gaussian, m_gaussian_overcomplete3))

    uniform_ns_n_gaussian_overcomplete1 = np.random.random(size=(number_samples_gaussian, n_gaussian_overcomplete1))
    uniform_ns_n_gaussian_overcomplete2 = np.random.random(size=(number_samples_gaussian, n_gaussian_overcomplete2))
    uniform_ns_n_gaussian_overcomplete3 = np.random.random(size=(number_samples_gaussian, n_gaussian_overcomplete3))

    probability = 0.5
    bernoulli_ns_m_gaussian_overcomplete1 = np.random.binomial(n=1, p=probability, size=(number_samples_gaussian, m_gaussian_overcomplete1))
    bernoulli_ns_m_gaussian_overcomplete2 = np.random.binomial(n=1, p=probability, size=(number_samples_gaussian, m_gaussian_overcomplete2))
    bernoulli_ns_m_gaussian_overcomplete3 = np.random.binomial(n=1, p=probability, size=(number_samples_gaussian, m_gaussian_overcomplete3))

    samples_m_ns_gaussian_overcomplete1 = random.sample(range(m_gaussian_overcomplete1), number_samples_gaussian)
    samples_m_ns_gaussian_overcomplete2 = random.sample(range(m_gaussian_overcomplete2), number_samples_gaussian)
    samples_m_ns_gaussian_overcomplete3 = random.sample(range(m_gaussian_overcomplete3), number_samples_gaussian)

    if n_gaussian_overcomplete1 >= m_gaussian_overcomplete1:
        samples_n_m_gaussian_overcomplete1 = random.sample(range(n_gaussian_overcomplete1), m_gaussian_overcomplete1)
    else:
        samples_n_m_gaussian_overcomplete1 = random.sample(range(m_gaussian_overcomplete1), m_gaussian_overcomplete1)

    if n_gaussian_overcomplete2 >= m_gaussian_overcomplete2:
        samples_n_m_gaussian_overcomplete2 = random.sample(range(n_gaussian_overcomplete2), m_gaussian_overcomplete2)
    else:
        samples_n_m_gaussian_overcomplete2 = random.sample(range(m_gaussian_overcomplete2), m_gaussian_overcomplete2)

    if n_gaussian_overcomplete3 >= m_gaussian_overcomplete3:
        samples_n_m_gaussian_overcomplete3 = random.sample(range(n_gaussian_overcomplete3), m_gaussian_overcomplete3)
    else:
        samples_n_m_gaussian_overcomplete3 = random.sample(range(m_gaussian_overcomplete3), m_gaussian_overcomplete3)

    samples_binary_m_gaussian_overcomplete1 = np.random.choice([-1,1], size=m_gaussian_overcomplete1)
    samples_binary_m_gaussian_overcomplete2 = np.random.choice([-1,1], size=m_gaussian_overcomplete2)
    samples_binary_m_gaussian_overcomplete3 = np.random.choice([-1,1], size=m_gaussian_overcomplete3)

    gaussian_ns_m_gaussian_overcomplete1 = np.random.randn(number_samples_gaussian, m_gaussian_overcomplete1)
    gaussian_ns_m_gaussian_overcomplete2 = np.random.randn(number_samples_gaussian, m_gaussian_overcomplete2)
    gaussian_ns_m_gaussian_overcomplete3 = np.random.randn(number_samples_gaussian, m_gaussian_overcomplete3)

    random_unitary_ns_m_gaussian = scipy.stats.unitary_group.rvs(number_samples_gaussian)

    tracked_shapes_gaussian = np.array((
        (number_samples_gaussian, m_gaussian1),
        (number_samples_gaussian, m_gaussian2),
        (number_samples_gaussian, m_gaussian3),

        (number_samples_gaussian, n_gaussian1),
        (number_samples_gaussian, n_gaussian2),
        (number_samples_gaussian, n_gaussian3),

        (number_samples_gaussian, m_gaussian_overcomplete1),
        (number_samples_gaussian, m_gaussian_overcomplete2),
        (number_samples_gaussian, m_gaussian_overcomplete3),

        (number_samples_gaussian, n_gaussian_overcomplete1),
        (number_samples_gaussian, n_gaussian_overcomplete2),
        (number_samples_gaussian, n_gaussian_overcomplete3),


        (number_samples_gaussian, m_gaussian1),
        (number_samples_gaussian, m_gaussian2),
        (number_samples_gaussian, m_gaussian3),

        (number_samples_gaussian, m_gaussian_overcomplete1),
        (number_samples_gaussian, m_gaussian_overcomplete2),
        (number_samples_gaussian, m_gaussian_overcomplete3),


        (m_gaussian1, number_samples_gaussian),
        (m_gaussian2, number_samples_gaussian),
        (m_gaussian3, number_samples_gaussian),

        (n_gaussian1, m_gaussian1),
        (n_gaussian2, m_gaussian2),
        (n_gaussian3, m_gaussian3),

        (m_gaussian_overcomplete1, number_samples_gaussian),
        (m_gaussian_overcomplete2, number_samples_gaussian),
        (m_gaussian_overcomplete3, number_samples_gaussian),

        (n_gaussian_overcomplete1 if n_gaussian_overcomplete1 >= m_gaussian_overcomplete1 else m_gaussian_overcomplete1, m_gaussian_overcomplete1),
        (n_gaussian_overcomplete2 if n_gaussian_overcomplete2 >= m_gaussian_overcomplete2 else m_gaussian_overcomplete2, m_gaussian_overcomplete2),
        (n_gaussian_overcomplete3 if n_gaussian_overcomplete3 >= m_gaussian_overcomplete3 else m_gaussian_overcomplete3, m_gaussian_overcomplete3),


        (m_gaussian1, -1),
        (m_gaussian2, -1),
        (m_gaussian3, -1),

        (m_gaussian_overcomplete1, -1),
        (m_gaussian_overcomplete2, -1),
        (m_gaussian_overcomplete3, -1),

        (number_samples_gaussian, m_gaussian1),
        (number_samples_gaussian, m_gaussian2),
        (number_samples_gaussian, m_gaussian3),

        (number_samples_gaussian, m_gaussian_overcomplete1),
        (number_samples_gaussian, m_gaussian_overcomplete2),
        (number_samples_gaussian, m_gaussian_overcomplete3),

        (number_samples_gaussian, -1)

        ), dtype=int)


if save_frame_stubs_gaussian:
    np.savez(dir_stubs + "gaussian",
             probability = 0.5,
             tracked_shapes_gaussian = tracked_shapes_gaussian,
             uniform_ns_m_gaussian1 = uniform_ns_m_gaussian1,
             uniform_ns_m_gaussian2 = uniform_ns_m_gaussian2,
             uniform_ns_m_gaussian3 = uniform_ns_m_gaussian3,

             uniform_ns_n_gaussian1 = uniform_ns_n_gaussian1,
             uniform_ns_n_gaussian2 = uniform_ns_n_gaussian2,
             uniform_ns_n_gaussian3 = uniform_ns_n_gaussian3,

             bernoulli_ns_m_gaussian1 = bernoulli_ns_m_gaussian1,
             bernoulli_ns_m_gaussian2 = bernoulli_ns_m_gaussian2,
             bernoulli_ns_m_gaussian3 = bernoulli_ns_m_gaussian3,

             samples_m_ns_gaussian1 = samples_m_ns_gaussian1,
             samples_m_ns_gaussian2 = samples_m_ns_gaussian2,
             samples_m_ns_gaussian3 = samples_m_ns_gaussian3,

             samples_n_m_gaussian1 = samples_n_m_gaussian1,
             samples_n_m_gaussian2 = samples_n_m_gaussian2,
             samples_n_m_gaussian3 = samples_n_m_gaussian3,

             samples_binary_m_gaussian1 = samples_binary_m_gaussian1,
             samples_binary_m_gaussian2 = samples_binary_m_gaussian2,
             samples_binary_m_gaussian3 = samples_binary_m_gaussian3,

             gaussian_ns_m_gaussian1 = gaussian_ns_m_gaussian1,
             gaussian_ns_m_gaussian2 = gaussian_ns_m_gaussian2,
             gaussian_ns_m_gaussian3 = gaussian_ns_m_gaussian3,


             uniform_ns_m_gaussian_overcomplete1 = uniform_ns_m_gaussian_overcomplete1,
             uniform_ns_m_gaussian_overcomplete2 = uniform_ns_m_gaussian_overcomplete2,
             uniform_ns_m_gaussian_overcomplete3 = uniform_ns_m_gaussian_overcomplete3,

             uniform_ns_n_gaussian_overcomplete1 = uniform_ns_n_gaussian_overcomplete1,
             uniform_ns_n_gaussian_overcomplete2 = uniform_ns_n_gaussian_overcomplete2,
             uniform_ns_n_gaussian_overcomplete3 = uniform_ns_n_gaussian_overcomplete3,

             bernoulli_ns_m_gaussian_overcomplete1 = bernoulli_ns_m_gaussian_overcomplete1,
             bernoulli_ns_m_gaussian_overcomplete2 = bernoulli_ns_m_gaussian_overcomplete2,
             bernoulli_ns_m_gaussian_overcomplete3 = bernoulli_ns_m_gaussian_overcomplete3,

             samples_m_ns_gaussian_overcomplete1 = samples_m_ns_gaussian_overcomplete1,
             samples_m_ns_gaussian_overcomplete2 = samples_m_ns_gaussian_overcomplete2,
             samples_m_ns_gaussian_overcomplete3 = samples_m_ns_gaussian_overcomplete3,

             samples_n_m_gaussian_overcomplete1 = samples_n_m_gaussian_overcomplete1,
             samples_n_m_gaussian_overcomplete2 = samples_n_m_gaussian_overcomplete2,
             samples_n_m_gaussian_overcomplete3 = samples_n_m_gaussian_overcomplete3,

             samples_binary_m_gaussian_overcomplete1 = samples_binary_m_gaussian_overcomplete1,
             samples_binary_m_gaussian_overcomplete2 = samples_binary_m_gaussian_overcomplete2,
             samples_binary_m_gaussian_overcomplete3 = samples_binary_m_gaussian_overcomplete3,

             gaussian_ns_m_gaussian_overcomplete1 = gaussian_ns_m_gaussian_overcomplete1,
             gaussian_ns_m_gaussian_overcomplete2 = gaussian_ns_m_gaussian_overcomplete2,
             gaussian_ns_m_gaussian_overcomplete3 = gaussian_ns_m_gaussian_overcomplete3,

             random_unitary_ns_m_gaussian = random_unitary_ns_m_gaussian
             )



if set_frame_stubs_cauchy:

    np.random.seed(9)
    random.seed(9)

    number_samples_cauchy = 25
    data = load_filter_data(dir_filters, "cauchy_frame")

    # Dictonaries:
    # ------------

    a0_cauchy1_filtered1 = data["a0_cauchy1_filtered1"]
    a0_cauchy2_filtered1 = data["a0_cauchy2_filtered1"]
    a0_cauchy3_filtered1 = data["a0_cauchy3_filtered1"]

    m_cauchy1, n_cauchy1 = a0_cauchy1_filtered1.shape
    m_cauchy2, n_cauchy2 = a0_cauchy2_filtered1.shape
    m_cauchy3, n_cauchy3 = a0_cauchy3_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_cauchy1 = np.random.random(size=(number_samples_cauchy, m_cauchy1))
    uniform_ns_m_cauchy2 = np.random.random(size=(number_samples_cauchy, m_cauchy2))
    uniform_ns_m_cauchy3 = np.random.random(size=(number_samples_cauchy, m_cauchy3))

    uniform_ns_n_cauchy1 = np.random.random(size=(number_samples_cauchy, n_cauchy1))
    uniform_ns_n_cauchy2 = np.random.random(size=(number_samples_cauchy, n_cauchy2))
    uniform_ns_n_cauchy3 = np.random.random(size=(number_samples_cauchy, n_cauchy3))

    probability = 0.5
    bernoulli_ns_m_cauchy1 = np.random.binomial(n=1, p=probability, size=(number_samples_cauchy, m_cauchy1))
    bernoulli_ns_m_cauchy2 = np.random.binomial(n=1, p=probability, size=(number_samples_cauchy, m_cauchy2))
    bernoulli_ns_m_cauchy3 = np.random.binomial(n=1, p=probability, size=(number_samples_cauchy, m_cauchy3))

    samples_m_ns_cauchy1 = random.sample(range(m_cauchy1), number_samples_cauchy)
    samples_m_ns_cauchy2 = random.sample(range(m_cauchy2), number_samples_cauchy)
    samples_m_ns_cauchy3 = random.sample(range(m_cauchy3), number_samples_cauchy)

    samples_n_m_cauchy1 = random.sample(range(n_cauchy1), m_cauchy1)
    samples_n_m_cauchy2 = random.sample(range(n_cauchy2), m_cauchy2)
    samples_n_m_cauchy3 = random.sample(range(n_cauchy3), m_cauchy3)

    samples_binary_m_cauchy1 = np.random.choice([-1,1], size=m_cauchy1)
    samples_binary_m_cauchy2 = np.random.choice([-1,1], size=m_cauchy2)
    samples_binary_m_cauchy3 = np.random.choice([-1,1], size=m_cauchy3)

    gaussian_ns_m_cauchy1 = np.random.randn(number_samples_cauchy, m_cauchy1)
    gaussian_ns_m_cauchy2 = np.random.randn(number_samples_cauchy, m_cauchy2)
    gaussian_ns_m_cauchy3 = np.random.randn(number_samples_cauchy, m_cauchy3)



    # Overcomplete dictonaries:
    # -------------------------

    a0_cauchy_overcomplete1_filtered1 = data["a0_cauchy_overcomplete1_filtered1"]
    a0_cauchy_overcomplete2_filtered1 = data["a0_cauchy_overcomplete2_filtered1"]
    a0_cauchy_overcomplete3_filtered1 = data["a0_cauchy_overcomplete3_filtered1"]

    m_cauchy_overcomplete1, n_cauchy_overcomplete1 = a0_cauchy_overcomplete1_filtered1.shape
    m_cauchy_overcomplete2, n_cauchy_overcomplete2 = a0_cauchy_overcomplete2_filtered1.shape
    m_cauchy_overcomplete3, n_cauchy_overcomplete3 = a0_cauchy_overcomplete3_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_cauchy_overcomplete1 = np.random.random(size=(number_samples_cauchy, m_cauchy_overcomplete1))
    uniform_ns_m_cauchy_overcomplete2 = np.random.random(size=(number_samples_cauchy, m_cauchy_overcomplete2))
    uniform_ns_m_cauchy_overcomplete3 = np.random.random(size=(number_samples_cauchy, m_cauchy_overcomplete3))

    uniform_ns_n_cauchy_overcomplete1 = np.random.random(size=(number_samples_cauchy, n_cauchy_overcomplete1))
    uniform_ns_n_cauchy_overcomplete2 = np.random.random(size=(number_samples_cauchy, n_cauchy_overcomplete2))
    uniform_ns_n_cauchy_overcomplete3 = np.random.random(size=(number_samples_cauchy, n_cauchy_overcomplete3))

    probability = 0.5
    bernoulli_ns_m_cauchy_overcomplete1 = np.random.binomial(n=1, p=probability, size=(number_samples_cauchy, m_cauchy_overcomplete1))
    bernoulli_ns_m_cauchy_overcomplete2 = np.random.binomial(n=1, p=probability, size=(number_samples_cauchy, m_cauchy_overcomplete2))
    bernoulli_ns_m_cauchy_overcomplete3 = np.random.binomial(n=1, p=probability, size=(number_samples_cauchy, m_cauchy_overcomplete3))

    samples_m_ns_cauchy_overcomplete1 = random.sample(range(m_cauchy_overcomplete1), number_samples_cauchy)
    samples_m_ns_cauchy_overcomplete2 = random.sample(range(m_cauchy_overcomplete2), number_samples_cauchy)
    samples_m_ns_cauchy_overcomplete3 = random.sample(range(m_cauchy_overcomplete3), number_samples_cauchy)

    if n_cauchy_overcomplete1 >= m_cauchy_overcomplete1:
        samples_n_m_cauchy_overcomplete1 = random.sample(range(n_cauchy_overcomplete1), m_cauchy_overcomplete1)
    else:
        samples_n_m_cauchy_overcomplete1 = random.sample(range(m_cauchy_overcomplete1), m_cauchy_overcomplete1)

    if n_cauchy_overcomplete2 >= m_cauchy_overcomplete2:
        samples_n_m_cauchy_overcomplete2 = random.sample(range(n_cauchy_overcomplete2), m_cauchy_overcomplete2)
    else:
        samples_n_m_cauchy_overcomplete2 = random.sample(range(m_cauchy_overcomplete2), m_cauchy_overcomplete2)

    if n_cauchy_overcomplete3 >= m_cauchy_overcomplete3:
        samples_n_m_cauchy_overcomplete3 = random.sample(range(n_cauchy_overcomplete3), m_cauchy_overcomplete3)
    else:
        samples_n_m_cauchy_overcomplete3 = random.sample(range(m_cauchy_overcomplete3), m_cauchy_overcomplete3)

    samples_binary_m_cauchy_overcomplete1 = np.random.choice([-1,1], size=m_cauchy_overcomplete1)
    samples_binary_m_cauchy_overcomplete2 = np.random.choice([-1,1], size=m_cauchy_overcomplete2)
    samples_binary_m_cauchy_overcomplete3 = np.random.choice([-1,1], size=m_cauchy_overcomplete3)

    gaussian_ns_m_cauchy_overcomplete1 = np.random.randn(number_samples_cauchy, m_cauchy_overcomplete1)
    gaussian_ns_m_cauchy_overcomplete2 = np.random.randn(number_samples_cauchy, m_cauchy_overcomplete2)
    gaussian_ns_m_cauchy_overcomplete3 = np.random.randn(number_samples_cauchy, m_cauchy_overcomplete3)

    random_unitary_ns_m_cauchy = scipy.stats.unitary_group.rvs(number_samples_cauchy)

    tracked_shapes_cauchy = np.array((
        (number_samples_cauchy, m_cauchy1),
        (number_samples_cauchy, m_cauchy2),
        (number_samples_cauchy, m_cauchy3),

        (number_samples_cauchy, n_cauchy1),
        (number_samples_cauchy, n_cauchy2),
        (number_samples_cauchy, n_cauchy3),

        (number_samples_cauchy, m_cauchy_overcomplete1),
        (number_samples_cauchy, m_cauchy_overcomplete2),
        (number_samples_cauchy, m_cauchy_overcomplete3),

        (number_samples_cauchy, n_cauchy_overcomplete1),
        (number_samples_cauchy, n_cauchy_overcomplete2),
        (number_samples_cauchy, n_cauchy_overcomplete3),


        (number_samples_cauchy, m_cauchy1),
        (number_samples_cauchy, m_cauchy2),
        (number_samples_cauchy, m_cauchy3),

        (number_samples_cauchy, m_cauchy_overcomplete1),
        (number_samples_cauchy, m_cauchy_overcomplete2),
        (number_samples_cauchy, m_cauchy_overcomplete3),


        (m_cauchy1, number_samples_cauchy),
        (m_cauchy2, number_samples_cauchy),
        (m_cauchy3, number_samples_cauchy),

        (n_cauchy1, m_cauchy1),
        (n_cauchy2, m_cauchy2),
        (n_cauchy3, m_cauchy3),

        (m_cauchy_overcomplete1, number_samples_cauchy),
        (m_cauchy_overcomplete2, number_samples_cauchy),
        (m_cauchy_overcomplete3, number_samples_cauchy),

        (n_cauchy_overcomplete1 if n_cauchy_overcomplete1 >= m_cauchy_overcomplete1 else m_cauchy_overcomplete1, m_cauchy_overcomplete1),
        (n_cauchy_overcomplete2 if n_cauchy_overcomplete2 >= m_cauchy_overcomplete2 else m_cauchy_overcomplete2, m_cauchy_overcomplete2),
        (n_cauchy_overcomplete3 if n_cauchy_overcomplete3 >= m_cauchy_overcomplete3 else m_cauchy_overcomplete3, m_cauchy_overcomplete3),


        (m_cauchy1, -1),
        (m_cauchy2, -1),
        (m_cauchy3, -1),

        (m_cauchy_overcomplete1, -1),
        (m_cauchy_overcomplete2, -1),
        (m_cauchy_overcomplete3, -1),

        (number_samples_cauchy, m_cauchy1),
        (number_samples_cauchy, m_cauchy2),
        (number_samples_cauchy, m_cauchy3),

        (number_samples_cauchy, m_cauchy_overcomplete1),
        (number_samples_cauchy, m_cauchy_overcomplete2),
        (number_samples_cauchy, m_cauchy_overcomplete3),

        (number_samples_cauchy, -1)

        ), dtype=int)


if save_frame_stubs_cauchy:
    np.savez(dir_stubs + "cauchy",
             probability = 0.5,
             tracked_shapes_cauchy = tracked_shapes_cauchy,
             uniform_ns_m_cauchy1 = uniform_ns_m_cauchy1,
             uniform_ns_m_cauchy2 = uniform_ns_m_cauchy2,
             uniform_ns_m_cauchy3 = uniform_ns_m_cauchy3,

             uniform_ns_n_cauchy1 = uniform_ns_n_cauchy1,
             uniform_ns_n_cauchy2 = uniform_ns_n_cauchy2,
             uniform_ns_n_cauchy3 = uniform_ns_n_cauchy3,

             bernoulli_ns_m_cauchy1 = bernoulli_ns_m_cauchy1,
             bernoulli_ns_m_cauchy2 = bernoulli_ns_m_cauchy2,
             bernoulli_ns_m_cauchy3 = bernoulli_ns_m_cauchy3,

             samples_m_ns_cauchy1 = samples_m_ns_cauchy1,
             samples_m_ns_cauchy2 = samples_m_ns_cauchy2,
             samples_m_ns_cauchy3 = samples_m_ns_cauchy3,

             samples_n_m_cauchy1 = samples_n_m_cauchy1,
             samples_n_m_cauchy2 = samples_n_m_cauchy2,
             samples_n_m_cauchy3 = samples_n_m_cauchy3,

             samples_binary_m_cauchy1 = samples_binary_m_cauchy1,
             samples_binary_m_cauchy2 = samples_binary_m_cauchy2,
             samples_binary_m_cauchy3 = samples_binary_m_cauchy3,

             gaussian_ns_m_cauchy1 = gaussian_ns_m_cauchy1,
             gaussian_ns_m_cauchy2 = gaussian_ns_m_cauchy2,
             gaussian_ns_m_cauchy3 = gaussian_ns_m_cauchy3,


             uniform_ns_m_cauchy_overcomplete1 = uniform_ns_m_cauchy_overcomplete1,
             uniform_ns_m_cauchy_overcomplete2 = uniform_ns_m_cauchy_overcomplete2,
             uniform_ns_m_cauchy_overcomplete3 = uniform_ns_m_cauchy_overcomplete3,

             uniform_ns_n_cauchy_overcomplete1 = uniform_ns_n_cauchy_overcomplete1,
             uniform_ns_n_cauchy_overcomplete2 = uniform_ns_n_cauchy_overcomplete2,
             uniform_ns_n_cauchy_overcomplete3 = uniform_ns_n_cauchy_overcomplete3,

             bernoulli_ns_m_cauchy_overcomplete1 = bernoulli_ns_m_cauchy_overcomplete1,
             bernoulli_ns_m_cauchy_overcomplete2 = bernoulli_ns_m_cauchy_overcomplete2,
             bernoulli_ns_m_cauchy_overcomplete3 = bernoulli_ns_m_cauchy_overcomplete3,

             samples_m_ns_cauchy_overcomplete1 = samples_m_ns_cauchy_overcomplete1,
             samples_m_ns_cauchy_overcomplete2 = samples_m_ns_cauchy_overcomplete2,
             samples_m_ns_cauchy_overcomplete3 = samples_m_ns_cauchy_overcomplete3,

             samples_n_m_cauchy_overcomplete1 = samples_n_m_cauchy_overcomplete1,
             samples_n_m_cauchy_overcomplete2 = samples_n_m_cauchy_overcomplete2,
             samples_n_m_cauchy_overcomplete3 = samples_n_m_cauchy_overcomplete3,

             samples_binary_m_cauchy_overcomplete1 = samples_binary_m_cauchy_overcomplete1,
             samples_binary_m_cauchy_overcomplete2 = samples_binary_m_cauchy_overcomplete2,
             samples_binary_m_cauchy_overcomplete3 = samples_binary_m_cauchy_overcomplete3,

             gaussian_ns_m_cauchy_overcomplete1 = gaussian_ns_m_cauchy_overcomplete1,
             gaussian_ns_m_cauchy_overcomplete2 = gaussian_ns_m_cauchy_overcomplete2,
             gaussian_ns_m_cauchy_overcomplete3 = gaussian_ns_m_cauchy_overcomplete3,

             random_unitary_ns_m_cauchy = random_unitary_ns_m_cauchy

             )



if set_frame_stubs_fourier:

    np.random.seed(9)
    random.seed(9)

    number_samples_fourier = 25
    data = load_filter_data(dir_filters, "fourier_frame")

    # Dictonaries:
    # ------------

    a0_fourier1_filtered1 = data["a0_fourier1_filtered1"]

    m_fourier1, n_fourier1 = a0_fourier1_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_fourier1 = np.random.random(size=(number_samples_fourier, m_fourier1))

    uniform_ns_n_fourier1 = np.random.random(size=(number_samples_fourier, n_fourier1))

    probability = 0.5
    bernoulli_ns_m_fourier1 = np.random.binomial(n=1, p=probability, size=(number_samples_fourier, m_fourier1))

    samples_m_ns_fourier1 = random.sample(range(m_fourier1), number_samples_fourier)

    samples_n_m_fourier1 = random.sample(range(n_fourier1), m_fourier1)

    samples_binary_m_fourier1 = np.random.choice([-1,1], size=m_fourier1)

    gaussian_ns_m_fourier1 = np.random.randn(number_samples_fourier, m_fourier1)

    random_unitary_ns_m_fourier = scipy.stats.unitary_group.rvs(number_samples_fourier)

    tracked_shapes_fourier = np.array((
        (number_samples_fourier, m_fourier1),

        (number_samples_fourier, n_fourier1),

        (number_samples_fourier, m_fourier1),

        (m_fourier1, number_samples_fourier),

        (n_fourier1, m_fourier1),

        (m_fourier1, -1),

        (number_samples_fourier, m_fourier1),

        (number_samples_fourier, -1)
        ), dtype=int)


if save_frame_stubs_fourier:
    np.savez(dir_stubs + "fourier",
             probability = 0.5,
             tracked_shapes_fourier = tracked_shapes_fourier,
             uniform_ns_m_fourier1 = uniform_ns_m_fourier1,

             uniform_ns_n_fourier1 = uniform_ns_n_fourier1,

             bernoulli_ns_m_fourier1 = bernoulli_ns_m_fourier1,

             samples_m_ns_fourier1 = samples_m_ns_fourier1,

             samples_n_m_fourier1 = samples_n_m_fourier1,

             samples_binary_m_fourier1 = samples_binary_m_fourier1,

             gaussian_ns_m_fourier1 = gaussian_ns_m_fourier1,

             random_unitary_ns_m_fourier = random_unitary_ns_m_fourier
             )
