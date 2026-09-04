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


set_frame_stubs_gaussian = True
save_frame_stubs_gaussian = True


# =============================================================================
# Main:
# =============================================================================

if set_frame_stubs_gaussian:

    np.random.seed(9)
    random.seed(9)

    number_samples_gaussian = 25
    data = load_filter_data(dir_filters, "gaussian_frame")

    # Dictonaries:
    # ------------

    a0_gaussian2_filtered1 = data["a0_gaussian2_filtered1"]

    m_gaussian2, n_gaussian2 = a0_gaussian2_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_gaussian2 = np.random.random(size=(number_samples_gaussian, m_gaussian2))

    uniform_ns_n_gaussian2 = np.random.random(size=(number_samples_gaussian, n_gaussian2))

    probability = 0.5
    bernoulli_ns_m_gaussian2 = np.random.binomial(n=1, p=probability, size=(number_samples_gaussian, m_gaussian2))

    samples_m_ns_gaussian2 = random.sample(range(m_gaussian2), number_samples_gaussian)

    samples_n_m_gaussian2 = random.sample(range(n_gaussian2), m_gaussian2)

    samples_binary_m_gaussian2 = np.random.choice([-1,1], size=m_gaussian2)

    gaussian_ns_m_gaussian2 = np.random.randn(number_samples_gaussian, m_gaussian2)


    # Overcomplete dictonaries:
    # -------------------------

    a0_gaussian_overcomplete2_filtered1 = data["a0_gaussian_overcomplete2_filtered1"]

    m_gaussian_overcomplete2, n_gaussian_overcomplete2 = a0_gaussian_overcomplete2_filtered1.shape


    # Stubbing arrays:
    # ................

    uniform_ns_m_gaussian_overcomplete2 = np.random.random(size=(number_samples_gaussian, m_gaussian_overcomplete2))

    uniform_ns_n_gaussian_overcomplete2 = np.random.random(size=(number_samples_gaussian, n_gaussian_overcomplete2))

    probability = 0.5
    bernoulli_ns_m_gaussian_overcomplete2 = np.random.binomial(n=1, p=probability, size=(number_samples_gaussian, m_gaussian_overcomplete2))

    samples_m_ns_gaussian_overcomplete2 = random.sample(range(m_gaussian_overcomplete2), number_samples_gaussian)

    if n_gaussian_overcomplete2 >= m_gaussian_overcomplete2:
        samples_n_m_gaussian_overcomplete2 = random.sample(range(n_gaussian_overcomplete2), m_gaussian_overcomplete2)
    else:
        samples_n_m_gaussian_overcomplete2 = random.sample(range(m_gaussian_overcomplete2), m_gaussian_overcomplete2)

    samples_binary_m_gaussian_overcomplete2 = np.random.choice([-1,1], size=m_gaussian_overcomplete2)

    gaussian_ns_m_gaussian_overcomplete2 = np.random.randn(number_samples_gaussian, m_gaussian_overcomplete2)

    random_unitary_ns_m_gaussian = scipy.stats.unitary_group.rvs(number_samples_gaussian)

    tracked_shapes_gaussian = np.array((
        (number_samples_gaussian, m_gaussian2),

        (number_samples_gaussian, n_gaussian2),

        (number_samples_gaussian, m_gaussian_overcomplete2),

        (number_samples_gaussian, n_gaussian_overcomplete2),


        (number_samples_gaussian, m_gaussian2),

        (number_samples_gaussian, m_gaussian_overcomplete2),


        (m_gaussian2, number_samples_gaussian),

        (n_gaussian2, m_gaussian2),

        (m_gaussian_overcomplete2, number_samples_gaussian),

        (n_gaussian_overcomplete2 if n_gaussian_overcomplete2 >= m_gaussian_overcomplete2 else m_gaussian_overcomplete2, m_gaussian_overcomplete2),


        (m_gaussian2, -1),

        (m_gaussian_overcomplete2, -1),

        (number_samples_gaussian, m_gaussian2),

        (number_samples_gaussian, m_gaussian_overcomplete2),

        (number_samples_gaussian, -1)

        ), dtype=int)


if save_frame_stubs_gaussian:
    np.savez(dir_stubs + "gaussian",
             probability = 0.5,
             tracked_shapes_gaussian = tracked_shapes_gaussian,
             uniform_ns_m_gaussian2 = uniform_ns_m_gaussian2,

             uniform_ns_n_gaussian2 = uniform_ns_n_gaussian2,

             bernoulli_ns_m_gaussian2 = bernoulli_ns_m_gaussian2,

             samples_m_ns_gaussian2 = samples_m_ns_gaussian2,

             samples_n_m_gaussian2 = samples_n_m_gaussian2,

             samples_binary_m_gaussian2 = samples_binary_m_gaussian2,

             gaussian_ns_m_gaussian2 = gaussian_ns_m_gaussian2,


             uniform_ns_m_gaussian_overcomplete2 = uniform_ns_m_gaussian_overcomplete2,

             uniform_ns_n_gaussian_overcomplete2 = uniform_ns_n_gaussian_overcomplete2,

             bernoulli_ns_m_gaussian_overcomplete2 = bernoulli_ns_m_gaussian_overcomplete2,

             samples_m_ns_gaussian_overcomplete2 = samples_m_ns_gaussian_overcomplete2,

             samples_n_m_gaussian_overcomplete2 = samples_n_m_gaussian_overcomplete2,

             samples_binary_m_gaussian_overcomplete2 = samples_binary_m_gaussian_overcomplete2,

             gaussian_ns_m_gaussian_overcomplete2 = gaussian_ns_m_gaussian_overcomplete2,

             random_unitary_ns_m_gaussian = random_unitary_ns_m_gaussian
             )


# =============================================================================