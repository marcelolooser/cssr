"""
The measurement_matrices module of the cssr package provides classes and functions
for constructing measurement matrices for compressed-sensing-based tasks. The
MeasurementMatrices class allows users to create various types of measurement
matrices that can be used in the compressive sensing porcedure. They are a critical
component of sensing matrices. Currently the class includes measurement matrices
such as random Gaussian, partial Fourier, binary block matrices as well as
optimized measurement matrices, etc.


Created on Wed Mar 17 15:55:09 2021
@author: marcelo looser
"""

import scipy
import random
import numpy as np
import scipy.linalg

# =============================================================================

class MeasurementMatrices:
    """
    The MeasurementMatrices class provides methods for constructing various
    types of measurement matrices used in compressed sensing. These matrices
    are a central component in constructing sensing matrices.

    Parameters
    ----------
    a_tr : array like
        Filtered sparsifying matrix. (Alternatively, the sparsifying matrix
        alone can be used.)
    number_samples : int
        Number of random samples.
    """

    def __init__(self, a_tr, number_samples):

        if not isinstance(a_tr, np.ndarray):
            raise ValueError("The second argument must be an array.")
        elif a_tr.ndim == 1 or (a_tr.ndim == 2 and a_tr.shape[1] == 1):
            raise ValueError("The second argument must be an 2D array of shape "\
                             "(n, m) with m > 1.")

        if not 0 < number_samples <= a_tr.shape[0]:
            raise ValueError("The number samples must be between 0 and "\
                             f"{a_tr.shape[0]}, but {number_samples} was provided.")

        self.a_tr = a_tr
        self.number_samples = number_samples
        self.m, self.n = self.a_tr.shape


    @staticmethod
    def pprint_implemented_mms():
        print("Implemented measurement matrices: ", [
                "random_gauss_matrix",
                "random_bernoulli_matrix",
                "random_partial_fourier_matrix",
                "random_partial_dct_matrix",
                "random_toeplitz_matrix",
                "binary_block",
                "random_sgn_matrix",
                "gdo_measurement_matrix",
                "gdo_measurement_matrix_adaptive",
                "ajs",
                "afms",
                "hblz",
                "ycwg",
                "xsfz"
                ]
            )


    def random_gauss_matrix(self):
        """
        Matrix from randomly chosen values of the gaussian distribution (normalized).

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """

        ar_matrix = np.sqrt(1/self.number_samples) * np.random.randn(self.number_samples, self.m)
        return ar_matrix


    def random_bernoulli_matrix(self, probability=0.1):
        """
        Binary matrix constructed from randomly chosen entries of the bernoulli
        distribution.

        Parameters
        ----------
        probability : float, optional
            Probability (between 0 and 1) of choosing 1 from a bernoulli
            distribution. The default is 0.1.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        ar_matrix = np.random.binomial(n=1, p=probability, size=(self.number_samples, self.m))
        return ar_matrix


    def random_partial_fourier_matrix(self):
        """
        Random selection of rows and columns of a discrete fourier matrix.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """

        indices1 = random.sample(range(self.m), self.number_samples)
        if self.n >= self.m:
            indices2 = random.sample(range(self.n), self.m)
            full_dft = scipy.linalg.dft(self.n)
        else:
            indices2 = random.sample(range(self.m), self.m)
            full_dft = scipy.linalg.dft(self.m)
        ar_matrix = full_dft[:,indices2][indices1,:]
        return ar_matrix


    def random_partial_dct_matrix(self):
        """
        Random selection of rows and columns of a discrete cosine matrix.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """

        indices1 = random.sample(range(self.m), self.number_samples)
        if self.n >= self.m:
            indices2 = random.sample(range(self.n), self.m)
            dct = scipy.fft.dct(np.identity(self.n), axis=0)
        else:
            indices2 = random.sample(range(self.m), self.m)
            dct = scipy.fft.dct(np.identity(self.m), axis=0)
        ar_matrix = dct[:,indices2][indices1,:]
        return ar_matrix


    def random_toeplitz_matrix(self):
        """
        Random toeplitz matrix, constructed via a vector of randomly chosen
        +/- 1 to get a circular matrix.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """

        b = np.random.choice((-1,1), size=self.m)
        indices1 = random.sample(range(self.m), self.number_samples)
        ar_matrix = scipy.linalg.toeplitz(b)[indices1,:]
        return ar_matrix


    def binary_block(self):
        """
        Binary block matrix. The given a_tr is an (M,N) array, constructed by
        inserting blocks of 1s in a staircase fashion into the diagonal entries
        of a zero matrix.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """

        block_length = self.m//self.number_samples
        vec_temp = list(np.ones(block_length)) + list(np.zeros(self.m-block_length))
        ar_matrix = scipy.linalg.circulant(vec_temp)[block_length-1::block_length,:][:self.number_samples,:]
        return ar_matrix


    def random_sgn_matrix(self):
        """
        Random sign matrix, diagonal matrix with randomly chosen values +/- 1.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """

        indices1 = random.sample(range(self.m), self.number_samples)
        item = np.random.choice((-1,1), size=self.m)
        idn = np.zeros((self.m,self.m))
        idn.ravel()[::self.m+1] = item
        ar_matrix = idn[indices1,:]
        return ar_matrix


    def gdo_measurement_matrix(self, mu=None, beta=7e-6, l=30, p=20):
        """
        Gradient-based measurment matrix optimization, see Reference [1].
        This algorithm is effective but takes long to be computed for the want-
        ed percession.

        Parameters
        ----------
        mu : float, optional
            Desired coherence. If None is provided, the sub-optimal Welch bound
            will be used. The default is None.
        beta : float, optional
            Hyperparameter, also referred to as the learning rate. The default
            is 1e-4.
        l : int, optional
            Number of outer loops. The default is 30.
        p : int, optional
            Number of inner loops. The default is 20.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.

        References
        ----------
        .. [1] V. Abolghasemi, S. Ferdowsi, and S. Sanei, "**A gradient-based
           alternating minimization approach for optimization of the measurement
           matrix in compressive sensing**," Signal Processing, vol. 92, no. 4,
           pp. 999–1009, Apr. 2012, doi: 10.1016/j.sigpro.2011.10.012.
        """

        if mu is not None:
            if not isinstance(mu, (float, int)):
                raise TypeError("mu must be a float or integer.")
            elif not (0 <= mu <=1):
                raise ValueError("mu must be a positive float or integer between 0 and 1.")

        if not isinstance(beta, (float, int)):
            raise TypeError("beta must be a float or integer.")
        elif beta < 0:
            raise ValueError("beta must be a positive float or integer.")

        if not (isinstance(l, int) and isinstance(p, int)):
            raise TypeError("l and p must be integers.")
        elif l < 0 or p < 0:
            raise ValueError("l and p must be positive integers.")

        ar_matrix = np.random.random(size=(self.number_samples, self.m))
        mu_opt = self.welch_bound
        if mu is None:
            mu = mu_opt + 1e-3
        else:
            mu = mu_opt + 1e-3 if mu < mu_opt else mu

        for _ in range(l):
            sensing_matrix = ar_matrix.dot(self.a_tr)
            gram = sensing_matrix.T.dot(sensing_matrix)
            gram = mu * np.sign(gram) * (abs(gram) > mu) + gram * (abs(gram) <= mu)
            gram.ravel()[::self.n+1] = 1 # assigning 1 to diagonal elements

            for _ in range(p):
                sensing_matrix = self._normalize_matrix(ar_matrix.dot(self.a_tr))
                ar_matrix = ar_matrix - beta * sensing_matrix.dot((sensing_matrix.T.dot(sensing_matrix) - gram).dot(self.a_tr.T))

        return ar_matrix


    def gdo_measurement_matrix_adaptive(self, mu=None, beta=1e-3, eta=1e-6, l=25, p=12):
        """
        Adaptive gradient-based measurment matrix optimization. See reference [1].
        This is far more intensive to calculate than gdo_measurement_matrix,
        it is recommended to use the former.

        Parameters
        ----------
        mu : float, optional
            Desired coherence. If None is provided, the sub-optimal Welch bound
            will be used. The default is None.
        beta : float, optional
            Hyperparameter, also referred to as the learning rate. This will be
            adapted. The default is 1e-3.
        eta : float, optional
            Hyperparameter, which adjust the learning rate beta incrementally.
            The default is 1e-6.
        l : int, optional
            Number of outer loops. The default is 25.
        p : int, optional
            Number of inner loops. The default is 12.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.

        References
        ----------
        .. [1] V. Abolghasemi, S. Ferdowsi, and S. Sanei, "**A gradient-based
           alternating minimization approach for optimization of the measurement
           matrix in compressive sensing**," Signal Processing, vol. 92, no. 4,
           pp. 999–1009, Apr. 2012, doi: 10.1016/j.sigpro.2011.10.012.
        """

        if mu is not None:
            if not isinstance(mu, (float, int)):
                raise TypeError("mu must be a float or integer.")
            elif not (0 <= mu <=1):
                raise ValueError("mu must be a positive float or integer between 0 and 1.")

        if not isinstance(beta, (float, int)):
            raise TypeError("beta must be a float or integer.")
        elif beta < 0:
            raise ValueError("beta must be a positive float or integer.")

        if not isinstance(eta, (float, int)):
            raise TypeError("eta must be a float or integer.")
        elif beta < 0:
            raise ValueError("eta must be a positive float or integer.")

        if not (isinstance(l, int) and isinstance(p, int)):
            raise TypeError("l and p must be integers.")
        elif l < 0 or p < 0:
            raise ValueError("l and p must be positive integers.")

        self.a_tr = self._normalize_matrix(self.a_tr, 1)
        ar_matrix = np.sqrt(1/self.number_samples) * np.random.randn(self.number_samples, self.m)
        mu_opt = self.welch_bound

        if mu is None:
            mu = mu_opt + 1e-3
        else:
            mu = mu_opt + 1e-3 if mu < mu_opt else mu

        for _ in range(l):
            sensing_matrix = ar_matrix.dot(self.a_tr)
            gram = sensing_matrix.T.dot(sensing_matrix)
            gram = mu * np.sign(gram) * (abs(gram) > mu) + gram * (abs(gram) <= mu)
            gram.ravel()[::self.n+1] = 1 # assigning 1 to diagonal elements

            for _ in range(p):

                sensing_matrix = ar_matrix.dot(self.a_tr)
                sensing_matrix_t = sensing_matrix.T

                a = sensing_matrix_t.dot(sensing_matrix)
                h = sensing_matrix.dot((sensing_matrix_t.dot(sensing_matrix) - gram).dot(self.a_tr.T))
                b = sensing_matrix_t.dot(h.dot(self.a_tr))
                c = self.a_tr.T.dot(h.T.dot(h.dot(self.a_tr)))

                ar_matrix = ar_matrix - beta * h
                beta = beta - eta * (- 2 * np.einsum('ij,ji->', (a - gram).T, (b + b.T)) # computing the trace via einstein summation convention

                                      + 2 * beta * (2*np.einsum('ij,ji->', c, (a - gram))
                                      + np.einsum('ij,ji->', (b + b.T), (b + b.T)))

                                      - 6 * (beta**2) * np.einsum('ij,ji->' , c, (b + b.T))
                                      + 4 * (beta**3) * np.einsum('ij,ji->', c, c)).real
        return ar_matrix


    def ajs(self, eta=0.006, max_iter=50, rtol=1e-6, rtol_estimate=True, **kwargs):
        """
        Gradient decent method for measurement matrix optimization based on a
        randomly chosen values from a gaussian distribution. Slightly modified
        algorithem of the presented one in [1], a normalization in of the
        gamma matrix has been added. Cf. reference [1].

        Parameters
        ----------
        eta : float, optional
            Hyperparameter, also referred to as the learning rate. The default is 0.006.
        max_iter : int, optional
            Maximum number of iterations. The default is 50.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq, where
            eps is the corresponding machine precision of the datatype of a.
            The default is 1e-6.
        rtol_estimate : bool, optional
            If True, the rtol value will be estimated based on the provided noise
            level or signal, which can be passed through **kwargs. If False,
            the default rtol value will be used. The default is True.
        **kwargs :
            Additional key-word arguments to be used for the rtol estimation.
            The keys 'signal' and 'noise_level' can be provided to estimate
            the rtol value based on the signal coherence or noise level.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.

        References
        ----------
        .. [1] V. Abolghasemi, D. Jarchi and S. Sanei, "**A robust approach for
           optimization of the measurement matrix in Compressed Sensing**," 2010
           2nd International Workshop on Cognitive Information Processing, pp.
           388-392, 2010, doi: 10.1109/CIP.2010.5604134.
        """

        if not isinstance(eta, (float, int)):
            raise TypeError("eta must be a float or integer.")
        elif eta < 0:
            raise ValueError("eta must be a positive float or integer.")

        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be a positive integer.")
        elif max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

        self.__validate_rtol(rtol=rtol, rtol_estimate=rtol_estimate)

        ar_matrix = np.sqrt(1/self.number_samples) * self.random_gauss_matrix()

        try:
            eigv, p = scipy.linalg.eig(self.a_tr.dot(self.a_tr.T))
            if np.any(np.isnan(eigv)) or np.any(np.isinf(eigv)):
                raise ValueError("Eigendecomposition produced NaN/Inf")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Eigendecomposition failed: {e}")

        gamma = ar_matrix.dot(p)
        z = np.diag(eigv)
        z = scipy.linalg.pinv(z, rtol=rtol)
        for _ in range(max_iter):
            gamma = self._normalize_matrix(gamma)
            gamma = gamma - eta* gamma.dot(gamma.T.dot(gamma) - z)
        ar_matrix = gamma.dot(p.T)
        return ar_matrix


    def afms(self, eta=0.001, max_iter=30, rtol=5e-2, rtol_estimate=True, **kwargs):
        """
        Gradient decent method for measurement matrix optimization based on a
        randomly chosen values from a gaussian distribution. See reference [1].

        Parameters
        ----------
        eta : float, optional
            Hyperparameter, also referred to as the learning rate. The default is 0.01.
        max_iter : int, optional
            Maximum number of iterations. The default is 10.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq, where
            eps is the corresponding machine precision of the datatype of a.
            The default is 5e-2.
        rtol_estimate : bool, optional
            If True, the rtol value will be estimated based on the provided noise
            level or signal, which can be passed through **kwargs. If False,
            the default rtol value will be used. The default is True.
        **kwargs :
            Additional key-word arguments to be used for the rtol estimation.
            The keys 'signal' and 'noise_level' can be provided to estimate
            the rtol value based on the signal coherence or noise level.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.

        References
        ----------
        .. [1]  V. Abolghasemi, D. Jarchi and S. Sanei, "**A robust approach for
           optimization of the measurement matrix in Compressed Sensing**," 2010
           2nd International Workshop on Cognitive Information Processing, pp.
           388-392, 2010, doi: 10.1109/CIP.2010.5604134.
        """

        if not isinstance(eta, (float, int)):
            raise TypeError("eta must be a float or integer.")
        elif eta < 0:
            raise ValueError("eta must be a positive float or integer.")

        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be a positive integer.")
        elif max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")

        self.__validate_rtol(rtol=rtol, rtol_estimate=rtol_estimate)

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

        d = np.random.random(size=(self.number_samples, self.n))
        idn = np.identity(self.n)
        for _ in range(max_iter):
            d = d - eta * d.dot(d.T.dot(d) - idn)
            d = self._normalize_matrix(d)
        ar_matrix = d.dot(scipy.linalg.pinv(self.a_tr, rtol=rtol))
        ar_matrix = self._normalize_matrix(ar_matrix)
        return ar_matrix


    def hblz(self, mu=None, l=30, p=25, rtol=7e-6, rtol_estimate=True, **kwargs):
        """
        Alternating optimization method for measurement matrix optimization
        based on minimizing the distance between the equivalent dictionary’s
        Gram matrix and a target equiangular tight frame. See reference [1].

        Parameters
        ----------
        mu : float, optional
            Desired coherence. If None is provided, the sub-optimal Welch bound will
            be used. The default is None.
        l : int, optional
            Number of outer loops. The default is 30.
        p : int, optional
            Number of inner loops. The default is 25.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq, where
            eps is the corresponding machine precision of the datatype of a.
            The default is 7e-6.
        rtol_estimate : bool, optional
            If True, the rtol value will be estimated based on the provided noise
            level or signal, which can be passed through **kwargs. If False,
            the default rtol value will be used. The default is True.
        **kwargs :
            Additional key-word arguments to be used for the rtol estimation.
            The keys 'signal' and 'noise_level' can be provided to estimate
            the rtol value based on the signal coherence or noise level.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.

        References
        ----------
        .. [1] T. Hong, H. Bai, S. Li, and Z. Zhu, "**An efficient algorithm for
           designing projection matrix in compressive sensing based on alternating
           optimization**," Signal Processing, vol. 125, pp. 9–20, Aug. 2016,
           doi: 10.1016/j.sigpro.2015.12.015.
        """

        if mu is not None:
            if not isinstance(mu, (float, int)):
                raise TypeError("mu must be a float or integer.")
            elif not (0 <= mu <=1):
                raise ValueError("mu must be a positive float or integer between 0 and 1.")

        if not (isinstance(l, int) and isinstance(p, int)):
            raise TypeError("l and p must be integers.")
        elif l < 0 or p < 0:
            raise ValueError("l and p must be positive integers.")

        self.__validate_rtol(rtol=rtol, rtol_estimate=rtol_estimate)

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

        ar_matrix = np.random.random(size=(self.number_samples, self.m)).astype(complex)
        mu_opt = self.welch_bound

        if mu is None:
            mu = mu_opt + 1e-3
        else:
            mu = mu_opt + 1e-3 if mu < mu_opt else mu

        try:
            u, s, vh = scipy.linalg.svd(self.a_tr, lapack_driver='gesvd')
            if np.max(s) == 0:
                raise ValueError("a_tr matrix is zero matrix")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD failed: {e}")

        sigma = np.diag(s[:self.number_samples])
        for _ in range(l):
            sensing_matrix = ar_matrix.dot(self.a_tr)
            gram = sensing_matrix.T.dot(sensing_matrix)
            gram = mu * np.sign(gram) * (abs(gram) > mu) + gram * (abs(gram) <= mu)
            gram.ravel()[::self.n+1] = 1. # assigning 1 to diagonal elements

            gram = vh.dot(gram.dot(vh.T))
            omega = ar_matrix[:,:self.number_samples].dot(sigma)
            for _ in range(p):
                ej = gram[:self.number_samples, :self.number_samples] - omega.T.dot(omega)

                try:
                    eigv, uj = scipy.linalg.eig(ej)
                    if np.any(np.isnan(eigv)) or np.any(np.isinf(eigv)):
                        raise ValueError("Eigendecomposition produced NaN/Inf")
                except np.linalg.LinAlgError as e:
                    raise ValueError(f"Eigendecomposition failed: {e}")

                binary_temp = (eigv > 0.)
                omega = (uj*np.sqrt(eigv*binary_temp)) + omega*(1 - binary_temp)

            ar_matrix[:, :self.number_samples] = omega.dot(scipy.linalg.pinv(sigma, rtol=rtol))
            ar_matrix = ar_matrix.dot(u.T)
        ar_matrix = self._normalize_matrix(ar_matrix)
        return ar_matrix


    # hard to compute for large matrices, due to the gram matrix
    def ycwg(self, c=0.015, max_iter=30, rtol=4e-2, rtol_estimate=True, **kwargs):
        """
        Alternating minimization method for measurement matrix optimization
        based on minimizing the difference between the sensing matrix’s Gram
        matrix and a target Gram matrix to reduce mutual coherence.
        See reference [1].

        Parameters
        ----------
        c : float, optional
            Hyperparameter that gets added to the Welch coherence bound. It leads to a
            gradual optimization of local minimum determination within the optimization
            procedure. The default is 0.015.
        max_iter : int, optional
            Maximum number of iterations. The default is 30.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq, where
            eps is the corresponding machine precision of the datatype of a.
            The default is 4e-2.
        rtol_estimate : bool, optional
            If True, the rtol value will be estimated based on the provided noise
            level or signal, which can be passed through **kwargs. If False,
            the default rtol value will be used. The default is True.
        **kwargs :
            Additional key-word arguments to be used for the rtol estimation.
            The keys 'signal' and 'noise_level' can be provided to estimate
            the rtol value based on the signal coherence or noise level.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.

        References
        ----------
        .. [1] R. Yi, C. Cui,  B. Wu, and Y. Gong, "**A New Method of Measurement
           Matrix Optimization for Compressed Sensing Based on Alternating
           Minimization**," Mathematics vol. 9, no. 4, Art. no. 329, 2021,
           doi: 10.3390/math9040329.
        """

        if not isinstance(c, (float, int)):
            raise TypeError("c must be a float or integer.")
        elif c < 0:
            raise ValueError("c must be a positive float or integer.")

        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be a positive integer.")
        elif max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")

        self.__validate_rtol(rtol=rtol, rtol_estimate=rtol_estimate)

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

        try:
            u, s, vh = scipy.linalg.svd(self.a_tr, lapack_driver='gesvd')
            if np.max(s) == 0:
                raise ValueError("a_tr matrix is zero matrix")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD failed: {e}")

        ar_matrix = np.random.random(size=(self.number_samples, self.m))
        mu_opt = self.welch_bound

        a = np.zeros((self.number_samples, self.n), dtype=complex)
        b = np.zeros((self.n, self.m), dtype=complex)
        b[:s.shape[0],:s.shape[0]] = scipy.linalg.pinv(np.diag(s), rtol=rtol)

        sigma = vh.T.dot(b.dot(u.T))
        uz = scipy.stats.unitary_group.rvs(self.number_samples) # random unitary matrix

        for _ in range(max_iter):
            sensing_matrix = self._normalize_matrix(ar_matrix.dot(self.a_tr))
            gram = sensing_matrix.T.dot(sensing_matrix)
            gram = (gram * (abs(gram) < mu_opt)
                  + mu_opt * np.sign(gram) * (mu_opt <= abs(gram)) * (abs(gram) <= (mu_opt + c))
                  + (mu_opt + c) * np.sign(gram) * (abs(gram) >= (mu_opt + c)))
            gram.ravel()[::self.n+1] = 1

            try:
                eigv, p = scipy.linalg.eigh(gram)
                if np.any(np.isnan(eigv)) or np.any(np.isinf(eigv)):
                    raise ValueError("Eigendecomposition produced NaN/Inf")
                if np.any(eigv < 0):
                    eigv = np.maximum(eigv, 0)  # Clip negative eigenvalues to 0
            except np.linalg.LinAlgError as e:
                raise ValueError(f"Eigendecomposition failed: {e}")

            lam = np.average(eigv[eigv.argsort()[::-1]][:self.number_samples])
            a.ravel()[::self.n+1] = np.sqrt(lam)

            ar_matrix = uz.dot(a.dot(p.dot(sigma)))
        ar_matrix = self._normalize_matrix(ar_matrix)
        return ar_matrix


    # the pseudo inverse makes this method infeasible for large problems
    # hard to compute for large matrices, due to the gram matrix
    def xsfz(self, mu=None, beta=0.55, max_iter=10, rtol=8e-4, rtol_estimate=True, **kwargs):
        """
        ETF-based iterative minimization method for measurement matrix
        optimization using a Takenaka–Malmquist dictionary to reduce mutual
        coherence between the sensing matrix and sparsifying dictionary.
        See reference [1].

        Parameters
        ----------
        mu : float, optional
            Desired coherence. If None is provided, the sub-optimal Welch bound will
            be used. The default is None.
        beta : float, optional
            Weight, where (1 - beta) is the fraction between old_gram matrix and
            the new gram matrix. The default is 0.55.
        max_iter : int, optional
            Maximum number of iterations. The default is 10.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq, where
            eps is the corresponding machine precision of the datatype of a.
            The default is 8e-4.
        rtol_estimate : bool, optional
            If True, the rtol value will be estimated based on the provided noise
            level or signal, which can be passed through **kwargs. If False,
            the default rtol value will be used. The default is True.
        **kwargs :
            Additional key-word arguments to be used for the rtol estimation.
            The keys 'signal' and 'noise_level' can be provided to estimate
            the rtol value based on the signal coherence or noise level.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.

        References
        ----------
        .. [1] Q. Xu, Z. Sheng, Y. Fang, L. Zhang, "Measurement Matrix Optimization
           for Compressed Sensing System with Constructed Dictionary via
           Takenaka–Malmquist Functions," 21, no. 4: 1229, 2021,
           doi: 10.3390/s21041229
        """

        if not isinstance(beta, (float, int)):
            raise TypeError("beta must be a float or integer.")
        elif beta < 0:
            raise ValueError("beta must be a positive float or integer.")

        if mu is not None:
            if not isinstance(mu, (float, int)):
                raise TypeError("mu must be a float or integer.")
            elif not (0 <= mu <=1):
                raise ValueError("mu must be a positive float or integer between 0 and 1.")

        if not isinstance(max_iter, int):
            raise ValueError("max_iter must be a positive integer.")
        elif max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")

        self.__validate_rtol(rtol=rtol, rtol_estimate=rtol_estimate)

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

        ar_matrix = np.random.random(size=(self.number_samples, self.m))
        mu_opt = self.welch_bound

        if mu is None:
            mu = mu_opt + 1e-3
        else:
            mu = mu_opt + 1e-3 if mu < mu_opt else mu

        gram_old = 0
        for _ in range(max_iter):
            sensing_matrix = self._normalize_matrix((ar_matrix.dot(self.a_tr)))
            gram_t = sensing_matrix.T.dot(sensing_matrix)

            gram_t = mu * np.sign(gram_t) * (abs(gram_t) > mu) + gram_t * (abs(gram_t) <= mu)
            gram_t.ravel()[::self.n+1] = 1 # assigning 1 to diagonal elements
            gram_t = beta * gram_t + (1 - beta) * gram_old
            gram_old = gram_t

            try:
                _, s, vh = scipy.linalg.svd(gram_t, lapack_driver='gesvd')
                if np.max(s) == 0:
                    raise ValueError("a_tr matrix is zero matrix")
            except np.linalg.LinAlgError as e:
                raise ValueError(f"SVD failed: {e}")
            index = s.argsort()[::-1]
            lam = np.average(s[index][:self.number_samples])
            ar_matrix = np.sqrt(lam) * vh[index,:][:self.number_samples, :] @ scipy.linalg.pinv(self.a_tr, rtol=rtol) # rtol: reduces the rank of the matrix by cutting off small singular values
        ar_matrix = self._normalize_matrix(ar_matrix)
        return ar_matrix


    @staticmethod
    def _normalize_matrix(a, norm=2):
        """
        Normalizes all matrix columns.

        Parameters
        ----------
        a : ndarray with shape (N,M)
            Matrix to be normalized.
        norm : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm (see table under Notes). inf means numpy's inf
            object. The default is 2. For further information see doctrings
            numpy linalg.norm.

        Returns
        -------
        Normalized matrix.
        """

        norms = np.linalg.norm(a, ord=norm, axis=0)

        if np.any(np.isnan(norms)) or np.any(np.isinf(norms)): # Check for invalid norms
            norms = np.where(np.isfinite(norms), norms, 1.0)
        result = np.divide(a, norms, out=np.zeros_like(a), where=norms != 0) # Divide with zero protection
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)  # Ensure no NaN columns remain
        return result


    @property
    def welch_bound(self):
        """ Computes the Welch bound."""
        return np.sqrt((self.n - self.number_samples)/(self.number_samples*(self.n-1)))


    def __estimate_rtol_adaptive(self, rtol_default=1e-8, signal=None, noise_level=None):
        """ Smart rtol selection considering full system."""
        try:
            _, s, _ = scipy.linalg.svd(self.a_tr, lapack_driver='gesvd')
            if np.max(s) == 0:
                raise ValueError("a_tr matrix is a zero matrix")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD failed: {e}")

        s_max = np.max(s)
        if s_max < rtol_default:
            rtol_estimate = rtol_default
        else:
            cond_number = s_max / np.min(s)

        if signal is not None: # Use signal coherence
            signal_proj = self.a_tr.T.dot(self.a_tr).dot(signal)
            coherence = np.linalg.norm(signal_proj) / np.linalg.norm(signal)
            rtol_estimate = s_max / (cond_number * (1 + coherence)) # Coherence dependent precision

        elif noise_level is not None: # Use noise level
            rtol_estimate = (9 * noise_level) / s_max # 3-sigma
        else:
            rtol_estimate = 1 / cond_number * 0.1 # Conservative default

        return np.clip(rtol_estimate, rtol_default, 1)


    @staticmethod
    def __validate_rtol(**kwargs):
        """
        Validate common parameters across methods.

        Raises
        ------
        ValueError
            If the input parameters are not valid.
        """

        for key, val in kwargs.items():
            if key == "rtol":
                if val > 1:
                    print(f"Warning: {key}={val} is > 1, all singular values are "\
                          "discarded.{key} will be set to 1.")
                    key = 1
                elif val < 0:
                    print(f"Warning: {key}={val} is < 0, all singular values are "\
                          "retained. {key} will be set to 0.")
                    key = 0
            elif key == "rtol_estimate":
                if not isinstance(val, bool):
                    raise ValueError(f"{key} must be a boolean.")


# =============================================================================
