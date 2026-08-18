"""
The measurementMatrices module of the cssr package provides classes and functions
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
    """Construction of the measurment matrix, this matrix will be multiplied
    by the a_tr matrix which gives the sensing matrix for compressed sensing.

    Parameters
    ----------
    number_samples : int
        Number of random samples.
    a_tr : array like
        Filtered  and truncated sparse matrix.
    """

    def __init__(self, number_samples, a_tr):
        self.number_samples = number_samples
        self.a_tr = a_tr
        self.m, self.n = self.a_tr.shape

        self.__validate_parameters()


    @staticmethod
    def pprint_implemented_mm():
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
        Randomly chosen values from a gaussian distribution, normalized.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """
        ar_matrix = np.sqrt(1/self.number_samples) * np.random.randn(self.number_samples, self.m)
        return ar_matrix


    def random_bernoulli_matrix(self, probability=0.1):
        """
        Randomly chosen values from a bernoulli distribution, 1, 0.

        Parameters
        ----------
        probability : float, optional
            Parameter of the distribution, >= 0 and <=1. The default is 0.01.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """
        ar_matrix = np.random.binomial(n=1, p=probability, size=(self.number_samples, self.m))
        return ar_matrix


    def random_partial_fourier_matrix(self):
        """ Random selection of rows and columns of a descrete fourier matrix."""
        indices1 = random.sample(range(self.m), self.number_samples)
        indices2 = random.sample(range(self.n), self.m)
        full_dft = scipy.linalg.dft(self.n)
        ar_matrix = full_dft[:,indices2][indices1,:]
        return ar_matrix


    def random_partial_dct_matrix(self):
        """ Random selection of rows and columns of a descrete cosine matrix."""
        indices1 = random.sample(range(self.m), self.number_samples)
        indices2 = random.sample(range(self.n), self.m)
        dct = scipy.fft.dct(np.identity(self.n), axis=0)
        ar_matrix = dct[:,indices2][indices1,:]
        return ar_matrix


    def random_toeplitz_matrix(self):
        """ Random toeplitz matrix. Constructed via a vector with random +/- 1
        as values, which gets used as the bulding block of a circular matrix."""
        b = np.random.choice([-1,1], size=self.m)
        indices1 = random.sample(range(self.m), self.number_samples)
        ar_matrix = scipy.linalg.toeplitz(b)[indices1,:]
        return ar_matrix


    def binary_block(self):
        """ Binary block matrix. The given a_tr is an (M,N) array, ...."""
        block_length = self.m//self.number_samples
        vec_temp = list(np.ones(block_length)) + list(np.zeros(self.m-block_length))
        ar_matrix = scipy.linalg.circulant(vec_temp)[block_length-1::block_length,:][:self.number_samples,:]
        return ar_matrix


    def random_sgn_matrix(self):
        """Random sign matrix, diagonal matrix with random values of -1, 1.

        Returns
        -------
        ar_matrix : ndarray
            Measurement matrix.
        """
        indices1 = random.sample(range(self.m), self.number_samples)
        item = np.random.choice([-1,1], size=self.m)
        idn = np.zeros((self.m,self.m))
        idn.ravel()[::self.m+1] = item
        ar_matrix = idn[indices1,:]
        return ar_matrix


    # there might be an issue with the normalization
    # hard to compute for large matrices, due to the gram matrix
    def gdo_measurement_matrix(self, mu=None, beta=7e-6, l=30, p=20):
        """
        Gradient-based measurment matrix optimization, see Reference [1].
        This algorithm is effective but takes long to be computed for the want-
        ed percession.

        Parameters
        ----------
        mu : float, optional
            Wanted coherence, if None is provided the sub optimal welch bound will
            be used. The default is None.
        beta : float, optional
            Hyperparameter. The default is 1e-4.
        l : int, optional
            Number of loops, outer loop. Default is 25.
        p : int, optional
            Number of loops, inner loop. Default is 12.

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
        ar_matrix = np.random.random(size=(self.number_samples, self.m))
        mu_opt = self._welch_bound()
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


    # inspect problem with normalization.
    # hard to compute for large matrices, due to the gram matrix
    def gdo_measurement_matrix_adaptive(self, mu=None, beta=1e-3, l=25, p=12):
        """
        Adaptive gradient-based measurment matrix optimization. See reference [1].
        This is far more intensive to calculate than gdo_measurement_matrix,
        it is recommended to use the former.

        Parameters
        ----------
        mu : float, optional
            Wanted coherence, if None is provided the sub optimal welch bound will
            be used. The default is None.
        beta : float, optional
            Initial hyperparameter, this will be adapted. Default is 1e-3.
        l : int, optional
            Number of loops, outer loop. Default is 25.
        p : int, optional
            Number of loops, inner loop. Default is 12.

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
        self.a_tr = self._normalize_matrix(self.a_tr, 1)
        ar_matrix = np.sqrt(1/self.number_samples) * np.random.randn(self.number_samples, self.m)
        mu_opt = self._welch_bound()

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
                beta = beta - 1e-5 * (- 2 * np.einsum('ij,ji->', (a - gram).T, (b + b.T)) # computing the trace via einstein summation convention

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
            Hyperparameter. The default is 0.006.
        max_iter : int, optional
            Maximal number of iterations. The default is 50.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq where
            eps is the corresponding machine precision value of the datatype of a.
            The default is 1e-6.

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

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

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
            Hyperparameter. The default is 0.01.
        max_iter : int, optional
            Maximal number of iterations. The default is 10.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq where
            eps is the corresponding machine precision value of the datatype of a.
            The default is 5e-4.

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
        mu : float
            Wanted coherence, if None is provided the sub optimal welch bound will
            be used. The default is None.
        l : int, optional
            Number of loops, outer loop. Default is 25.
        p : int, optional
            Number of loops, inner loop. Default is 12.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq where
            eps is the corresponding machine precision value of the datatype of a.
            The default is 5e-4.

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

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

        ar_matrix = np.random.random(size=(self.number_samples, self.m)).astype(complex)
        mu_opt = self._welch_bound()

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
    # the probelm lies within the singular value decomposition for small eigenvalues the construction diverges
    def ycwg(self, c=0.015, max_iter=30, rtol=4e-2, rtol_estimate=True, **kwargs):
        """
        Alternating minimization method for measurement matrix optimization
        based on minimizing the difference between the sensing matrix’s Gram
        matrix and a target Gram matrix to reduce mutual coherence.
        See reference [1].

        Parameters
        ----------
        c : float, optional
            Parameter which gets added to the welch coherence bound, which gives
            leads to a better local minimum in the optimization. The default is
            0.01.
        max_iter : int, optional
            Maximal number of iterations. The default is 20.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq where
            eps is the corresponding machine precision value of the datatype of a.
            I recomend setting rtol to 0.0 for signal to noise ratios above 1000,
            in this way, way sparser solutions can be found !!!
            The default is 5e-4.

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

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

        try:
            u, s, vh = scipy.linalg.svd(self.a_tr, lapack_driver='gesvd')
            if np.max(s) == 0:
                raise ValueError("a_tr matrix is zero matrix")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD failed: {e}")

        ar_matrix = np.random.random(size=(self.number_samples, self.m))
        mu_opt = self._welch_bound()

        a = np.zeros((self.number_samples, self.n), dtype=complex)
        b = np.zeros((self.n, self.m), dtype=complex)
        b[:self.m,:self.m] = scipy.linalg.pinv(np.diag(s), rtol=rtol) # cuttiing of too small singular values, which lead to

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
    # and seems to performs badly in tests with noise
    # hard to compute for large matrices, due to the gram matrix
    def xsfz(self, mu=None, beta=0.55, max_iter=10, rtol=8e-4, rtol_estimate=True, **kwargs):
        """
        ETF-based iterative minimization method for measurement matrix
        optimization using a Takenaka–Malmquist dictionary to reduce mutual
        coherence between the sensing matrix and sparsifying dictionary.
        See reference [1].

        Parameters
        ----------
        mu : float
            Wanted coherence, if None is provided the sub optimal welch bound will
            be used. The default is None.
        beta : float, optional
            Weight, where (1 - beta) is the fraction between old_gram matrix and
            the new gram matrix. The default is 0.55.
        max_iter : int, optional
            Maximal number of iterations. The default is 10.
        rtol : float, optional
            Cutoff factor for 'small' singular values. In lstsq, singular values
            less than rtol*largest_singular_value will be considered as zero.
            If 0., the default value max(M, N) * eps is passed to lstsq where
            eps is the corresponding machine precision value of the datatype of a.
            I recomend setting rtol to 0.0 for signal to noise ratios above 1000,
            in this way, way sparser solutions can be found !!!
            The default is 8e-4.

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

        if rtol_estimate:
            rtol = self.__estimate_rtol_adaptive(rtol_default=rtol, **kwargs)

        ar_matrix = np.random.random(size=(self.number_samples, self.m))
        mu_opt = self._welch_bound()

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
                u, s, vh = scipy.linalg.svd(self.a_tr, lapack_driver='gesvd')
                if np.max(s) == 0:
                    raise ValueError("a_tr matrix is zero matrix")
            except np.linalg.LinAlgError as e:
                raise ValueError(f"SVD failed: {e}")
            index = s.argsort()[::-1]
            lam = np.average(s[index][:self.number_samples])
            ar_matrix = np.sqrt(lam) * vh[index,:][:self.number_samples, :] @ scipy.linalg.pinv(self.a_tr, rtol=rtol) # rtol: reduces the rank of the matrix by cuting off small singular values
        ar_matrix = self._normalize_matrix(ar_matrix)
        return ar_matrix


    @staticmethod
    def _normalize_matrix(a, norm=2):
        """ Normalizes all matrix columns.

        Parameters
        ----------
        a : ndarray with shape (N,M)
            Matrix to be normalized.
        norm : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm (see table under Notes). inf means numpy's inf object. The default is 2.
            For further information see doctrings numpy linalg.norm.

        Returns
        -------
        Normalized matrix.
        """

        norms = np.linalg.norm(a, ord=norm, axis=0)

        if np.any(np.isnan(norms)) or np.any(np.isinf(norms)): # Check for invalid norms
            norms = np.where(np.isfinite(norms), norms, 1.0)
        result = np.divide(a, norms, out=np.zeros_like(a, dtype=complex), where=norms != 0) # Divide with zero protection
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)  # Ensure no NaN columns remain
        return result

        norms = np.linalg.norm(a, ord=norm, axis=0)
        return np.divide(a, norms, out=np.zeros_like(a, dtype=complex), where=norms != 0)


    def _welch_bound(self):
        """
        Computes the Welch bound.
        """
        return np.sqrt((self.n - self.number_samples)/(self.number_samples*(self.n-1)))


    def __estimate_rtol_adaptive(self, rtol_default=1e-8, signal=None, noise_std=None):
        """
        Smart rtol selection considering full system.
        """
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

        elif noise_std is not None: # Use noise level
            rtol_estimate = (9 * noise_std) / s_max # 3-sigma
        else:
            rtol_estimate = 1 / cond_number * 0.1 # Conservative default

        return np.clip(rtol_estimate, rtol_default, 1)


    def __validate_parameters(self, **kwargs):
        """Validate common parameters across methods."""
        if self.number_samples <= 0 or self.number_samples > self.a_tr.shape[0]:
            raise ValueError(f"The number samples must be between 0 and {self.a_tr.shape[0]}, but {self.number_samples} was provided.")

        for key, val in kwargs.items():
            if key in ["mu", "eta", "beta", "probability"]:
                if not (0 < val < 1):
                    print(f"Warning: {key}={val} outside (0,1), may cause convergence issues")
            elif key == 'max_iter':
                if val <= 0:
                    raise ValueError(f"{key} must be positive")
            elif key == "rtol":
                if val < 0:
                    raise ValueError(f"{key} must be non-negative")



# =============================================================================
