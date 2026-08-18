"""
The superresolvers module of the cssr package provides classes and functions
for reconstructing signals via compressive sensing which admit a sparse
representation in some domain and have been corrupted by a linear operation
such as a low-pass filter, and noise. The Superresolvers class allows users
to perform superresolution on given data using various algorithms, including
basis pursuit and matching pursuit based algorithms, (see references in the
documentation). In particuar, the class includes a decorator intended to improve
the performance of the superresolution algorithms by using a matching pursuit
assisted LASSO algorithm, as described in the reference provided in the
documentation.


Created on Fri Feb 12 19:37:54 2021
@author: marcelo looser
"""

import cvxpy
import scipy
import numpy as np
from functools import wraps

# =============================================================================

def _boost_superresolver(func):
    """ Decorator for better (or at least as good) and more effi-
    cient superresolution of signals via matching pursuit assisted LASSO algo-
    rithms. See reference [1].

    Parameters
    ----------
    func : callable
        LASSO based l1 sparse solver.
    r : float, optional
        The booster can be deactivated by setting r to zero.
        The default is 20.
    max_iter : int, optional
        Maximal number of iterations. The default is 20.

    Reference
    ---------
    .. [1] M. Tan, I. W. Tsang and L. Wang, "**Matching Pursuit LASSO Part I:
       Sparse Recovery Over Big Dictionary**," in IEEE Transactions on Signal
       Processing, vol. 63, no. 3, pp. 727-741, Feb.1, 2015,
       doi: 10.1109/TSP.2014.2385036.
    """
    @wraps(func)
    def inner(self, *args, eta=0.8, rho=None, max_iter=30, atol=1e-8, **kwargs):
        """
        Parameters
        ----------
        *args :
            Additional arguments to be passed to the superresolver.
        eta : float, optional
            Hyperparameter to guess the hyperparameter rho, The default is 0.8.
        rho : float, optional
            Hyperparameter to mitigate "overfitting". The default is None.
        max_iter :  int, optional
            Maximal number of iterations. The default is 30.
        atol : float, optional
            Absolute tolerance. The default is 1e-5.
        **kwargs :
            Additional key-word arguments to be passed to the superresolver.
        """
        if rho:
            pass
        else:
            temp1 = (self.a.T.dot(self.ar.dot(args[0].reshape((-1,1))))).ravel()
            rho = np.count_nonzero(temp1 >= eta* np.linalg.norm(temp1, np.inf))

        if rho == 0: # perform the unboosted reconstruction
            y_hat, y0 = func(self, *args, **kwargs)

        else:
            alpha0 = self.ar.dot(args[0].reshape((-1,1)).copy())
            alpha = alpha0.copy()
            obj_fun = (0.5 * np.linalg.norm(alpha, "fro")**2 - alpha.T.dot(alpha))[0,0] # objective function of the dual problem
            y0 = np.zeros((self.n, alpha.shape[1]), dtype=complex)

            s, sc = [], list(range(self.n)) # initiate active and complementary set
            for _ in range(max_iter):

                # conduct worst case analysis for most active atoms in dictionary
                g = self.a.T.dot(alpha)
                add_s = list(abs(g).ravel().argsort()[::-1]) # the biggest values indicate the most active atoms in the dictionary
                add_s = self._reduce(add_s, s, stop_cond=rho)
                s.extend(add_s)
                sc = self._reduce(sc, s)
                y0[sc,:] = 0

                y_hat, y_sparse_hat = func(self, *args, **kwargs, y0=y0[s,:], support=s)  # perform superresolution
                y0[s,:] = y_sparse_hat

                alpha = alpha0 - self.a.dot(y0)
                obj_fun_new = (0.5 * np.linalg.norm(alpha, "fro")**2 - alpha.T.dot(alpha0))[0,0] # objective function of the dual problem
                stop_iteration = abs(obj_fun - obj_fun_new) / (rho * np.linalg.norm(alpha0, "fro"))
                obj_fun = obj_fun_new

                if stop_iteration <= atol:
                    break

        return y_hat, y0
    return inner

# =============================================================================

class Superresolvers:
    """ Perform superresolutio on given data.

    Parameters
    ----------
    a0 : array like
        Sparsifying basis.
    a_tr : array like
        Low pass filtered sparsifying matrix.
    ar : array like
        Measurement matrix.
    """

    def __init__(self, a0, a_tr, ar):
        self.a0 = a0
        self.n = a0.shape[1] # number of columns
        self.m = ar.shape[0]

        self.ar = ar
        self.a_tr = a_tr
        self.a = ar.dot(a_tr)


    @staticmethod
    def pprint_implemented_superresolvers():
        print("Implemented superreolvers: ", [
                  "bp",
                  "bpd",
                  "ic",
                  "nlht",
                  "nlht_lasso",
                  ]
            )


    def bp(self, y_signal,  solver="CLARABEL"):
        """ Basis pursuit. See reference [1].

        Parameters
        ----------
        x_signal : array like
            X component of signal.
        y_signal : array like
            Signal.
        solver : str, optional
            Convex solver used in the package cvxpy. The default is "CLARABEL".

        Returns
        -------
        y_hat : array like
            Reconstructed array.
        y_sparse_hat : array like
            Sparse representation of reconstructed array (in general more di-
            mensions than reconstructed array, with shape (a0.shape[1], y_signal.shape[1])).

        References
        ----------
        .. [1] S. S. Chen, D. L. Donoho, and M. A. Saunders, "**Atomic
           decomposition by basis pursuit**," SIAM Review, vol. 43, no. 1,
           pp. 129–159, 2001, doi: 10.1137/S003614450037906X
        """
        y_t = y_signal.reshape((-1,1))
        signal_length, axes3d = y_t.shape

        b = self.ar.dot(y_t)                               # random measured signal
        y0 = np.dot(self.a.T, b)                                # initial vector

        vx = cvxpy.Variable((self.n, axes3d), complex=True)
        vx.value = y0                                           # assigning initial vector to the vx

        objective = cvxpy.Minimize(cvxpy.norm(vx, 1))
        constraints = [self.a @ vx == b]
        prob = cvxpy.Problem(objective, constraints)
        prob.solve(solver=solver)

        y_sparse_hat = vx.value .real                           # recovered sparse signal
        y_hat = self.a0.dot(y_sparse_hat).real
        return y_hat, y_sparse_hat


    def bpd(self, y_signal, noise_level, solver="CLARABEL"):
        """Basis pursuit denoising. See reference [1].

        Parameters
        ----------
        x_signal : array like
            X component of signal.
        y_signal : array like
            Signal.
        noise_level : float
            Magnitude of estimated noise.
        solver : str, optional
            Convex solver used in the package cvxpy. The default is "CLARABEL".

        Returns
        -------
        y_hat : array like
            Reconstructed array.
        y_sparse_hat : array like
            Sparse representation of reconstructed array (in general more di-
            mensions than reconstructed array, with shape (a0.shape[1], y_signal.shape[1])).

        References
        ----------
        .. [1] S. S. Chen, D. L. Donoho, and M. A. Saunders, "**Atomic
           decomposition by basis pursuit**," SIAM Review, vol. 43, no. 1,
           pp. 129–159, 2001, doi: 10.1137/S003614450037906X

        """
        y_t = y_signal.reshape((-1,1))
        signal_length, axes3d = y_t.shape

        b = self.ar.dot(y_t)                               # random measured signal
        y0 = np.dot(self.a.T, b)                                # initial vector

        vx = cvxpy.Variable((self.n, axes3d), complex=True)
        vx.value = y0                                           # assigning initial vector to the vx

        objective = cvxpy.Minimize(cvxpy.norm(vx, 1))
        constraints = [cvxpy.sum_squares(self.a @ vx - b) <= noise_level]
        prob = cvxpy.Problem(objective, constraints)
        prob.solve(solver=solver)

        y_sparse_hat = vx.value.real                            # recovered sparse signal
        y_hat = self.a0.real.dot(y_sparse_hat)
        return y_hat, y_sparse_hat


    @_boost_superresolver  # slightly modified ic-algorithm  to fit boost_superresolver
    def ic(self, y_signal, lam=None, l=25, max_iter=20, y0=None, support=None, solver="CLARABEL"):
        """Basis pursuit denoising, via in-crowd method. See reference [1].

        Parameters
        ----------
        x_signal : array like
            X component of signal.
        y_signal : array like
            Signal.
        lam : float, optional
            Parameter for the LASSO optimization. If None lam is 5% of the ma-
            gnitude of a_tr.T*y_signal. The default is None.
        l : int, optional
            Maximal number of items to be included to the active set of atoms,
            (the active set grows over each iteration). The default is 25.
        max_iter : int optional
            Maximal number of iterations. The default is 20.
        solver : str, optional
            Convex solver used in the package cvxpy. The default is "CLARABEL".

        Returns
        -------
        y_hat : array like
            Reconstructed array.
        y_sparse_hat : array like
            Sparse representation of reconstructed array (in general more di-
            mensions than reconstructed array, with shape (a0.shape[1], y_signal.shape[1])).

        Refernces
        ---------
        .. [1] P. R. Gill, A. Wang and A. Molnar, "**The In-Crowd Algorithm for
           Fast Basis Pursuit Denoising**," in IEEE Transactions on Signal
           Processing, vol. 59, no. 10, pp. 4595-4605, Oct. 2011,
           doi: 10.1109/TSP.2011.2161292.
        """

        y_t = y_signal.reshape((-1,1))
        signal_length, axes3d = y_t.shape
        lam = 0.007 * np.linalg.norm(self.a_tr.T.dot(y_t).ravel(), np.inf) + 1e-6 if lam is None else lam

        if support is None:
            support = range(self.n)

        b = self.ar.dot(y_t)                                        # random measured signal
        y0 = np.zeros((self.n, axes3d))[support] if y0 is None else y0 # initial vector (proxy vor sparse vector)
        vx = cvxpy.Variable((y0.shape[0], axes3d), complex=True)
        vx.value = y0                                           # assigning initial vector to the vx, rough guess

        s, sc = [], list(range(y0.shape[0]))                        # active set and complementary set
        loop_count = 0
        while loop_count < max_iter:

            res = (b - self.a[:,support] @ vx.value).T if not len(s) else (b - self.a[:,support][:,s] @ vx.value[s,:]).T
            usefulness = abs(res @ self.a[:,support]).ravel()[sc]
            usefulness = usefulness * (usefulness > lam)
            length_usefulness = np.count_nonzero(usefulness > 0.)

            if not length_usefulness:
                break
            else:
                max_items = l * (length_usefulness >= l) + length_usefulness * (length_usefulness < l) # needed if the len(usefulness) is smaller than l
                indices_shifted = np.argpartition(usefulness, -max_items)[-max_items:] # searches for the max_items biggest values in usefulness
                indices = np.array(sc)[indices_shifted]
                s = list(set(s + list(indices)))
                sc = self._reduce(sc, s)

                objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(self.a[:,support][:,s] @ vx[s,:] - b) + lam * cvxpy.norm(vx[s,:], 1)) # lagrangian method
                prob = cvxpy.Problem(objective)
                prob.solve(solver=solver)

                add_off_support_indices = [i for i, item in enumerate(abs(vx.value[s,:]) <= 0.) if item[0]]
                s = self._reduce(s, add_off_support_indices)
                sc = list(set(sc + add_off_support_indices))
                vx.value[sc] = 0
                loop_count += 1

        y_sparse_hat = vx.value                            # recovered sparse signal
        y_hat = self.a0[:,support][:,s].dot(y_sparse_hat[s,:])
        return y_hat, y_sparse_hat


    # basis pursuit denoising resp. spectral projected gradient for L1 minimization
    # (SPGL1) + off-support search
    def nlht(self, y_signal, noise_level, nnw=9, zeta0=0.025, dzeta=0.025, y0=None, support=None, solver="CLARABEL"):
        """Basis pursuit denoising via non-local hard threshholding. See reference [1].

        Parameters
        ----------
        x_signal : array like
            X component of signal.
        y_signal : array like
            Signal.
        noise_level : float
            Magnitude of estimated noise.
        nnw : int, optional
            nnw neatrest neighbours to be searched for off support addition.
            The default is 9.
        zeta0 : float, optional
            Fraction of max(magnitude) of the reconstructed vector in the sparse
            representation. Is needed as a measure of noise consideration.
            The default is 0.025.
        dzeta : float, optional
            Step size for increasing zeta0 over each iteration.
            The default is 0.025.
        solver : str, optional
            Convex solver used in the package cvxpy. The default is "CLARABEL".

        Returns
        -------
        y_hat : array like
            Reconstructed array.
        y_sparse_hat : array like
            Sparse representation of reconstructed array (in general more di-
            mensions than reconstructed array, with shape (a0.shape[1], y_signal.shape[1])).

        References
        ----------
        .. [1] S. Gazit, A. Szameit, Y. C. Eldar, and M. Segev, “**Super-resolution
           and reconstruction of sparse sub-wavelength images**,” Optics Express,
           vol. 17, no. 26, pp. 23920–23946, 2009, doi: 10.1364/OE.17.023920.
        """
        y_t = y_signal.reshape((-1,1))
        signal_length, axes3d = y_t.shape

        if support is None:
            support = range(self.n)

        b = self.ar.dot(y_t) # random measured signal
        y0 = np.dot(self.a.conj().T, b)[support] if y0 is None else y0 # initial vector (proxy for sparse vector)
        vx = cvxpy.Variable((y0.shape[0], axes3d), complex=True)
        vx.value = y0  # assigning initial vector to the vx

        s, sc = [], [] # the active and inactive atoms in the dictionary respectively
        temp = list(range(y0.shape[0]))
        loop_count = 0

        while len(sc) <= np.count_nonzero(abs(vx.value) < 0.): # the conditional statements were made to enter the while loop

            red = self._reduce(temp, sc)
            objective = cvxpy.Minimize(cvxpy.norm(vx[red,:], 1))
            constraints = [cvxpy.sum_squares(self.a[:,support][:,red] @ vx[red,:] - b) <= noise_level,
                           vx[sc,:] == 0] if len(sc) else [cvxpy.sum_squares(self.a[:,support][:,red] @ vx[red,:] - b) <= noise_level]
            prob = cvxpy.Problem(objective, constraints)
            prob.solve(solver=solver)

            index = []
            s = self._reduce(temp, sc) if not len(sc) else s
            for i, item in self.__nnwindow(s, nnw):
                count = np.count_nonzero(abs(vx.value[item]) <= zeta0 * max(abs(vx.value))[0])
                if count == len(item):
                    index.extend(item)

            sc = list(set(sc + index))
            s = self._reduce(s, sc)

            if len(index) == 0:
                zeta0 += dzeta
                nnw -= 1
                if nnw == 0:
                    break
            loop_count += 1

        if loop_count == 0:
            red = temp


        y_sparse_hat = vx.value                                # recovered sparse signal
        y_hat = self.a0[:,support].dot(y_sparse_hat)
        return y_hat, y_sparse_hat


    # basis pursuit denoising resp. spectral projected gradient for L1 minimization
    # (SPGL1) + off-support search, modified to fit boosted_superresolver
    @_boost_superresolver
    def nlht_lasso(self, y_signal, lam=None, nnw=9, zeta0=0.025, dzeta=0.025, y0=None, support=None, solver="CLARABEL"):
        """Basis pursuit denoising via non-local hard threshholding, in lasso
        form. See reference [1].

        Parameters
        ----------
        x_signal : array like
            X component of signal.
        y_signal : array like
            Signal.
        lam : float, optional
            Parameter for the LASSO optimization. If None lam is 5% of the ma-
            gnitude of a_tr.T*y_signal. The default is None.
        nnw : int, optional
            nnw neatrest neighbours to be searched for off support addition.
            The default is 9.
        zeta0 : float, optional
            Fraction of max(magnitude) of the reconstructed vector in the sparse
            representation. Is needed as a measure of noise consideration.
            The default is 0.025.
        dzeta : float, optional
            Step size for increasing zeta0 over each iteration.
            The default is 0.025.
        solver : str, optional
            Convex solver used in the package cvxpy. The default is "CLARABEL".

        Returns
        -------
        y_hat : array like
            Reconstructed array.
        y_sparse_hat : array like
            Sparse representation of reconstructed array (in general more di-
            mensions than reconstructed array, with shape (a0.shape[1], y_signal.shape[1])).

        References
        ----------
        .. [1] S. Gazit, A. Szameit, Y. C. Eldar, and M. Segev, “**Super-resolution
           and reconstruction of sparse sub-wavelength images**,” Optics Express,
           vol. 17, no. 26, pp. 23920–23946, 2009, doi: 10.1364/OE.17.023920.
        """

        y_t = y_signal.reshape((-1,1))
        signal_length, axes3d = y_t.shape

        lam = 0.007 * np.linalg.norm(self.a_tr.T.dot(y_t).ravel(), np.inf) + 1e-6 if lam is None else lam

        if support is None:
            support = range(self.n)
        elif 1 < len(support) <= nnw:
            nnw = len(support)//2
        else:
            nnw = 1

        b = self.ar.dot(y_t)  # random measured signal
        y0 = np.zeros((self.n, axes3d))[support] if y0 is None else y0 # initial vector (proxy for sparse vector)
        vx = cvxpy.Variable((y0.shape[0], axes3d), complex=True)
        vx.value = y0   # assigning initial vector to the vx

        temp = list(range(y0.shape[0]))
        s, sc = temp, [] # the active and inactive atoms in the dictionary respectively

        loop_count = 0
        objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(self.a[:,support] @ vx - b) + lam * cvxpy.norm(vx, 1))
        prob = cvxpy.Problem(objective)
        while len(sc) <= np.count_nonzero(abs(vx.value) < 0.): # the conditional statements were made to enter the while loop

            prob.solve(solver=solver)
            if len(sc):
                vx.value[sc] = 0.

            index = []
            s = self._reduce(temp, sc) if not len(sc) else s
            for i, item in self.__nnwindow(s, nnw):
                count = np.count_nonzero(abs(vx.value[item]) <= zeta0 * max(abs(vx.value))[0])
                if count == len(item):
                    index.extend(item)

            sc = list(set(sc + index))
            s = self._reduce(s, sc)

            if len(index) == 0:
                zeta0 += dzeta
                nnw -= 1
                if nnw == 0:
                    break
            loop_count += 1


        y_sparse_hat = vx.value                      # recovered sparse signal
        y_hat = self.a0[:,support][:,s].dot(y_sparse_hat[s,:])
        return y_hat, y_sparse_hat


    @staticmethod
    def __nnwindow(red, nnw):
        """Helper function for superresolver_nlht."""
        l = []
        dim = len(red)
        for i, item in enumerate(red):
            low_bound = i - i*(i < nnw//2) - (nnw//2)*(i > nnw)
            up_bound  = i + (nnw//2 + 1)*((dim - i) > nnw//2) + (dim - i)*((dim - i) < nnw//2)
            temp = [j for j in red[low_bound: up_bound] if abs(j - item) <= nnw//2]
            # if item in temp:
            #     temp.remove(item)
            l.append((item, temp))
        return l


    @staticmethod # Deprecated, will not be used, hence will be deleted in later versions
    def __GSA(y_hat, y_t, g_iter):
        """Gerchberg-Saxton algorithm for phase retrival."""
        for _ in range(g_iter):
            y_est_fft = scipy.fft.fft(y_t, axis=0)
            y_est_fft = np.abs(scipy.fft.fft(y_hat, axis=0))*np.exp(1j*np.angle(y_est_fft))
            y_est = scipy.fft.ifft(y_est_fft, axis=0)
            y_t = np.abs(y_hat)*np.exp(1j*np.angle(y_est))
        return y_t


    @staticmethod
    def _reduce(l1, l2, stop_cond=False):
        """Gives a list back which elements of list l1 are not in l2."""
        if stop_cond is False:
            return [item for item in l1 if item not in l2]

        if len(l2) == 0:
            return l1[:stop_cond]
        temp = []
        for item in l1 :
            if item not in l2:
                temp.append(item)
                if len(temp) == stop_cond:
                    break
        return temp

# =============================================================================