"""
The frames module of the cssr package provides a class for constructing frames,
i.e. (overcomplete) dictionaries for super-resolution via compressive sensing.
The Frames class allows users to create various types of frames that can be used
as sparsifying bases for signal processing tasks. They are a critical component
of sensing matrices. Currently the class includes frames such as Heaviside,
Gaussian, and Cauchy, as well as their overcomplete versions, etc.


Created on Fri Feb 12 19:47:33 2021
@author: marcelo looser
"""

import scipy
import numpy as np
import pylab as py
from functools import lru_cache

# =============================================================================

class Frames:
    """
    The Frames class provides methods to construct various types of frames 
    ((overcomplete) dictionaries) for signal processing tasks. These frames can 
    be used as sparsifying bases for compressive sensing applications and are 
    essential for constructing sensing matrices. 

    Parameters
    ----------
    x_signal : array like
        X-component of the signal.
    """

    a_frame = 0 # dynamically inherited by the FrameCheck class
    def __init__(self, x_signal):

        if not isinstance(x_signal, np.ndarray):
            raise ValueError("The first argument must be an array.")
        elif not (x_signal.ndim == 1 or (x_signal.ndim == 2 and x_signal.shape[1] == 1)):
            raise ValueError("The first argument must be an array of shape (n,) or (n,1).")

        self.signal_length = x_signal.shape[0]
        self.x = x_signal.reshape((-1,))


    @staticmethod
    def pprint_implemented_frames():
        print("Implemented frames: ", [
                    "heaviside", 
                    "heaviside_overcomplete",
                    "gaussian",
                    "gaussian_overcomplete",
                    "cauchy",
                    "cauchy_overcomplete",
                    "fourier"
                  ]
            )


    def heaviside(self, box_width): # if box_width 1, dirac frame
        """
        Heaviside frame, vectors (atoms) consist of binary entries (1's and 0's).
        The box_width parameter determines the width of the peaks.

        Parameters
        ----------
        box_width: int
            Width of the peaks.

        Returns
        -------
        a_frame : ndarray
            Sparsifying basis.
        """

        if not isinstance(box_width, int):
            raise ValueError("box_width must be an integer.")
        elif not (1 <= box_width < self.signal_length):
            raise ValueError(f"box_width must lie in the range [1, {self.signal_length-1}].")

        type(self).a_frame = np.zeros((self.signal_length, self.signal_length), dtype=complex)
        for j in range(1, self.signal_length+1):
            if j <= box_width:
                index = range(0, j)
                value = 1/np.sqrt(j)
            else:
                index = range(j-box_width, j)
            self.a_frame[index, j-1] = value
        return self.a_frame


    def heaviside_overcomplete(self, box_width_interval=None, step_size=None):
        """
        Heaviside frame, vectors (atoms) consist of binary entries (1's and 0's).
        The box_width_interval parameter determines the range of peak widths.

        Parameters
        ----------
        box_width_interval : list, optional
            Interval as a list of two integers [low, high] representing the box
            range for the frame. If None, a overcomplete dictionary with box widths ranging 
            from 1 to the signal length gets constructed. The default is None.
        step_size : int, optional
            A step size used to generate a list of equidistantly spaced box widths
            values from the box_width_interval. The default is None, corresponding 
            to a integer-step size that devides the box_width_interval length d times, 
            where 1 <= d <= 20 such that modulo(box_width_interval length, d) = 0.

        Returns
        -------
        a_frame : ndarray
            Sparsifying basis.
        """

        if box_width_interval is not None:
            if not isinstance(box_width_interval, list):
                raise ValueError("box_width_interval must be a list.")
            elif not len(box_width_interval) == 2:
                raise ValueError("box_width_interval must be a list of two integers.")
            elif not all(isinstance(i, int) for i in box_width_interval):
                raise ValueError("box_width_interval must be a list of two integers.")
            elif not (box_width_interval[0] < box_width_interval[1]):
                raise ValueError("box_width_interval must be a list of two integers in ascending order.")
            elif not (1 <= box_width_interval[0] < box_width_interval[1] < self.signal_length):
                raise ValueError(f"box_width_interval values must be in the range [1, {self.signal_length - 1}].")
        else:
            box_width_interval = [1, self.signal_length]

        if step_size is not None:
            if not isinstance(step_size, int):
                raise ValueError("step_size must be an integer.")
            elif not (1 <= step_size <= box_width_interval[1] - box_width_interval[0]):
                raise ValueError(f"step_size must be in the range [1, {box_width_interval[1] - box_width_interval[0]}].")
        else:
            step_size = max(d for d in range(1, 20) if (box_width_interval[1] - box_width_interval[0] + 1) % d == 0)

        delta = box_width_interval[1] - box_width_interval[0] + 1
        corr = sum(range(box_width_interval[0]-1, box_width_interval[1], step_size))
        type(self).a_frame = np.zeros((self.signal_length, (delta//step_size)*self.signal_length - corr), dtype=complex) # shape[1] = n(n+1)/2 - 1, spark = 3
        box_width = box_width_interval[0]
        for j in range(self.a_frame.shape[1]):
            temp_correction = self.__index_calc(box_width-1, start=box_width_interval[0]-1)
            index = range(j + temp_correction, j + temp_correction + box_width)
            self.a_frame[index, j] = 1/np.sqrt(box_width)
            if (j+1) + temp_correction == self.signal_length - (box_width-1):
                box_width += 1
        return self.a_frame


    @lru_cache 
    def __index_calc(self, box_width, start=0):
        """
        Helper function for heaviside_overcomplete.

        Parameters
        ----------
        box_width : int
            Width of the peaks.
        start : int, optional
            Starting index. The default is 0.

        Returns
        -------
        int
            Integer to locate new atom entries in the overcomplete dictonary.
        """

        if box_width == start:
            return 0
        elif box_width == start+1:
            return -self.signal_length + start
        else:
            return -(self.signal_length - (box_width-1)) + self.__index_calc(box_width-1, start=start)


    def gaussian(self, sigma):
        """
        Gaussian frame, the vectors (atoms) are normal PDFs.

        Parameters
        ----------
        sigma : float
            Standard deviation.

        Returns
        -------
        a_frame : ndarray
            Sparsifying basis.
        """

        if not isinstance(sigma, (int, float)):
            raise ValueError("sigma must be a float or an integer.")
        elif not (0 < sigma < abs(self.x[-1] - self.x[0])):
            raise ValueError("sigma must be between 0 and the absolute "\
                             f"length of the range of x {abs(self.x[-1] - self.x[0])}.")

        type(self).a_frame = np.zeros((self.signal_length, self.signal_length), dtype=complex)
        for i, mu in enumerate(self.x):
            gdist = scipy.stats.norm.pdf(self.x, mu, sigma).reshape((self.signal_length,))
            self.a_frame[:,i] = gdist/np.linalg.norm(gdist, 2)
        return self.a_frame


    def gaussian_overcomplete(self,  sigma_interval=None, step_size=None):
        """
        Gaussian frame, the vectors (atoms) are normal PDFs. The list expectation
        values is given by the entries in x, while the list of equidistant standard 
        deviations is determined through the sigma_interval and step_size. 

        Parameters
        ----------
        sigma_interval : list, optional
            If None, integers with values ranging from 1 to the signal length 
            will be used to determine a list of standard deviations of the 
            distributions. Else, a list of two floats must be passed. The first 
            entry is the lowest standard deviation and the second the highest 
            standard deviation. The default is None.
        step_size : float, optional
            A step size used to generate a list of equidistantly spaced standard 
            deviations from the sigma_interval values. The default is None, 
            corresponding to a step size of |x[-1]-x[0]|/10.

        Returns
        -------
        a_frame : ndarray
            Sparsifying basis.
        """

        if sigma_interval is not None:
            if not isinstance(sigma_interval, list):
                raise ValueError("sigma_interval must be a list.")
            elif len(sigma_interval) != 2:
                raise ValueError("sigma_interval must contain two numbers.")
            elif not all(isinstance(i, (int, float)) for i in sigma_interval):
                raise ValueError("sigma_interval must contain two numbers.")
            elif not (0 < sigma_interval[0] < sigma_interval[1] < abs(self.x[-1] - self.x[0])):
                raise ValueError("sigma_interval must contain two numbers "\
                                 f"between 0 and the absolute length of the range of x {abs(self.x[-1] - self.x[0])}.")
        else:
            sigma_interval = [abs(self.x[1] - self.x[0]), abs(self.x[-1] - self.x[0]) - abs(self.x[1] - self.x[0])]
        
        if step_size is not None:
            if not isinstance(step_size, (int, float)):
                raise ValueError("step_size must be a float or an integer.")
            elif not (0 < step_size < sigma_interval[1] - sigma_interval[0]):
                raise ValueError(f"step_size must be between 0 and the sigma_interval size {sigma_interval[1] - sigma_interval[0]}.")
        else:
            step_size = abs(self.x[-1] - self.x[0])/10

        steps = (sigma_interval[1] - sigma_interval[0])/step_size
        n = int(steps) + 1*(steps - int(steps) != 0.0) # there might be a probability that a FloatingPointError will accure, fix it
        type(self).a_frame = np.zeros((self.signal_length, self.signal_length*n), dtype=complex)
        for j, sigma in enumerate(py.arange(sigma_interval[0], sigma_interval[1], step_size)):
            for i, mu in enumerate(self.x):
                gdist = scipy.stats.norm.pdf(self.x, mu, sigma)
                self.a_frame[:,j*self.signal_length + i] = gdist/np.linalg.norm(gdist, 2)
        return self.a_frame


    def cauchy(self, gamma):
        """
        Cauchy (Lorentz) frame, the vectors (atoms) are cauchy PDFs.

        Parameters
        ----------
        gamma:
            Half width at half maximum (HWHM).

        Returns
        -------
        a_frame : ndarray
            Sparsifying basis.
        """

        if not isinstance(gamma, (int, float)):
            raise ValueError("gamma must be a float or an integer.")
        elif not (0 < gamma < abs(self.x[-1] - self.x[0])):
            raise ValueError("gamma must be between 0 and the absolute "\
                             f"length of the range of x {abs(self.x[-1] - self.x[0])}.")
        
        type(self).a_frame = np.zeros((self.signal_length, self.signal_length), dtype=complex)
        for i, mu in enumerate(self.x):
            cdist = self.__cauchy_pdf(mu, gamma).reshape((self.signal_length,))
            self.a_frame[:,i] = cdist/np.linalg.norm(cdist, 2)
        return self.a_frame
    

    def cauchy_overcomplete(self, gamma_interval=None, step_size=None):
        """
        Cauchy (Lorentz) frame, the vectors (atoms) are Cauchy (Lorentz) PDFs. The list expectation
        values is given by the entries in x, while the list of half width at half maximas is 
        determined through the gamma_interval and step_size.

        Parameters
        ----------
        gamma_interval : list, optional
            If None, integer values from 1 to the signal length will be used for
            the half width at half maximum (HWHM) of the distributions. Else a
            list with two values must be passed, where the first one is the lowest
            HWHM and the second the highest HWHM.
            The default is None.
        step_size : float, optional
            A step size used to generate a list of equidistantly spaced HWHM
            values from the gamma_interval values. The default is None, 
            corresponding to a step size of |x[-1]-x[0]|/10.

        Returns
        -------
        a_frame : ndarray
            Sparsifying basis.
        """

        if gamma_interval is not None:
            if not isinstance(gamma_interval, list):
                raise ValueError("gamma_interval must be a list.")
            elif len(gamma_interval) != 2:
                raise ValueError("gamma_interval must contain two numbers.")
            elif not all(isinstance(i, (int, float)) for i in gamma_interval):
                raise ValueError("gamma_interval must contain two numbers.")
            elif not (0 < gamma_interval[0] < gamma_interval[1] < abs(self.x[-1] - self.x[0])):
                raise ValueError("gamma_interval must contain two numbers "\
                                 f"between 0 and the absolute length of the range of x {abs(self.x[-1] - self.x[0])}.")
        else:
            gamma_interval = [abs(self.x[1] - self.x[0]), abs(self.x[-1] - self.x[0]) - abs(self.x[1] - self.x[0])]

        if step_size is not None:
            if not isinstance(step_size, (int, float)):
                raise ValueError("step_size must be a float or an integer.")
            elif not (0 < step_size < gamma_interval[1] - gamma_interval[0]):
                raise ValueError(f"step_size must be between 0 and the gamma_interval size {gamma_interval[1] - gamma_interval[0]}.")
        else:
            step_size = abs(self.x[-1] - self.x[0])/10

        steps = (gamma_interval[1] - gamma_interval[0])/step_size
        n = int(steps) + 1*(steps - int(steps) != 0.0) # !!!there is a probability that a FloatingPointError will occure, fix it
        type(self).a_frame = np.zeros((self.signal_length, self.signal_length*n), dtype=complex)
        for j, gamma in enumerate(py.arange(gamma_interval[0], gamma_interval[1], step_size)):
            for i, mu in enumerate(self.x):
                cdist = self.__cauchy_pdf(mu, gamma)
                self.a_frame[:,j*self.signal_length + i] = cdist/np.linalg.norm(cdist, 2)
        return self.a_frame


    def __cauchy_pdf(self, mu, gamma):
        """
        Cauchy (Lorentz) PDF.

        Parameters
        ----------
        mu:
            Expectation value.
        gamma:
            Half width at half maximum (HWHM).
            
        Returns
        -------
        PDF value at x.
        """
        return (1/np.pi)*(gamma/((self.x - mu)**2 + gamma**2))


    def fourier(self):
        """
        DFT frame, centered around the spatial frequency zero.

        Returns
        -------
        a_frame : ndarray
            Sparsifying basis.
        """
        
        type(self).a_frame = np.zeros((self.signal_length, self.signal_length), dtype=complex)
        self.a_frame = scipy.fft.fftshift(scipy.linalg.dft(self.signal_length))
        return self.a_frame


    # implementation needed
    def _wavelet(self, *args):
        """Wavelet dictionary."""
        pass

    # implementation needed
    def _mega_dict(self, *args):
        """Combines all frames for a mega dictionary."""
        pass

    # implementation needed
    def _mega_dict_overcomplete(self, *args):
        """Combines all frames for a mega overcomplete dictionary."""
        pass


# =============================================================================
# under construction ...
class FrameCheck(Frames):
    """Some tools to test the dictonaries."""

    def __init__(self):
        self.a_frame = Frames.a_frame


    def mutual_coherence(self, a=False):
        """Caculates the mutual cohrence of the matrix a_frame. If one wants
        to check a different matrix set the variable a to the corresponding
        matrix to be checked."""
        if a is False:
            a = self.a_frame
        gram, _ = FrameCheck.gram_matrix(self, normed=True)
        gram.ravel()[::gram.shape[1]+1] = 0 # setting diagonal elements to zero
        return abs(gram).max()


    def gram_matrix(self, a=False, normed=False):
        """Calculates the gram matrix."""
        if a is False:
            a = self.a_frame
        if normed:
            a = a/np.linalg.norm(a, ord=2, axis=0)
        ah = a.conj().T
        gram = ah.dot(a)
        return gram, ah


    # Deprecated
    def _upper_bound(self, p, a=False):
        gram, _ = FrameCheck.gram_matrix(self, a)
        bound = max([abs(sum(gram[:, j]))**p for j in range(gram.shape[1])])**(1/p)
        return bound

# =============================================================================
