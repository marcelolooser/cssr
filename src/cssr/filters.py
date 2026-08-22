
"""
The filters module of the cssr package provides a class for filtering signals and
sensing matrices. The Filters class allows users to apply these filters to input
arrays, either as a critical ingridient for constructing sensing matrices or to
filter signals itself. Currently it includes various low-pass filters, such as
ideal low-pass, instrumental, and thermal filters, as well as other general filters 
(high-pass, band-pass, and band-stop filters). In paricular, the instrumental
and thermal filters are based on specific physical models, as described in the
reference provided in the documentation. 


Created on Wed Mar 17 19:53:02 2021
@author: marcelo looser
"""

import scipy
import numpy as np
import scipy.signal

# =============================================================================

class Filters:
    """
    The Filters class provides a collection of (primaryly low-pass) filter operators that 
    are used first and foremost as a central component in constructing sensing matrices, 
    but can also be applied to signals. In particular, the class is initialized 
    with a sparsifying matrix or datapoints as its first parameter.
    
    Parameters
    ----------
    a0 : array like
        Sparsifying matrix (or signal).
    x_signal : array like
        X-components of the signal.
    y_signal : array like, optional
        Y-components of the signal can be provided iff no cutoff is provided 
        and a rough estimation of the cutoff frequency is known. The latter 
        must be provided through the optional parameter threshold_level. 
        The default is None.
    cutoff : float or list, optional
        Cutoff "frequency" of filter (expressed in the same units as the fourier 
        transform of x_signal) or a list of cutoff "frequencies" (that is, band 
        edges) for a band-pass or band-stop filters. The latter option is used 
        for FIR and Butterworth filters. For the thermal and instrumental low-pass 
        filters, this parameter represents the temperature and energy cutoff, 
        respectively. If no cuttoff is provided, a rough estimation is made.
        The default is None.
    threshold_level : float, optional
        If cutoff is None, threshold_level will be used to make a rough estimate 
        of the cutoff "frequency" using threshold_level*max(magnitude of y_fft),
        and it is assumed that the used filter is a low-pass filter.
        (This is an experimental feature.) The default is 2e-2.
    filter_signal : boolean, optional
        If True, a filter will be applied to the signal a0. If False, 
        a filter will be applied to the sparsifying matrix a0. 
        The default is False.
    """

    def __init__(self, a0, x_signal, cutoff=None, y_signal=None,  threshold_level=2e-2, filter_signal=False):

        if not isinstance(filter_signal, bool):
            raise TypeError("filter_signal must be a boolean.")

        if not isinstance(a0, np.ndarray):
            raise ValueError("The first argument must be an array.")
        else:
            self.a_tr = a0.copy()
            if filter_signal:
                if not (self.a_tr.ndim == 1 or (self.a_tr.ndim == 2 and self.a_tr.shape[1] == 1)):
                    raise ValueError("The first argument must be an array of shape (n,) "\
                                    "or (n,1) if filter_signal is True.")
                else: 
                    self.a_tr = self.a_tr.reshape((-1,))
            elif self.a_tr.ndim == 1 or (self.a_tr.ndim == 2 and self.a_tr.shape[1] == 1):
                    raise ValueError("The first argument must be an array of shape (n, m) "\
                                    "if filter_signal is False.")

        if not isinstance(x_signal, np.ndarray):
            raise ValueError("The second argument must be an array.")
        elif not (x_signal.ndim == 1 or (x_signal.ndim == 2 and x_signal.shape[1] == 1)):
            raise ValueError("The second argument must be an array of shape (n,) or (n,1).")

        if cutoff is not None:
            self.cutoff_detector_estimate = False
            if isinstance(cutoff, (float, int)):
                if cutoff <= 0:
                    raise ValueError("cutoff must be a positive non-zero value.")
            elif isinstance(cutoff, list):
                if len(cutoff) != 2:
                    raise ValueError("If cutoff is provided as a list, it must contain "\
                                     "exactly two values.")
                elif cutoff[0] >= cutoff[1]:
                    raise ValueError("If cutoff is provided as a list, its entries must be "\
                                     "sorted in ascending order.")
            if y_signal is not None:
                print("Warning: y_signal is provided but cutoff is not None; y_signal will "\
                      "be ignored.")
                
        elif cutoff is None:
            if (self.a_tr.ndim == 1 or (self.a_tr.ndim == 2 and self.a_tr.shape[1] == 1)):
                raise ValueError("Missing required third (keyword) argument cutoff.")
            elif isinstance(y_signal, np.ndarray):
                if (y_signal.ndim == 1 or (y_signal.ndim == 2 and y_signal.shape[1] == 1)):
                    self.y = y_signal.reshape((-1,1))
                else:
                    raise ValueError("If cutoff is None, y_signal must be provided as an array of shape (n,) or (n,1) "\
                                     "in order to make a crude estimation of the cutoff \"frequency\". Else, provide a cutoff.")
            else:
                raise ValueError("If cutoff is None, y_signal must be provided as an array of shape (n,) or (n,1) "\
                                 "in order to make a crude estimation of the cutoff \"frequency\". Else, provide a cutoff.")

        if not (0 < threshold_level < 1):
            raise ValueError("Threshold level must be a positive value between 0 and 1.")

        self.cutoff = cutoff
        self.filter_signal = filter_signal
        self.y = y_signal
        self.x = x_signal.reshape((-1,1))

        self.__x_fft()
        self.__low_pass_detector(threshold_level=threshold_level) #will be modified in later versions


    @staticmethod
    def pprint_implemented_filters():
        print("Implemented filters: ", [
                  "heaviside_lowpass_filter",
                  "fir_filter",
                  "butter_filter",
                  "instrumental_lowpass_filter",
                  "thermal_lowpass_filter",
                  ]
            )


    def heaviside_lowpass_filter(self):
        """
        Ideal low-pass filter.

        Returns
        -------
        a_tr : ndarray
            An ideal low-pass filter convoluted with a0.
        """

        coeff = scipy.fft.fftshift(scipy.fft.fft(self.__heaviside_box_function()))
        coeff = coeff/(sum(coeff.real))
        self.__convolve(coeff)
        return self.a_tr


    def __heaviside_box_function(self):
        """
        A variation of the heaviside step function. The function value is one
        in the interval [-cutoff, cutoff] and otherwise zero. Note, this is done in a
        reversed manner , i.e., (abs(x_fft) > cutoff) instead of (abs(x_fft) < cutoff),
        due to scipy.fft.fft (alternatively x_fft must be generated with an addional 
        shift scipy.fft.fftshift).

        Returns
        -------
        Binary array based on the cutoff frequency.
        """
        return 1*(abs(self.x_fft) > abs(self.x_fft).max() - self.cutoff)


    def fir_filter(self, numtabs=4, pass_zero="lowpass"):
        """
        Finite impulse response (FIR) filter.

        Parameters
        ----------
        numtabs : int, optional
            Length of the filter (number of coefficients, i.e., the filter
            order + 1). numtaps must be odd if a passband includes the Nyquist
            frequency. The default is 4.
        pass_zero :  {True, False, "bandpass", "lowpass", "highpass", "bandstop"}, optional
            If True, the gain at the frequency 0 (i.e., the "DC gain") is 1. If
            False, the DC gain is 0. Can also be a string parameter for the desired 
            filter type (equivalent to btype in IIR design functions).
            If a passband is used the cutoff must be an interval, i.e., 
            a list of two floats. The default is "lowpass".

        Returns
        -------
        a_tr : ndarray
            A FIR filter convoluted with a0.
        """

        tabs = self.__fir(numtabs, pass_zero)
        self.a_tr = scipy.signal.filtfilt(tabs, 1, self.a_tr, padlen=0) # foward-backwards filtering
        return self.a_tr


    def __fir(self, numtabs=4, pass_zero="lowpass"):
        """
        This function computes the coefficients of a finite impulse response
        filter. For more information see documentation "firwin" in scipy.signal.

        Parameters
        ----------
        numtabs : int, optional
            Length of the filter (number of coefficients, i.e. the filter
            order + 1). numtaps must be odd if a passband includes the Nyquist
            frequency. The default is 4.
        pass_zero :  {True, False, "bandpass", "lowpass", "highpass", "bandstop"}, optional
            If True, the gain at the frequency 0 (i.e., the "DC gain") is 1. If
            False, the DC gain is 0. Can also be a string parameter for the de-
            sired filter type (equivalent to btype in IIR design functions).
            If a passband is used the cutoff must be an interval, i.e., 
            a list of two floats. The default is "lowpass".

        Returns
        -------
        tabs : ndarray
            An array of coefficients for the FIR filter.
        """

        if pass_zero in ["bandpass", "bandstop"]:
            if self.cutoff_detector_estimate:
                print("Warning: pass_zero is set to \"bandpass\" or \"bandstop\", but the estimated cutoff "\
                     "is presumed to characterize a low-pass filter. pass_zero will be set to \"lowpass\".")
                pass_zero = "lowpass"
            else:
                if not isinstance(self.cutoff, list):
                    raise ValueError("If pass_zero is \"bandpass\" or \"bandstop\", cutoff must be a list of two floats.")

        fs = abs(self.x_fft[-1] - self.x_fft[0]) # Sampling rate, or number of measurements per second
        nyq = 0.5*fs # nyquist frequency
        cutoff = self.cutoff / nyq if isinstance(self.cutoff, (float, int)) else [c/nyq for c in self.cutoff]
        tabs = scipy.signal.firwin(numtabs, cutoff, pass_zero=pass_zero)
        return tabs


    def butter_filter(self, order=4, btype="lowpass"):
        """
        Butterworth filter.

        Parameters
        ----------

        N : int
            Order of the filter.
        btype : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
            Filter type. The default is "lowpass".

        Returns
        -------
        a_tr : ndarray
            A Butterworth filter convoluted with a0.
        """

        sos = self.__butter(order, btype)
        self.a_tr = scipy.signal.sosfiltfilt(sos, self.a_tr, padlen=0) # foward-backwards filtering
        return self.a_tr


    def __butter(self, order=4, btype="lowpass"):
        """
        Butterworth filter implemented via the scipy signal package. 
        For more information see documentation "butter" in scipy.signal.

        Parameters
        ----------

        N : int
            Order of the filter.
        btype : {"lowpass", "highpass", "bandpass", "bandstop"}, optional
            Filter type. Default is "lowpass".

        Returns
        -------
        sos : ndarray
            Second-order sections representation of the IIR filter.
        """

        if btype in ["bandpass", "bandstop"]:
            if self.cutoff_detector_estimate:
                print("Warning: btype is set to \"bandpass\" or \"bandstop\", but the estimated cutoff "\
                     "is presumed to characterize a low-pass filter. btype will be set to \"lowpass\".")
                btype = "lowpass"
            else:
                if not isinstance(self.cutoff, list):
                    raise ValueError("If btype is \"bandpass\" or \"bandstop\", cutoff must be a list of two floats.")

        fs = abs(self.x_fft[-1] - self.x_fft[0]) # Sampling rate, or number of measurements per second
        nyq = 0.5*fs # nyquist frequency
        cutoff = self.cutoff / nyq if isinstance(self.cutoff, (float, int)) else [c/nyq for c in self.cutoff]
        sos = scipy.signal.butter(order, cutoff, btype=btype, output="sos")
        return sos


    def instrumental_lowpass_filter(self):
        """
        Instrumental function used as lowpass filter. See reference [1].

        Returns
        -------
        a_tr : ndarray
            Instrumental function convoluted with a0.

        References
        ----------
        .. [1] J. Klein, A. Léger, M. Belin, D. Défourneau, and M. J. L. Sangster,
           "Inelastic-electron-tunneling spectroscopy of metal-insulator-metal
           junctions**," Physical Review B, vol. 7, no. 6, pp. 2336–2348, Mar.
           1973, doi: 10.1103/PhysRevB.7.2336.
        """

        coeff = self.__instrumental_function()
        coeff = coeff/(sum(coeff))
        self.__convolve(coeff)
        return self.a_tr


    def __instrumental_function(self):
        """
        Instrumental function up to a factor 1/e where e is the elementary
        charge. See reference [1].

        Returns
        -------
        Instrumental function evaluated at x values.

        References
        ----------
        .. [1] J. Klein, A. Léger, M. Belin, D. Défourneau, and M. J. L. Sangster,
           "Inelastic-electron-tunneling spectroscopy of metal-insulator-metal
           junctions**," Physical Review B, vol. 7, no. 6, pp. 2336–2348, Mar.
           1973, doi: 10.1103/PhysRevB.7.2336.
        """

        Vmod = self.cutoff
        x = self.x.ravel()
        if Vmod < 0.:
            Vmod = 1
        return (8/(3*np.pi)) * ((np.sign(abs(Vmod**2 - x**2)) * (abs(Vmod**2 - x**2)**(3/2)))/(Vmod**4)) * (abs(x) < Vmod)


    def thermal_lowpass_filter(self):
        """
        Thermal function times e. See reference [1].

        Returns
        -------
        a_tr : ndarray
            Thermal function convoluted with a0.

        References
        ----------
        .. [1] J. Klein, A. Léger, M. Belin, D. Défourneau, and M. J. L. Sangster,
           "Inelastic-electron-tunneling spectroscopy of metal-insulator-metal
           junctions**," Physical Review B, vol. 7, no. 6, pp. 2336–2348, Mar.
           1973, doi: 10.1103/PhysRevB.7.2336.
        """

        coeff = self.__thermal_function()
        coeff = coeff/(sum(coeff))
        self.__convolve(coeff)
        return self.a_tr


    def __thermal_function(self):
        """
        Thermal function times e. See reference [1].

        Returns
        -------
        Thermal function evaluated at x values.

        References
        ----------
        .. [1] J. Klein, A. Léger, M. Belin, D. Défourneau, and M. J. L. Sangster,
           "Inelastic-electron-tunneling spectroscopy of metal-insulator-metal
           junctions**," Physical Review B, vol. 7, no. 6, pp. 2336–2348, Mar.
           1973, doi: 10.1103/PhysRevB.7.2336.
        """

        temperature = self.cutoff
        x = self.x.ravel()
        if temperature <= 0:
            return np.ones(x.shape)
        else:
            factor = (1.60217662/1.38064852) * 1e4 # e/k
            x_temp = np.array(x, dtype=float) # for np.exp
            v = (x_temp/temperature) * factor

            with np.errstate(over='ignore', invalid='ignore'): # supress overflow and invaild value warnings
                u = np.exp(v)
                chi = ((1/temperature) * u * ((v - 2) * u + v + 2)/(u - 1)**3)

            chi = chi * (chi <= 1/(6*temperature)) + 1/(6*temperature) * (chi > 1/(6*temperature)) + 1/(6*temperature) * (chi == 0)*(abs(x_temp) <= 1e-3)  # there is a numerical probelm wich is handeld by cutting off values which tend to diverge but are in fact bounded
            chi = chi* factor * (chi >= 1e-15) # drop to small values
            chi = np.nan_to_num(chi, copy=True, nan=0.0) # drop nans and replace them by zero, these is caused by overflow in the power operation
            return chi


    def __convolve(self, coeff):
        """
        Convolves a coeff with a0.

        Parameters
        ----------
        coeff : ndarray
            Coefficients of the filter.

        Returns
        -------
        None.
        """

        if not self.filter_signal:
            self.a_tr = scipy.signal.convolve(self.a_tr, coeff[:,None], mode="same")
        else:
            self.a_tr = scipy.signal.convolve(coeff, self.a_tr, mode="same")


    # experimental feature, will be modified in later versions
    def __low_pass_detector(self, threshold_level=2e-2):
        """
        Detects the lowest passed frequency, provided the threshold is 
        suited, presuming the signal was band-limited by low-pass filtering. 
        This function will be modified in later versions.

        Parameters
        ----------
        threshold_level : float, optional
            Threshold as a fraction of maximal amplitude of the fourier trans-
            formed signal. This is used to find a rough estimation of the cutoff
            frequency. (Will be modified in later versions). The default is 2e-2.

        Returns
        -------
        None.
        """

        if self.cutoff is None:
            y_fft = scipy.fft.fftshift(scipy.fft.fft(self.y, axis=0))
            y_fft = y_fft/np.linalg.norm(y_fft, "fro")
            threshold = max(abs(y_fft))*threshold_level
            cuttoff_arg = next(i for i, item in enumerate(abs(y_fft) >= threshold) if item)
            self.cutoff = abs(self.x_fft[cuttoff_arg])
            self.cutoff_detector_estimate = True


    def __x_fft(self):
        """ Calculates the frequency component in the fourier domain."""
        m = len(self.a_tr)
        x_fft = scipy.fft.fftshift(scipy.fft.fftfreq(m, d=abs(self.x[-1,0]-self.x[0,0])/self.x.shape[0])) # spatial frequency centered around 0
        self.x_fft = np.array(x_fft, dtype=object) # formating due to python 32bit


# =============================================================================
