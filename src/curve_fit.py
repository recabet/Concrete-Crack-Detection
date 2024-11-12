import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, List, Any, Optional


class CurveFit:
    def __init__ (self):
        self.__x: Optional[np.ndarray] = None
        self.__y: Optional[np.ndarray] = None
        self.amplitudes: Optional[Tuple[float, float]] = None
        self.phases: Optional[Tuple[float, float]] = None
        self.frequencies: Optional[Tuple[float, float]] = None
    
    @staticmethod
    def extract_crack_coordinates (mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the coordinates of the crack (non-zero pixels) from the binary mask.

        Parameters:
        mask (np.ndarray): Binary mask representing the crack.

        Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y coordinates of the crack pixels.
        """
        y, x = np.nonzero(mask)
        return x, y
    
    def sine_cosine_function (self, x: np.ndarray,
                              amplitude_sin: float,
                              frequency_sin: float,
                              phase_sin: float,
                              amplitude_cos: float,
                              frequency_cos: float,
                              phase_cos: float) -> np.ndarray:
        """
        Sine-cosine function used for fitting.

        Parameters:
        x (np.ndarray): X-coordinates for the sine-cosine function.
        amplitude_sin (float): Amplitude of the sine component.
        frequency_sin (float): Frequency of the sine component.
        phase_sin (float): Phase shift of the sine component.
        amplitude_cos (float): Amplitude of the cosine component.
        frequency_cos (float): Frequency of the cosine component.
        phase_cos (float): Phase shift of the cosine component.

        Returns:
        np.ndarray: Computed Y-values based on the sine-cosine function.
        """
        self.amplitudes = (amplitude_sin, amplitude_cos)
        self.phases = (phase_sin, phase_cos)
        self.frequencies = (frequency_sin, frequency_cos)
        
        return (amplitude_sin * np.sin(frequency_sin * x + phase_sin) +
                amplitude_cos * np.cos(frequency_cos * x + phase_cos))
    
    @staticmethod
    def polynomial_fit (x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
        """
        Fits a polynomial function to the extracted crack coordinates.

        Parameters:
        x (np.ndarray): X-coordinates of the crack.
        y (np.ndarray): Y-coordinates of the crack.
        degree (int): Degree of the polynomial.

        Returns:
        np.ndarray: Coefficients of the polynomial fit.
        """
        coeffs = np.polyfit(x, y, degree)
        return coeffs
    
    def sine_cosine_fit (self, x: np.ndarray, y: np.ndarray, initial_guess: List[float]) -> Optional[np.ndarray]:
        """
        Fits a sine and cosine function to the extracted crack coordinates.

        Parameters:
        x (np.ndarray): X-coordinates of the crack.
        y (np.ndarray): Y-coordinates of the crack.
        initial_guess (List[float]): Initial guesses for the sine-cosine function parameters.

        Returns:
        Optional[np.ndarray]: Fitted parameters of the sine-cosine function, or None if fitting fails.
        """
        try:
            params, _ = curve_fit(self.sine_cosine_function, x, y, p0=initial_guess, maxfev=10000)
            return params
        except RuntimeError as e:
            print(f"Sine-Cosine fit failed: {e}")
            return None
    
    @staticmethod
    def plot_fit (x: np.ndarray, y: np.ndarray, fit_x: np.ndarray, fit_y: np.ndarray, fit_type: str) -> None:
        """
        Plots the original crack coordinates and the fitted curve.

        Parameters:
        x (np.ndarray): Original X-coordinates of the crack.
        y (np.ndarray): Original Y-coordinates of the crack.
        fit_x (np.ndarray): X-coordinates for the fitted curve.
        fit_y (np.ndarray): Y-coordinates for the fitted curve.
        fit_type (str): Type of fit (e.g., 'polynomial' or 'sine_cosine').

        Returns:
        None
        """
        plt.scatter(x, y, label="Original Crack", color="blue", s=10)
        plt.plot(fit_x, fit_y, label=f"{fit_type.capitalize()} Fit", color="red")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Crack Fitting - {fit_type.capitalize()}")
        plt.legend()
        plt.show()
    
    def fit_crack_function (self, mask: np.ndarray, fit_type: str = "polynomial", degree: int = 1,
                            plot: bool = False) -> None:
        """
        Fits a mathematical function to the crack in a binary mask.

        Parameters:
        mask (np.ndarray): Binary mask representing the crack.
        fit_type (str): Type of fit ('polynomial' or 'sine_cosine'). Defaults to 'polynomial'.
        degree (int): Degree of the polynomial fit. Only relevant for polynomial fitting. Defaults to 1.
        plot (bool): Whether to plot the fitted function. Defaults to False.

        Returns:
        None
        """
        self.__x, self.__y = self.extract_crack_coordinates(mask)
        
        if fit_type == "polynomial":
            poly_coeffs = self.polynomial_fit(self.__x, self.__y, degree)
            fit_y = np.polyval(poly_coeffs, self.__x)
            if plot:
                self.plot_fit(self.__x, self.__y, self.__x, fit_y, "polynomial")
            print(f"Polynomial coefficients: {poly_coeffs}")
        
        elif fit_type == "sine_cosine":
            initial_guess = [np.ptp(self.__y) / 2, 0.01, 0, np.ptp(self.__y) / 2, 0.01, 0]
            sine_cosine_params = self.sine_cosine_fit(self.__x, self.__y, initial_guess)
            if sine_cosine_params is not None:
                fit_y = self.sine_cosine_function(self.__x, *sine_cosine_params)
                if plot:
                    self.plot_fit(self.__x, self.__y, self.__x, fit_y, "sine_cosine")
                print(f"Sine-Cosine parameters: {sine_cosine_params}")
        
        else:
            print("Invalid fit type. Choose 'polynomial' or 'sine_cosine'.")
    
    def get_amplitudes (self) -> Optional[Tuple[float, float]]:
        """
        Returns the amplitudes of the sine and cosine components after fitting.

        Returns:
        Optional[Tuple[float, float]]: Amplitudes of sine and cosine components.
        """
        return self.amplitudes
    
    def get_phases (self) -> Optional[Tuple[float, float]]:
        """
        Returns the phases of the sine and cosine components after fitting.

        Returns:
        Optional[Tuple[float, float]]: Phases of sine and cosine components.
        """
        return self.phases
    
    def get_frequencies (self) -> Optional[Tuple[float, float]]:
        """
        Returns the frequencies of the sine and cosine components after fitting.

        Returns:
        Optional[Tuple[float, float]]: Frequencies of sine and cosine components.
        """
        return self.frequencies
