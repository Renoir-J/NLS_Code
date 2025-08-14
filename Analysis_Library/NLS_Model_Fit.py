import h5py
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import NLS_Data_Analysis
import NLS_Data_Store



class nls_model_fit:
    def __init__(self, t, delta_p, p_s, bounds:bool = False, bounded_parameter:list = [], fit_A:bool = True, fit_B:bool = True):
        self.bounds = bounds
        self.bounded_parameter = bounded_parameter
        self.fit_A = fit_A
        self.fit_B = fit_B  
        self.t1 = None
        self.w = None
        self.A = None
        self.B = None 
        self.t = t
        self.y = delta_p/p_s



    # NLS (Nucleation-Limited Switching) fit with Lorentzian distribution of log t0
    # Requirements from user:
    # 1) Input: time array (seconds) and target array y = ΔP / Ps
    # 2) LMSE (least-mean-square error) fitting
    # 3) Print fitted parameters and plot the fitted curve
    #
    # Notes:
    # - We implement the NLS response as:
    #     y_model(t) = A * ∫ F(x; x0=ln(t1), w) * [1 - exp(-(t/exp(x))**2)] dx + B
    #   where F is a normalized Lorentzian in x = ln(t0).
    # - We use natural logarithm (ln). If your paper used log10, convert initial guesses accordingly
    #   or pass base='10' to automatically convert provided initial guesses to ln internally.
    #
    # - The integral is evaluated numerically on a symmetric grid around x0 with ±L*w span.
    #   L is set to 12 by default to capture Lorentzian tails.
    #
    # This cell defines:
    #   fit_nls_lorentz(t, y, ...): returns dict of fitted parameters and plots result.
    #
    # At the end, we include a small demo with synthetic data if you run this cell standalone.
    @classmethod
    def _lorentz_pdf(cls, x, x0, w):
        """Normalized Lorentzian (Cauchy) PDF in x with center x0 and HWHM w (natural log domain)."""
        return (1/np.pi) * (w / ((x - x0)**2 + w**2))



    @classmethod
    def _nls_model(cls, t, t1, w, A=1.0, B=0.0, L=12, N=1000):
        # if want to change the integration range & accuracy, change L and N here !!!
        """
        Compute y_model(t) for given params using numerical quadrature over x=ln t0.
        t  : array of times (s)
        t1 : center time (s) corresponding to x0 = ln t1
        w  : HWHM in ln-space (dimensionless)
        A  : amplitude scaling for ΔP/Ps (default 1)
        B  : baseline offset (default 0)
        L  : integration span in units of w (±L*w)
        Consider as effective range, because the portion beyond +_12*w_eff is too small, the contribution to the overall polarization can be ignored
        N  : number of points for numeric integration
        That is, the inverse of integration step_size, indicating the integration accuracy
        """
        t = np.asarray(t, dtype=float)
        # Integration grid over x = ln(t0)
        x0 = np.log(t1)
        # guard w to avoid zero/negative
        w_eff = max(w, 1e-10)
        x_min = x0 - L * w_eff
        x_max = x0 + L * w_eff
        xs = np.linspace(x_min, x_max, N)
        dx = xs[1] - xs[0]
        # Lorentzian weights
        F = nls_model_fit._lorentz_pdf(xs, x0, w_eff)  # shape (N,)
        # For each t, compute integrand and integrate over x
        # integrand = F(xs) * (1 - exp(-(t/exp(xs))**2))
        t0s = np.exp(xs)  # (N,)
        # Broadcast to (M,N)
        T = t.reshape(-1, 1)
        ratio = np.clip(T / t0s, 0, 1e200)  # avoid overflow
        integrand = F * (1.0 - np.exp(-(ratio**2)))
        # for typical model, we take n = 2 in thin film case. 
        integral = np.trapz(integrand, xs, axis=1)  # (M,)
        y_model = A * integral + B
        return y_model



    # Bounds, set which fitting paramenter need to be bounded within a certain range
    def _get_bounds(self):
            lb = []
            ub = []
            # t1
            if self.bounds and 't1' in self.bounded_parameter:
                lb.append(self.bounded_parameter['t1'][0]); ub.append(self.bounded_parameter['t1'][1])
            else:
                lb.append(1e-12); ub.append(1e6*max(1.0, np.max(self.t)))
            # w
            if self.bounds and 'w' in self.bounded_parameter:
                lb.append(self.bounded_parameter['w'][0]); ub.append(self.bounded_parameter['w'][1])
            else:
                lb.append(1e-4); ub.append(10.0)
            # A
            if self.fit_A:
                if self.bounds and 'A' in self.bounded_parameter:
                    lb.append(self.bounded_parameter['A'][0]); ub.append(self.bounded_parameter['A'][1])
                else:
                    lb.append(0.8); ub.append(1.2)
            # B
            if self.fit_B:
                if self.bounds and 'B' in self.bounded_parameter:
                    lb.append(self.bounded_parameter['B'][0]); ub.append(self.bounded_parameter['B'][1])
                else:
                    lb.append(-0.2); ub.append(0.2)
                    # default bounded value has been set, if user did not provide bounded parameter but bounds = True, usedefault value

            return (np.array(lb, float), np.array(ub, float))



    # Residual function
    def residuals(self, p):
        # unpack
        i = 0
        t1 = p[i]; i += 1
        w  = p[i]; i += 1
        if self.fit_A:
            A = p[i]; i += 1
        else:
            A = 1 # default value
        if self.fit_B:
            B = p[i]; i += 1
        else:
            B = 0 # default value
        # model
        ym = self._nls_model(self.t, t1, w, A=A, B=B)
        return ym - self.y



    def fit_nls_lorentz(self,
        init_t1=None, init_w=0.5, init_A=1.0, init_B=0.0,
        base='e', verbose=True, plot=True
    ):
        """
        Fit ΔP/Ps vs time using NLS model with Lorentzian distribution in ln(t0).
        Parameters
        ----------
        t, y : arrays
            Time (s) and normalized ΔP/Ps (target) of same length.
        init_t1 : float or None
            Initial guess for center time t1 (s). If None, we set to exp(weighted median) from data.
        init_w : float
            Initial HWHM in ln-space (default 0.5).
        init_A, init_B : float
            Initial amplitude and baseline.
        fit_A, fit_B : bool
            Flags to fit amplitude, baseline.
        bounds : dict or None
            Optional parameter bounds as dict with keys 't1','w','A','B', each a (low, high) tuple.
            If None, reasonable defaults are used.
        base : {'e','10'}
            If '10', we interpret init_w as HWHM in log10 and convert to ln internally.
        L, N : int
            Integration span and number of grid points for the numeric integral.
        verbose : bool
            Print fitted parameters and RMSE.
        plot : bool
            Plot data and fitted curve.
        Returns
        -------
        result : dict
            {'t1','w','A','B','rmse','success','y_fit'}
        """
        t = np.asarray(self.t, dtype=float)
        y = np.asarray(self.y, dtype=float)
        assert t.shape == y.shape and t.ndim == 1 and len(t) >= 5, "t and y must be 1D arrays with ≥5 points"
        # Sort by time just in case
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        # Initial guesses
        if init_t1 is None:
            # Use time at which y crosses ~0.5 (or median)
            try:
                idx = np.argmin(np.abs(y - 0.5))
                init_t1 = float(max(min(t[idx], np.max(t)), np.min(t)))
            except Exception:
                init_t1 = float(np.median(t))
        if base == '10':
            # Convert HWHM from log10 to natural log
            init_w = float(init_w) * np.log(10.0)
        # Safety
        init_t1 = float(max(init_t1, 1e-12))
        init_w = float(max(init_w, 1e-6))
        init_A = float(init_A)
        init_B = float(init_B)
        # Parameter vector packing
        names = ['t1','w']
        p0 = [init_t1, init_w]
        if self.fit_A:
            names.append('A'); p0.append(init_A)
        if self.fit_B:
            names.append('B'); p0.append(init_B)
        lb, ub = self._get_bounds()
        res = least_squares(self.residuals, p0, bounds=(lb, ub), method='trf', max_nfev=200)
        # residuals is a function object, not a parameter, a special syntax of optimizer, which overrides the original residuals function
        p_opt = res.x
        success = res.success
        # Unpack optimal parameters, which is the parameters that minimize the residual
        i = 0
        t1_opt = float(p_opt[i]); i+=1
        w_opt  = float(p_opt[i]); i+=1
        if self.fit_A:
            A_opt  = float(p_opt[i]); i+=1
        else:
            A_opt = 1 # default value
        if self.fit_B:
            B_opt  = float(p_opt[i]); i+=1
        else:
            B_opt = 0 # default value
        y_fit = self._nls_model(t, t1_opt, w_opt, A=A_opt, B=B_opt)
        rmse = float(np.sqrt(np.mean((y_fit - y)**2)))
        if verbose:
            # verbose is a flag to determine whether to print detailed information about the fitted parameters and RMSE
            print("=== NLS + Lorentzian Fit (ln-domain) ===")
            print(f"t1 (s)      = {t1_opt:.6g}")
            print(f"w (HWHM ln) = {w_opt:.6g}   -> FWHM(ln) = {2*w_opt:.6g}")
            print(f"A (scale)   = {A_opt:.6g}")
            print(f"B (offset)  = {B_opt:.6g}")
            print(f"RMSE        = {rmse:.6g}")
            # Also report time-domain half-maximum span
            lo = np.exp(np.log(t1_opt) - w_opt)
            hi = np.exp(np.log(t1_opt) + w_opt)
            print(f"Half-height time span: [{lo:.6g} s, {hi:.6g} s] (using ln HWHM)")
        if plot:
            plt.semilogx(t, y, marker='o', markersize=4, label='Data')
            t_dense = np.logspace(np.log10(np.min(t)*0.8), np.log10(np.max(t)*1.2), 400)
            # equally spaced points from log10(np.min(t)*0.8) to log10(np.max(t)*1.2), 400 points
            y_dense = self._nls_model(t_dense, t1_opt, w_opt, A=A_opt, B=B_opt)
            plt.semilogx(t_dense, y_dense, '-', label='Fit')
            # t_dense = logspace(...) is equally spaced points on log scale, which is more visual and numerical properties of semilogx, can clearly show the continuous trend of the model on multiple orders of magnitude.
            # expand the range slightly to [0.8×min, 1.2×max], can avoid the curve just touching the data boundary, which is convenient to observe the trend at both ends
        # store the optimal fitted parameters in the class
        self.t1 = t1_opt
        self.w = w_opt
        self.A = A_opt
        self.B = B_opt
        return dict(t1=t1_opt, w=w_opt, A=A_opt, B=B_opt, rmse=rmse, success=success, y_fit=y_fit)



class plot_and_fit_data:
    def __init__(self):
        self.plot_data_path = NLS_Data_Analysis.select_file()
        # finally decided to use dictionary structure to store data
        self.plot_data = dict()
        self.get_data()


    def get_data(self):
        with h5py.File(self.plot_data_path, 'r') as f:
            storage_data = f['Data']['NLS_Trend']
            row_num , column_num = storage_data.shape
            for i in range(row_num):
                if storage_data[i][0] not in self.plot_data:
                    self.plot_data[storage_data[i][0]] = [(storage_data[i][1], -storage_data[i][2])]
                    # the negative sign is because the data is stored as the change of polarization, so negative delta P indicated positive induced polarization
                else:
                    self.plot_data[storage_data[i][0]].append((storage_data[i][1], -storage_data[i][2]))
            return
