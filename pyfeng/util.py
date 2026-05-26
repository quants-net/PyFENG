import warnings

import numpy as np
from statsmodels.stats.moment_helpers import cum2mc

class MathConsts:
    """

    References:
        https://github.com/lattera/glibc/blob/master/math/math.h
    """

    M_E         = 2.7182818284590452354   # e
    M_LOG2E     = 1.4426950408889634074   # log_2 e
    M_LOG10E    = 0.43429448190325182765  # log_10 e
    M_LN2       = 0.69314718055994530942  # log_e 2
    M_LN10      = 2.30258509299404568402  # log_e 10
    M_PI        = 3.14159265358979323846  # pi
    M_PI_2      = 1.57079632679489661923  # pi/2
    M_PI_4      = 0.78539816339744830962  # pi/4
    M_1_PI      = 0.31830988618379067154  # 1/pi
    M_2_PI      = 0.63661977236758134308  # 2/pi
    M_2_SQRTPI  = 1.12837916709551257390  # 2/sqrt(pi)
    M_SQRT2     = 1.41421356237309504880  # sqrt(2)
    M_SQRT1_2   = 0.70710678118654752440  # 1/sqrt(2)

    # values below are from https://keisan.casio.com/calculator
    M_SQRT2PI   = 2.506628274631000502416  # sqrt(2pi)
    M_1_SQRT2PI = 0.3989422804014326779399  # 1/sqrt(2pi)
    M_SQRT_PI_2 = 1.253314137315500251208  # sqrt(pi/2)
    M_LN2PI_2   = 0.9189385332046727417803  # log(sqrt(2pi)) = log(2pi)/2


class MathFuncs:

    @staticmethod
    def logrel(x):
        """
        Numerically stable computation of ``log(1 + x) / x``.

        This equals the average of ``1/t`` over the interval ``[1, 1+x]``:

        .. math::

            \\frac{1}{x} \\int_1^{1+x} \\frac{1}{t} \\, dt = \\frac{\\ln(1+x)}{x}

        Uses ``np.log1p(x)`` internally to avoid catastrophic cancellation
        when ``x`` is close to zero. The limit as ``x → 0`` is ``1``, which is
        returned exactly for ``x = 0``.

        Args:
            x: argument (scalar or ndarray, any numeric dtype). Must satisfy
               ``x > -1``; raises ``ValueError`` otherwise.

        Returns:
            ``log(1 + x) / x``, with the limiting value ``1.0`` at ``x = 0``.
            Shape and dtype match those of ``x`` (always float).

        Examples:
            >>> MathFuncs.logrel(0.0)           # limit at zero → 1
            1.0
            >>> MathFuncs.logrel(1.0)           # log(2) / 1 ≈ 0.6931
            0.6931471805599453
            >>> MathFuncs.logrel(np.array([0.0, 1.0, -0.5]))
            array([1.        , 0.69314718, 1.38629436])
        """
        if not np.all(x > -1.0):
            raise ValueError("x must be greater than -1.0.")
        rv = np.ones_like(x, dtype=float)
        np.divide(np.log1p(x), x, out=rv, where=(x != 0.0))
        return rv

    @staticmethod
    def powrel(x, a):
        """
        Numerically stable computation of ``((1+x)^a - 1) / (a * x)``.

        Average of ``t^(a-1)`` over the interval ``[1, 1+x]``:

        .. math::

            \\frac{1}{x} \\int_1^{1+x} t^{a-1} \\, dt =
            \\begin{cases}
            \\dfrac{(1+x)^a - 1}{a\\,x} & a \\neq 0 \\\\[6pt]
            \\dfrac{\\ln(1+x)}{x}       & a = 0
            \\end{cases}

        The case ``a = 0`` reduces to :meth:`logrel`. In all cases the limit
        as ``x → 0`` is ``1``, which is returned exactly for ``x = 0``.

        Args:
            x: argument (scalar or ndarray, any numeric dtype). Must satisfy
               ``x > -1``; raises ``ValueError`` otherwise.
            a: power exponent (scalar or ndarray broadcastable with ``x``).

        Returns:
            ``((1+x)^a - 1) / (a * x)``, with the limiting value ``1.0``
            at ``x = 0``. Shape is broadcast shape of ``x`` and ``a``, dtype
            is always ``float``.

        Examples:
            >>> MathFuncs.powrel(0.0, 2.0)      # limit at zero → 1
            1.0
            >>> MathFuncs.powrel(1.0, 3.0)      # (2^3 - 1) / (3 * 1) = 7/3 ≈ 2.3333
            2.3333333333333335
            >>> MathFuncs.powrel(1.0, 0.0)      # reduces to logrel: log(2) ≈ 0.6931
            0.6931471805599453
        """
        if not np.all(x > -1.0):
            raise ValueError("x must be greater than -1.0.")
        rv = np.ones(np.broadcast_shapes(np.shape(x), np.shape(a)), dtype=float)
        np.divide(np.expm1(a * np.log1p(x)), a * x, out=rv, where=(x != 0.0) & (a != 0.0))
        np.divide(np.log1p(x), x, out=rv, where=(x != 0.0) & (a == 0.0))
        return rv

    @staticmethod
    def rational_dg85(x, xlim, f0, f1, f2=None, mode='left'):
        """Shape-preserving rational-cubic interpolation (Delbourgo & Gregory, 1985).

        Interpolates a function on the interval [x_l, x_r] = xlim matching the
        endpoint values f0 = [y_l, y_r] and first derivatives f1 = [d_l, d_r].
        With s = (x - x_l)/h, h = x_r - x_l, cs = 1 - s, the interpolant is

            g = [ y_r*s^3 + (r*y_r - h*d_r)*s^2*cs
                  + (r*y_l + h*d_l)*s*cs^2 + y_l*cs^3 ] / [ 1 + (r-3)*s*cs ],

        a rational cubic with a single shape parameter r [DG85 eqs (2.3)-(2.5)].
        It is exact at the endpoints (g=y_l, y_r) with the prescribed slopes
        (g'=d_l, d_r). The denominator is positive (pole-free) iff r > -1
        [DG85 (2.7)]; for monotone data the interpolant is monotone if
        r >= (d_l + d_r)/Delta [DG85 (3.8)].

        r is set by `mode`:
          'left'     : match f''(x_l) = f2  [DG85 (3.15)-(3.16) at theta=0]
                       -> r = (h/2*f2 + d_r - d_l)/(Delta - d_l)
          'right'    : match f''(x_r) = f2  [DG85 (3.15)-(3.16) at theta=1]
                       -> r = (h/2*f2 + d_r - d_l)/(d_r - Delta)
          'monotone' : r = 1 + (d_l + d_r)/Delta
                       [DG85 (3.9); reduces to the rational quadratic (3.10)]
          'convex'   : r = 1 + M/m + m/M,  M = max{d_r-Delta, Delta-d_l},
                       m = min{d_r-Delta, Delta-d_l}            [DG85 (3.21)]
        where Delta = (y_r - y_l)/h is the secant slope; f2 is used only by
        'left'/'right'. An infinite endpoint slope d_l gives r = 1 ('left') or
        r = -inf ('right').

        Emits a RuntimeWarning if r <= -1 (denominator pole inside the interval)
        or if r < (d_l+d_r)/Delta with endpoint slopes consistent with the
        secant (monotonicity not guaranteed).

        Parameters
        ----------
        x : float or array       evaluation point(s) in [x_l, x_r]
        xlim : (x_l, x_r)        interval endpoints
        f0 : (y_l, y_r)          function values at the endpoints
        f1 : (d_l, d_r)          first derivatives at the endpoints
        f2 : float, optional     target second derivative (used only by mode 'left'/'right')
        mode : str               'left', 'right', 'monotone', or 'convex' (see above)

        Returns
        -------
        Interpolated value(s) g(x).

        Reference
        ---------
        Delbourgo R, Gregory JA (1985) Shape preserving piecewise rational
        interpolation. SIAM Journal on Scientific and Statistical Computing
        6:967-976. https://doi.org/10.1137/0906065
        """

        h = xlim[1] - xlim[0]
        s = (x - xlim[0])/h

        f1av = (f0[1] - f0[0])/h
        if mode == 'left':                              # match f''(x_l) [DG85 (3.15) @ theta=0]
            r = 1.0 if f1[0] == np.inf else (0.5*h*f2 + f1[1] - f1[0])/(f1av - f1[0])
        elif mode == 'right':                           # match f''(x_r) [DG85 (3.15) @ theta=1]
            r = -np.inf if f1[0] == np.inf else (0.5*h*f2 + f1[1] - f1[0])/(f1[1] - f1av)
        elif mode == 'monotone':                        # DG85 (3.9): rational quadratic (3.10)
            r = 1.0 + (f1[0] + f1[1])/f1av
        elif mode == 'convex':                          # DG85 (3.21)
            M = max(f1[1] - f1av, f1av - f1[0])
            m = min(f1[1] - f1av, f1av - f1[0])
            r = 1.0 + M/m + m/M
        else:
            raise ValueError(f"rational_dg85: unknown mode={mode!r}; "
                             f"use 'left', 'right', 'monotone', or 'convex'")

        if r <= -1.0:                                   # DG85 (2.7): denominator pole
            warnings.warn(
                f"rational_dg85: r={r:.4g} <= -1; denominator vanishes in (x_l, x_r) "
                f"[DG85 (2.7)]", RuntimeWarning)
        elif np.isfinite(f1[0]) and np.isfinite(f1[1]) and f1av != 0.0 \
                and f1[0]*f1av >= 0.0 and f1[1]*f1av >= 0.0:   # slopes consistent w/ secant
            thr = (f1[0] + f1[1]) / f1av                # DG85 (3.8) monotonicity bound
            if r < thr:
                warnings.warn(
                    f"rational_dg85: r={r:.4g} < (d_l+d_r)/Delta={thr:.4g}; "
                    f"monotonicity not guaranteed [DG85 (3.8)]", RuntimeWarning)

        rv = (f0[1]*s + (r*f0[1] - h*f1[1])*(1-s))*s**2 + ((r*f0[0] + h*f1[0])*s + f0[0]*(1-s))*(1-s)**2
        rv /= 1 + (r-3)*s*(1-s)

        return rv


class StatFuncs:

    def gramcharA_coefs(
            cumuls: np.ndarray,
            sigma: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Gram-Charlier A series coefficients in terms of standardised variable z.

        The expansion is:
            f(z) = phi(z) * sum_n  c[n] * He_{n+1}(z)

        where z = (x - mu) / sigma and phi(z) is the standard normal density.

        Convention: cumuls[k] = cumul_{k+1}, k = 0, ..., m-1
            cumuls[0] = cumul_1  (mean,     absorbed into mu)
            cumuls[1] = cumul_2  (variance, used for sigma)
            cumuls[2] = cumul_3  (skewness driver)
            ...

        Output: c[k] = tilde_c_{k+1}, k = 0, ..., m-1
            c[0] = 0                           (c_1, centering)
            c[1] = 0                           (c_2, centering)
            c[2] = cumul_3 / (6 sigma^3)       (c_3)
            c[3] = cumul_4 / (24 sigma^4)      (c_4)
            ...
            c[k] = cumul_{k+1} / ((k+1)! sigma^{k+1})

        Parameters
        ----------
        cumuls : np.ndarray, shape (m,)
            Cumulants. cumuls[k] = cumul_{k+1}.
        sigma  : float or None
            Defaults to sqrt(cumuls[1]) = sqrt(cumul_2).

        Returns
        -------
        c     : np.ndarray, shape (m,)
            Gram-Charlier coefficients in z-space.
        sigma : float
            Sigma actually used.
        """

        m = len(cumuls)

        # --- Resolve sigma ---
        if sigma is None:
            if cumuls[1] <= 0:
                raise ValueError(f"cumul_2 = cumuls[1] = {cumuls[1]} must be positive.")
            sigma = np.sqrt(cumuls[1])

        # --- sigma powers: sigma_pow[k] = sigma^{-(k+1)} ---
        cumul_std = np.cumprod(np.full(m, 1/sigma))
        # --- Standardised excess cumulants for cum2mc ---
        cumul_std[0:2] = 0.0
        cumul_std[2:] *= cumuls[2:]

        # --- B[k] = 1/(k+1)! ---
        c = 1.0/np.cumprod(np.arange(1, m + 1, dtype=float))
        # --- Bell polynomials via cum2mc: returns [B_1, ..., B_m] ---
        # --- c[k] = B[k] / (k+1)!  (no sigma division in z-space) ---
        c *= np.array(cum2mc(cumul_std))

        return c, sigma
