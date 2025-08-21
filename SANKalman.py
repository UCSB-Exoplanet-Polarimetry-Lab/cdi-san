import numpy as np

# define a safe division to avoid error
TOL = 1e-12
safe_div = lambda num, den: np.divide(
    num, den,
    out=np.zeros_like(num, dtype=float),
    where=np.abs(den) > TOL
)

BIG_VAR = 1e10    # big covariance for zero denominator

class SANKalman:
    """
    Vectorized Kalman filter for Speckle-Area Nulling (SAN).

    Each pixel inside the supplied dark-hole mask gets its own (p, q) state.
    """

    def __init__(self, mask: np.ndarray, gain: float, q_process: float = 1e-3):
        """
        Parameters
        ----------
        mask : bool 2-D array
            True for pixels to be controlled (dark hole), False elsewhere.
        gain : float
            Detector gain G [electrons / ADU].
        q_process : float
            Process-noise from the DM.
        """
        self.mask = mask
        self.npix = mask.sum()
        self.G = gain
        self.Q = np.eye(2) * q_process     # same Q for every pixel

        # State:   shape == (2, Npix)
        self.x = np.zeros((2, self.npix))
        # State cov: one 2×2 block per pixel -> (Npix, 2, 2)
        self.P = np.zeros((self.npix, 2, 2))


    # method to build mask
    @staticmethod
    def darkhole_mask(nside: int,                     # image is nside × nside
                      inner_r: float, outer_r: float, # radii in pixels
                      theta_start: float = -np.pi/2.1, theta_end: float = np.pi/2.1):
        """
        Creates a semi-annular mask and only keep pixels within.
        """
        y, x = np.indices((nside, nside)) - nside / 2.0
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        thetamask = (theta >= theta_start) & (theta <= theta_end)
        annulus = (r >= inner_r) & (r <= outer_r)
        return annulus & thetamask


    def initialize(self,
                   I0, I1p, I1m, I2p, I2m,
                   corner_slice=(slice(0, 20), slice(0, 20)),
                   rigorous=False, Ns=None, Nd=None, Nr=None):
        """
        Build measurement noise R and inital state covariance P_0 from the first five-frame burst.

        Parameters
        ----------
        I0, I1p, I1m, I2p, I2m, : ndarray
            The 5 modulation images fron SAN.
        corner_slice
            the corner that will be extracted from `I0` to calculate measurement noise.
        rigorous : bool
            If True, supply `Ns, Nd, Nr` (in electrons/photons).
            Otherwise the function uses the dark-corner trick.
        Ns : float
            the total number of photons/pixel from background or sky.
        Nd : float
            the total number of dark current electrons/pixel.
        Nr : float
            the total number of electrons/pixel from read noise.
        """
        m = self.mask
        # ----- per-pixel measurement variances V_I (ADU²) -----
        if rigorous:
            V_I0  = I0[m]/self.G + (Ns + Nd + Nr**2)/self.G**2
            V_I1p = I1p[m]/self.G + (Ns + Nd + Nr**2)/self.G**2
            V_I1m = I1m[m]/self.G + (Ns + Nd + Nr**2)/self.G**2
            V_I2p = I2p[m]/self.G + (Ns + Nd + Nr**2)/self.G**2
            V_I2m = I2m[m]/self.G + (Ns + Nd + Nr**2)/self.G**2
        else:
            corner = corner_slice
            V_dark = np.var(I0[corner])
            V_I0  = I0[m]/self.G + V_dark
            V_I1p = I1p[m]/self.G + V_dark
            V_I1m = I1m[m]/self.G + V_dark
            V_I2p = I2p[m]/self.G + V_dark
            V_I2m = I2m[m]/self.G + V_dark

        # ----- p, q -----
        Np = I1p[m] - I1m[m]
        Dp = 2*(I1p[m] + I1m[m] - 2*I0[m])
        Nq = I2p[m] - I2m[m]
        Dq = 2*(I2p[m] + I2m[m] - 2*I0[m])

        p_raw = safe_div(Np , Dp)
        q_raw = safe_div(Nq , Dq)
        self.x[0, :] = p_raw
        self.x[1, :] = q_raw

        # ----- covariances -----
        sig2_p = np.full_like(Dp, BIG_VAR, dtype=float)
        sig2_q = np.full_like(Dq, BIG_VAR, dtype=float)

        good = np.abs(Dp) > TOL
        if np.any(good):
            d_p_I1p = (Dp - 2*Np)/Dp**2
            d_p_I1m =-(Dp + 2*Np)/Dp**2
            d_p_I0  = 4*Np/Dp**2
            sig2_p[good] = (d_p_I1p[good]**2)*V_I1p[good] \
                        + (d_p_I1m[good]**2)*V_I1m[good] \
                        + (d_p_I0 [good]**2)*V_I0 [good]

        good = np.abs(Dq) > TOL
        if np.any(good):
            d_q_I2p = (Dq - 2*Nq)/Dq**2
            d_q_I2m =-(Dq + 2*Nq)/Dq**2
            d_q_I0  = 4*Nq/Dq**2
            sig2_q[good] = (d_q_I2p[good]**2)*V_I2p[good] \
                        + (d_q_I2m[good]**2)*V_I2m[good] \
                        + (d_q_I0 [good]**2)*V_I0 [good]


        # ----- build R (Npix, 2, 2) -----
        self.R = np.zeros_like(self.P)
        self.R[:, 0, 0] = sig2_p
        self.R[:, 1, 1] = sig2_q

        # ----- build P0 -----
        a = 10.0
        self.P[:, 0, 0] = a * 1e-8
        self.P[:, 1, 1] = a * 1e-8


    '''def predict(self, control = True):
        """
        One Kalman *predict* step.

        If no control signal is being sent from last iteration,
        `control` should be set to False.
        """
        if control:
            self.x[:] = 0 
        self.P += self.Q'''
    
    def predict(self, u=None):
        """
        One Kalman predict step with optional control input u.
        u: array-like of shape (2, Npix) or None. State evolves as x <- x + gamma * u.
        """
        if u is not None:
            u = np.asarray(u)
            if u.shape == (2, self.npix):
                self.x += u
            else:
                raise ValueError(f"u must have shape (2, Npix) = (2, {self.npix}), got {u.shape}")
        self.P += self.Q


    def update(self, I0, I1p, I1m, I2p, I2m):
        """One Kalman *update* step using the measured 5-frame burst."""
        m = self.mask

        Np = I1p[m] - I1m[m]
        Dp = 2*(I1p[m] + I1m[m] - 2*I0[m])
        Nq = I2p[m] - I2m[m]
        Dq = 2*(I2p[m] + I2m[m] - 2*I0[m])

        good_p = np.abs(Dp) > TOL
        good_q = np.abs(Dq) > TOL
        z = np.zeros((2, self.npix), dtype=float)
        z[0, good_p] = Np[good_p] / Dp[good_p]
        z[1, good_q] = Nq[good_q] / Dq[good_q]

        # ----- Kalman gain -----
        denom11 = self.P[:, 0, 0] + self.R[:, 0, 0]
        denom22 = self.P[:, 1, 1] + self.R[:, 1, 1]
        K11 = self.P[:, 0, 0] / denom11
        K22 = self.P[:, 1, 1] / denom22

        # ----- innovation & state update -----
        y0 = z[0] - self.x[0]         
        y1 = z[1] - self.x[1]
        self.x[0] += K11 * y0
        self.x[1] += K22 * y1

        # ----- covariance update -----
        self.P[:, 0, 0] *= (1.0 - K11)
        self.P[:, 1, 1] *= (1.0 - K22)

        # innovation for residual analysis
        self.last_innovation = np.vstack((y0, y1)) 



# SANKalman.py v2
import numpy as np

TOL   = 1e-10
BIG_R = 1e12

def _safediv(num, den):
    return np.divide(num, den, out=np.zeros_like(num, float), where=np.abs(den) > TOL)

class KFSAN:
    """
    Kalman filter for SAN basis, per-pixel state x = [p, q].
    Shapes:
      x: (2, N), P,Q,R: (2, N). gamma can be scalar or shape (2,).
    Variance handling follows your first class (with corrected VI0 terms).
    """

    def __init__(self, mask, q_process=1e-4, init_var=1.0, gamma=1.0, gate_sigma=5.0, gain=1.0):
        m = np.asarray(mask, bool)
        self.pix_inds = np.flatnonzero(m.ravel())
        self.npix = self.pix_inds.size

        self.x = np.zeros((2, self.npix), float)           # [p, q]
        self.P = np.full((2, self.npix), init_var, float)  # diag per-pixel
        self.Q = np.full((2, self.npix), q_process, float)
        self.R = np.ones((2, self.npix), float)            # filled per burst

        self.gamma = gamma                                  # scalar or (2,)
        self.gate_sigma = float(gate_sigma)
        self.last_innovation = None

        # NEW: detector gain (electrons/ADU), used to build frame variances
        self.G = float(gain)

        # NEW: noise model config (populated in initialize_from_burst)
        self._noise_cfg = dict(rigorous=False, Ns=None, Nd=None, Nr=None,
                               corner_slice=(slice(0, 20), slice(0, 20)))

    # ---------- helpers ----------
    def _extract(self, img):
        arr = np.asarray(img, float).ravel()
        return arr[self.pix_inds]

    def _pq(self, I0, I1p, I1m, I2p, I2m):
        """SAN algebra (no stray 0.5):
           den = 2*(I+ + I- - 2*I0) = 4|ΔE|^2 ;  z = (I+ - I-) / den
        """
        I0  = self._extract(I0)
        I1p = self._extract(I1p); I1m = self._extract(I1m)
        I2p = self._extract(I2p); I2m = self._extract(I2m)

        Np = I1p - I1m
        Nq = I2p - I2m
        Dp = 2.0 * (I1p + I1m - 2.0 * I0)
        Dq = 2.0 * (I2p + I2m - 2.0 * I0)

        p = _safediv(Np, Dp)
        q = _safediv(Nq, Dq)
        good_p = np.abs(Dp) > TOL
        good_q = np.abs(Dq) > TOL
        return p, q, Dp, Dq, good_p, good_q, (Np, Nq), (I0, I1p, I1m, I2p, I2m)

    # NEW: build per-pixel frame variances following your first class
    def _frame_variances(self, I0, Iplus, Iminus):
        """Return (V_I0, V_Iplus, V_Iminus) on DH pixels in ADU^2."""
        cfg = self._noise_cfg
        G = self.G

        I0m   = self._extract(I0)
        Iplusm= self._extract(Iplus)
        Iminusm=self._extract(Iminus)

        if cfg["rigorous"]:
            Ns, Nd, Nr = cfg["Ns"], cfg["Nd"], cfg["Nr"]
            if Ns is None or Nd is None or Nr is None:
                raise ValueError("rigorous=True but Ns/Nd/Nr not provided")
            const = (Ns + Nd + Nr**2) / (G**2)         # ADU^2
            V_I0   = I0m   / G + const
            V_Ipl  = Iplusm/ G + const
            V_Imin = Iminusm/ G + const
        else:
            # dark-corner estimate of additive variance
            sl = cfg.get("corner_slice", (slice(0, 20), slice(0, 20)))
            # use raw I0 to compute corner variance in ADU^2
            V_dark = float(np.var(np.asarray(I0)[sl], dtype=float))
            V_I0   = I0m   / G + V_dark
            V_Ipl  = Iplusm/ G + V_dark
            V_Imin = Iminusm/ G + V_dark

        return V_I0, V_Ipl, V_Imin

    # CHANGED: pass V arrays (not intensities) into the derivative-based var
    def _var_from_frames(self, N, D, good, V_I0, V_Iplus, V_Iminus):
        """Var[z], z=N/D with additive intensity noise on frames.
           dN=(+1,-1,0), dD=(+2,+2,-4).
        """
        sig2 = np.full_like(D, BIG_R, float)
        if not np.any(good):
            return sig2

        g  = good
        Dg = D[g]; Ng = N[g]
        common = 1.0 / (Dg**2)

        # ∂z/∂I+:  (D - 2N)/D^2 ; ∂z/∂I-: (-D - 2N)/D^2 ; ∂z/∂I0: (4N)/D^2
        dZ_dIpl  = ( Dg*1.0  - 2.0*Ng) * common
        dZ_dImin = (-Dg*1.0  - 2.0*Ng) * common
        dZ_dI0   = ( 4.0*Ng)           * common

        # use PROVIDED variances (ADU^2), not raw intensities
        sig2[g] = (dZ_dIpl**2)  * V_Iplus[g] + \
                  (dZ_dImin**2) * V_Iminus[g] + \
                  (dZ_dI0**2)   * V_I0[g]
        return np.maximum(sig2, 1e-12)

    # ---------- public API ----------
    def initialize_from_burst(self, I0, I1p, I1m, I2p, I2m,
                              rigorous=False, Ns=None, Nd=None, Nr=None,
                              corner_slice=(slice(0, 20), slice(0, 20))):
        """Initialize x and P from first burst, and store noise model config."""
        # store noise-model configuration for future updates
        self._noise_cfg.update(dict(rigorous=rigorous, Ns=Ns, Nd=Nd, Nr=Nr,
                                    corner_slice=corner_slice))

        p, q, Dp, Dq, gp, gq, (Np, Nq), (mI0, mI1p, mI1m, mI2p, mI2m) = \
            self._pq(I0, I1p, I1m, I2p, I2m)

        self.x[0], self.x[1] = p, q

        # build per-frame variances on DH pixels (ADU^2)
        V_I0_p, V_I1p, V_I1m = self._frame_variances(I0, I1p, I1m)
        V_I0_q, V_I2p, V_I2m = self._frame_variances(I0, I2p, I2m)

        # per-component measurement variances using derivative propagation
        self.R[0] = self._var_from_frames(Np, Dp, gp, V_I0_p, V_I1p, V_I1m)
        self.R[1] = self._var_from_frames(Nq, Dq, gq, V_I0_q, V_I2p, V_I2m)

        # initialize P ~ R (avoid overconfident start)
        self.P[:] = np.maximum(self.R, 1e-6)

    def predict(self, u=None):
        """x <- x + Γ ⊙ u ;  P <- P + Q. Γ ok as scalar or (2,)."""
        self.P += self.Q
        if u is None:
            return
        u = np.asarray(u, float)
        if u.shape != (2, self.npix):
            raise ValueError(f"u must have shape (2,{self.npix}), got {u.shape}")

        g = self.gamma
        if np.isscalar(g):
            self.x += g * u
        elif np.ndim(g) == 1 and g.shape == (2,):
            self.x += g[:, None] * u
        else:
            raise ValueError("gamma must be scalar or shape (2,)")

    def update_with_burst(self, I0, I1p, I1m, I2p, I2m):
        """H=I, Joseph covariance update, innovation gating. R from frame variances."""
        p, q, Dp, Dq, gp, gq, (Np, Nq), (mI0, mI1p, mI1m, mI2p, mI2m) = \
            self._pq(I0, I1p, I1m, I2p, I2m)
        z = np.vstack((p, q))

        # rebuild per-frame variances for THIS burst (ADU^2)
        V_I0_p, V_I1p, V_I1m = self._frame_variances(I0, I1p, I1m)
        V_I0_q, V_I2p, V_I2m = self._frame_variances(I0, I2p, I2m)

        # component-wise measurement variances
        R0 = self._var_from_frames(Np, Dp, gp, V_I0_p, V_I1p, V_I1m)
        R1 = self._var_from_frames(Nq, Dq, gq, V_I0_q, V_I2p, V_I2m)

        # innovation and cov
        y0 = z[0] - self.x[0]
        y1 = z[1] - self.x[1]
        S0 = self.P[0] + R0
        S1 = self.P[1] + R1

        # gate wild ratios (equivalent to K=0 on those pixels)
        gate0 = self.gate_sigma * np.sqrt(np.maximum(S0, 1e-12))
        gate1 = self.gate_sigma * np.sqrt(np.maximum(S1, 1e-12))
        y0 = np.clip(y0, -gate0, gate0)
        y1 = np.clip(y1, -gate1, gate1)

        # Kalman gain
        K0 = np.divide(self.P[0], S0, out=np.zeros_like(S0), where=S0 > 0)
        K1 = np.divide(self.P[1], S1, out=np.zeros_like(S1), where=S1 > 0)

        # state update
        self.x[0] += K0 * y0
        self.x[1] += K1 * y1

        # Joseph-form covariance (H=I): P <- (I-K)P(I-K) + K R K
        self.P[0] = (1.0 - K0)**2 * self.P[0] + (K0**2) * R0
        self.P[1] = (1.0 - K1)**2 * self.P[1] + (K1**2) * R1

        # store last innovation (for diagnostics)
        self.last_innovation = np.vstack((y0, y1))

    def control(self, loop_gain=0.2, clip=5.0):
        """u = -g * x  (minus sign!). Returns shape (2, Npix)."""
        u = -float(loop_gain) * self.x.copy()
        return np.clip(u, -clip, clip) if clip is not None else u


