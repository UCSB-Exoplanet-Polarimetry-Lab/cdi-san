import numpy as np

# define a safe division to avoid error
TOL = 1e-12
safe_div = lambda num, den: np.divide(
    num, den,
    out=np.zeros_like(num, dtype=float),
    where=np.abs(den) > TOL
)

BIG_VAR = 1e30    # big covariance for zero denominator

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
        a = 15.0
        self.P[:, 0, 0] = a * sig2_p
        self.P[:, 1, 1] = a * sig2_q


    def predict(self, control = True):
        """
        One Kalman *predict* step.

        If no control signal is being sent from last iteration,
        `control` should be set to False.
        """
        if control:
            self.x[:] = 0 
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
