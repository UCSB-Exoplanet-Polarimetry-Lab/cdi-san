from prysm.mathops import np
from prysm.coordinates import make_xy_grid, cart_to_polar

class SpeckleAreaNulling:

    def __init__(self, propagation, dx_img, epd, efl, wvl, dm, IWA, OWA,
                 ref_contrast=1, edge=None, angular_range=[-90, 90],
                 starting_wfe=None):
        """Instance of Speckle Area Nulling

        Parameters
        ----------
        propagation : callable
            function that, when called, returns an intensity image
        dx_img : float
            Image pixel size in microns
        epd : float
            Entrance pupil diameter in milimeters
        efl : float
            Effective focal length in milimeters
        wvl : float
            Wavelength in microns
        dm : DM
            Instance of DM class
        IWA : float
            Inner working angle in lambda / D
        OWA : float
            Outer working angle in lambda / D
        ref_contrast : float, optional
            Reference contrast, used to divide intensity images to get in units of
            normalized intensity, by default 1, which returns values in units of intensity
        edge : float, optional
            A knife-edge limit to the dark hole region in lambda / D, by default None, which
            returns a full dark hole
        angular_range : list, optional
            Angular range in degrees for the dark hole region, by default [-90, 90]

        Returns
        -------
        SpeckleAreaNulling
            Instance of SpeckleAreaNulling class
        """

        self.fwd = propagation
        self.dm = dm
        self.dx_img = dx_img
        self.epd = epd
        self.efl = efl
        self.wvl = wvl
        self.image_dx_lamD = self.dx_img * 1e-3 / (self.efl) * (self.epd) / (self.wvl * 1e-3)
        self.IWA = IWA
        self.OWA = OWA
        self.kvec = 2 * np.pi / wvl
        self.images = []
        self.mean_in_dh = []
        self.dm_surface = []
        self.ref_contrast = ref_contrast
        self.nact = self.dm.Nact
        self.edge = edge
        if starting_wfe is None:
            self.wfe_offset = 0
        else:
            self.wfe_offset = starting_wfe

        # construct a dark hole
        self.Nimg = self.fwd().shape[0]
        self.x, self.y = make_xy_grid(self.Nimg, dx=self.dx_img * 1e-3)

        self.u, self.v = make_xy_grid(self.Nimg, dx=self.image_dx_lamD)
        r, t = cart_to_polar(self.u, self.v)

        self.dh = np.zeros([self.Nimg, self.Nimg])
        self.dh[r < self.OWA] = 1.
        self.dh[r < self.IWA] = 0.

        self.dh[t < angular_range[0]] = 1.
        self.dh[t > angular_range[1]] = 1.

        if self.edge is not None:
            self.dh[self.u < self.edge] = 0.

        # Bulid up the fourier modes in the dark hole region
        kfreq = self.kvec / (self.efl * 1e-3)  # convert wavelength to milimeters
        xfreq = (kfreq * self.x)[self.dh == 1]
        yfreq = kfreq * self.y[self.dh == 1]

        # Make the DM shapes
        xdm, ydm = make_xy_grid(self.nact, dx=self.dm.act_pitch)
        arg = xfreq[..., None, None] * xdm + yfreq[..., None, None] * ydm
        self.cos_modes = np.cos(arg)
        self.sin_modes = np.sin(arg)

        # NOTE This is a spatial frequency fudge factor, something is off
        # by a couple factors of two
        xdm /= 8
        ydm /= 8

        V1sum = np.sum(self.sin_modes, axis=0)
        V2sum = np.sum(self.cos_modes, axis=0)

        V1norm = np.abs(V1sum).max()
        V2norm = np.abs(V2sum).max()
        V1sum /= V1norm
        V2sum /= V2norm

        self.sin_probe = V1sum / 10
        self.cos_probe = V2sum / 10

        self.sin_modes /= V1norm * 10
        self.cos_modes /= V2norm * 10

    def step(self, regularization=0):
        """Advance the algorithm one iteration

        Parameters
        ----------
        regularization : float
            Regularization parameter to minimize the inversion of small signals.
            This is roughly analogous to the change in contrast that you wish to
            ignore.
        """

        # Starting image acquisition
        I0 = self.fwd() / self.ref_contrast

        # Four probe steps
        for probe in [-self.sin_probe, self.sin_probe, -self.cos_probe, self.cos_probe]:
            self.dm.actuators[:] += probe
            I = self.fwd(self.dm.render(wfe=True) + self.wfe_offset) / self.ref_contrast
            self.dm.actuators[:] -= probe
            self.images.append(I)


        Im1 = self.images[-4]  # minus sin probe
        Ip1 = self.images[-3]  # plus sin probe
        Im2 = self.images[-2]  # minus cos probe
        Ip2 = self.images[-1]  # plus cos probe

        self.Im1 = Im1
        self.Im2 = Im2

        # Compute the relevant quantities
        dE1 = (Ip1 - Im1) / 4
        dE2 = (Ip2 - Im2) / 4
        dE1sq = (Ip1 + Im1 - 2*I0) / 2
        dE2sq = (Ip2 + Im2 - 2*I0) / 2

        # Regularized sin / cosine coefficients
        sin_coeffs = dE1 / (dE1sq + regularization)
        cos_coeffs = dE2 / (dE2sq + regularization)


        # Just the ones in the dark hole
        sin_coeffs = sin_coeffs[self.dh==1, None, None]
        cos_coeffs = cos_coeffs[self.dh==1, None, None]

        # apply the correction
        # CONTROL SIGNAL
        correction = sin_coeffs * self.sin_modes + cos_coeffs * self.cos_modes
        correction = np.sum(correction, axis=0)
        self.dm.actuators[:] += correction
        self.dm_surface.append(self.dm.actuators)

        # return an image
        img = self.fwd(self.dm.render(wfe=True) + self.wfe_offset) / self.ref_contrast

        # get mean in dark hole
        self.mean_in_dh.append(np.mean(img[self.dh==1]))

        return img
