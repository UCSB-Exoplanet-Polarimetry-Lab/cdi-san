# Local imports from cdi-san
from models import KeckTelescope
from san import SpeckleAreaNulling
from influenc_funcs import gaussian_influence_function

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from prysm.interferogram import render_synthetic_surface
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.x.dm import DM

keck = KeckTelescope(center_wavelength=1.65, # microns
                     bandwidth=1)           # percent

# Detector Quantities can be set here
keck.dark_current = 5 # electrons / s
keck.read_noise = 1 # electrons / s
keck.bias = 10 # electrons
keck.bits = 16 # bits
keck.full_well_capacity = 2 ** keck.bits # bit depth
keck.conversion_gain = 1 # electrons / DN
keck.exposure_time = 1000 # seconds

plt.figure()
plt.title("Dark Frame")
plt.imshow(keck.get_dark_image(), cmap="gray")
plt.colorbar()
plt.show()

# Set up wavefront error - ideally this doesn't change
x, y, z = render_synthetic_surface(10, keck.Npup, a=5e4, b=1/1000, c=3)
wavefront_error = z

# Here for normalization, not strictly necessary
ref = keck.get_coronagraph_image(include_fpm=False).max()

plt.figure(figsize=[15, 4])
plt.subplot(131)
plt.title("Entrance Pupil")
plt.imshow(keck.get_entrance_pupil() * z, cmap="gray")
plt.colorbar()
plt.subplot(132)
plt.title("Direct Image H-band")
plt.imshow(keck.get_direct_image() / ref, cmap="inferno", norm=LogNorm())
plt.colorbar()
plt.subplot(133)
plt.title("Coronagraphic Image H-band")
plt.imshow(keck.get_coronagraph_image(include_fpm=True, wfe=wavefront_error) / ref, cmap="inferno", norm=LogNorm())
plt.colorbar(label="Normalized Intensity")
plt.show()


# Set up a deformable mirror
nact = 22
act_pitch = 3
samples_per_act = 21
sampling_pitch = act_pitch / samples_per_act

x, y = make_xy_grid(keck.Npup, dx=keck.dx_pup)
r, th = cart_to_polar(x, y)
influence_func = gaussian_influence_function(r, act_pitch)
Nout = x.shape[0]
dm = DM(influence_func, Nout, nact, samples_per_act, rot=(0,0,0), shift=(0,0))
dm.act_pitch = act_pitch


# Try out running SAN
san = SpeckleAreaNulling(
    propagation=keck.get_coronagraph_image,
    dx_img=keck.dx_img,
    epd=keck.epd,
    efl=keck.efl,
    wvl=keck.center_wavelength,
    dm=dm,
    IWA=4,
    OWA=7,
    angular_range=[-85, 85],
    starting_wfe=wavefront_error
)

NSTEPS = 5
corrected_images = []
for i in range(NSTEPS):
    img = san.step()
    corrected_images.append(img)
    plt.figure()
    plt.imshow(img, cmap="inferno", norm=LogNorm())
    plt.colorbar()
    plt.show()
