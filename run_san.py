# Local imports from cdi-san
from models import KeckTelescope
from san import SpeckleAreaNulling
from influenc_funcs import gaussian_influence_function
import ipdb
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from prysm.interferogram import render_synthetic_surface
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.x.dm import DM

# Set up wavefront error - ideally this doesn't change
Npup = 1024
x, y, z = render_synthetic_surface(10, Npup, a=5e4, b=1/1000, c=3)
wavefront_error = z / 2

keck = KeckTelescope(center_wavelength=1.65, # microns
                     bandwidth=1, # percent
                     starting_wfe=wavefront_error)

# Detector Quantities can be set here
keck.dark_current = 5 # electrons / s
keck.read_noise = 1 # electrons / s
keck.bias = 10 # electrons
keck.bits = 16 # bits
keck.full_well_capacity = 2 ** keck.bits # bit depth
keck.conversion_gain = 1 # electrons / DN
keck.exposure_time = 100 # seconds

plt.figure()
plt.title("Dark Frame")
plt.imshow(keck.get_dark_image(), cmap="gray")
plt.colorbar()

# Here for normalization, not strictly necessary
ref = 1

plt.figure(figsize=[15, 4])
plt.subplot(131)
plt.title("Entrance Pupil")
plt.imshow(keck.get_entrance_pupil() * z, cmap="gray")
plt.colorbar()
plt.subplot(132)
plt.title("Direct Image H-band")
plt.imshow(keck.get_direct_image() / ref, cmap="inferno")
plt.colorbar()
plt.subplot(133)
plt.title("Coronagraphic Image H-band")
plt.imshow(keck.get_coronagraph_image(include_fpm=True, wfe=wavefront_error) / ref, cmap="inferno", norm=LogNorm())
plt.colorbar(label="Normalized Intensity")


# Set up a deformable mirror
nact = 22
act_pitch = 3
samples_per_act = 42
sampling_pitch = act_pitch / samples_per_act

x, y = make_xy_grid(keck.Npup, dx=sampling_pitch)
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
)

plt.figure()
plt.subplot(121)
plt.title("Sin Probe")
plt.imshow(san.sin_probe, cmap="RdBu_r")
plt.subplot(122)
plt.title("Cos Probe")
plt.imshow(san.cos_probe, cmap="RdBu_r")

NSTEPS = 5
corrected_images = []
mean_in_dh = []
for i in range(NSTEPS):
    img = san.step(regularization=100000)
    corrected_images.append(img)
    mean_in_dh.append(np.mean(img[san.dh==1]))
    plt.figure(figsize=[12, 4])
    plt.suptitle(f"Iteration {i+1}")
    plt.subplot(131)
    plt.imshow(san.Im1, cmap="coolwarm")
    plt.contour(san.dh, levels=[0,5], colors="w")
    plt.subplot(132)
    plt.imshow(san.Im2, cmap="coolwarm")
    plt.contour(san.dh, levels=[0,5], colors="w")
    plt.subplot(133)
    plt.imshow(img, norm=LogNorm())
    plt.contour(san.dh, levels=[0,5], colors="w")

plt.figure()
plt.plot(mean_in_dh, marker="o")
plt.ylabel("Intensity, arb. units")
plt.title("Mean in Dark Hole v.s. Iteration")
plt.show()
