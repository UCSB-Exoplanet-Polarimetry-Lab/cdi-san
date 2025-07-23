# cdi-san
Project to organize code for CDI-SAN effort at UCSB

[Document](./docs/7.2.pdf) summarizing the CDI-SAN effort. Contains azimuthal intensity profiles comparing the relative gains achieved by SAN wavefront control and CDI-SAN post-processing.

## Kalman Filter

[Document](docs/Kalman_Filter_SAN.pdf) describing the process of using Kalman filter as SAN regularization. Page 3 includes a proposed noise term computation.

## Installation
The SAN simulator is built on the latest version of the `prysm` optical propagation package. To install, first clone from source and then pip install.

```bash
git clone https://github.com/brandondube/prysm
cd prysm
pip install -e .
```
