# LayoutField
Layout Field is a repository containing software for computing metrics to compare wind farm layouts or sets of layouts. The metrics are based on superimposed two-dimensional Gaussian distributions, which define the *layout fields*. These fields can be superimposed and subtracted to obtain the qualitative metric *Layout Field Error*. Integrating the resulting field yields two quantitative metrics: the *Layout Field Integrated Error* and the *Effective Layout Distance*.

# Related publications
The theory underlying these metrics and their applications are described in the following publications:
- \[1\] M.Baricchio, J. Iori, T.S. Lokken, P.M.O. Gebraad and J.W. van Wingerden. ‘A novel metric to compare wind farm layouts and identify trends in the layout optimization problem’. In: *Journal of Physics: Conference Series* 3224.3 (2026), p. 032020. DOI: [10.1088/1742-6596/3224/3/032020](https://doi.org/10.1088/1742-6596/3224/3/032020)
- \[2\] M.Baricchio, J. Iori, P.M.O. Gebraad and J.W. van Wingerden. ‘What matters for wind farm layout optimization? An assessment based on the effective layout distance’. In: *Wind Energ. Sci. Discuss.* \[preprint\] (2026). in review. DOI: [10.5194/wes-2026-176](https://doi.org/10.5194/wes-2026-176)

# How to use
The code related to the metrics can be found in the file **metrics_layout.py**, and a basic example how to use these functions is made available in the jupyter notebook **example.ipynb** (recommended to get started). 
The folder **examples/wflo** contains the software and data related to \[2\], for which a dedicated README file is made available in the corresponding folder.

# Dependecies
This code requires the python packages numpy (>=2.0.1 recommended), matplotlib (>=3.10.3 recommeded), and scipy (.=1.15.3 recommended).
To run the code included in the folder **examples/wflo** several additional packages are required. All the packages (with their versions) present in the environment adopted for this example are listed in the corresponding folder.


Technische Universiteit Delft hereby disclaims all copyright interest in the
program Layout Field supported by the SUDOCO project, which receives the funding
from the European Union’s Horizon Europe Programme under the grant No. 101122256.
written by the Author(s).
Prof. Dr. Ir. Fred van Keulen, Dean of Mechanical Engineering
