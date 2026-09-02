# The Python package *cssr*

*cssr* is a Python package that functions as a Compressed Sensing-based super-resolution framework of 
(primarily) low-pass filtered and noisy ground truth signals which admit a sparse representation in some domain. 

An elmentary tutorial and a dedicated scanning-tunneling-microcopy-based tutorial are avilable in the 
directory [tutorials](tutorials/). A speciallized introduction to the underlying concepts can be found in the following publication

> S. Gazit, A. Szameit, Y. C. Eldar, and M. Segev,
  *Super-resolution and reconstruction of sparse sub-wavelength images*,
  [Optics Express, vol. 17, no. 26, pp. 23920–23946](https://doi.org/10.1364/OE.17.023920) (2009)


#### Table of Contents 
- [The Python package cssr](#the-python-package-cssr)
    - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project status](#project-status)
  - [Limitations](#limitations)
  - [References](#references)


## Overview
The *cssr* package is as a sparse recovery framework for signals that have been subject to 
physics-imposed resolution limitations and noise corruption, and is based on a collection of various 
algorithms (cf. [References](#references)). Its central components include:
- (overcomplete) sparsifying dictionaries
- (low-pass) filtering operators
- (optimized) measurement matrics
- sparse recovery algorithms


## Project status
The *cssr* package was originally developed as part of a BSc thesis project for scanning tunneling 
microscopy to overcome thermal and instrumental broadening (+ noise). It is currently in an alpha 
stage and can best be understood as a computational prototype for academic and exploratory use.


## Limitations
The *cssr* package is still under development and, because the limitations have not yet been fully 
determined, released only as a alpha version. While additional testing, documentation, and robustness 
checks are required, the package is already fully functional and documented through docstrings. 


## License
This project is distributed for academic and research use. See the LICENSE file for the full licensing 
details. *cssr* is free software; *cssr* is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY, see the [MIT licence](LICENSE) for more details.


## References
[1] S. S. Chen, D. L. Donoho, and M. A. Saunders, "Atomic decomposition by basis pursuit," SIAM Review, vol. 
43, no. 1, pp. 129–159, 2001, 
doi: [10.1137/S003614450037906X](https://doi.org/10.1137/S003614450037906X).

[2] P. R. Gill, A. Wang and A. Molnar, "The In-Crowd Algorithm for Fast Basis Pursuit Denoising," in IEEE 
Transactions on Signal Processing, vol. 59, no. 10, pp. 4595-4605, Oct. 2011, 
doi: [10.1109/TSP.2011.2161292](https://doi.org/10.1109/TSP.2011.2161292). 

[3] S. Gazit, A. Szameit, Y. C. Eldar, and M. Segev, "Super-resolution and reconstruction of sparse 
sub-wavelength images," Optics Express, vol. 17, no. 26, pp. 23920–23946, 2009, 
doi: [10.1364/OE.17.023920](https://doi.org/10.1364/OE.17.023920).

[4] M. Tan, I. W. Tsang and L. Wang, "Matching Pursuit LASSO Part I: Sparse Recovery Over Big Dictionary," 
in IEEE Transactions on Signal Processing, vol. 63, no. 3, pp. 727-741, Feb.1, 2015, 
doi: [10.1109/TSP.2014.2385036](https://doi.org/10.1109/TSP.2014.2385036). 

[5] V. Abolghasemi, S. Ferdowsi, and S. Sanei, "A gradient-based alternating minimization approach for optimization 
of the measurement matrix in compressive sensing," Signal Processing, vol. 92, no. 4, pp. 999–1009, Apr. 2012, 
doi: [10.1016/j.sigpro.2011.10.012](https://doi.org/10.1016/j.sigpro.2011.10.012).

[6] V. Abolghasemi, D. Jarchi and S. Sanei, "A robust approach for optimization of the measurement matrix in 
Compressed Sensing," 2010 2nd International Workshop on Cognitive Information Processing, pp. 388-392, 2010, 
doi: [10.1109/CIP.2010.5604134](https://doi.org/10.1109/CIP.2010.5604134).

[7] T. Hong, H. Bai, S. Li, and Z. Zhu, "An efficient algorithm for designing projection matrix in compressive 
sensing based on alternating optimization," Signal Processing, vol. 125, pp. 9–20, Aug. 2016, 
doi: [10.1016/j.sigpro.2015.12.015](https://doi.org/10.1016/j.sigpro.2015.12.015).

[8] R. Yi, C. Cui,  B. Wu, and Y. Gong, "A New Method of Measurement Matrix Optimization for Compressed Sensing 
Based on Alternating Minimization," Mathematics vol. 9, no. 4, Art. no. 329, 2021,
doi: [10.3390/math9040329](https://doi.org/10.3390/math9040329).

[9] Q. Xu, Z. Sheng, Y. Fang, L. Zhang, "Measurement Matrix Optimization for Compressed Sensing System with 
Constructed Dictionary via Takenaka–Malmquist Functions," 21, no. 4: 1229, 2021, 
doi: [10.3390/s21041229](https://doi.org/10.3390/s21041229). 

[10] J. Klein, A. Léger, M. Belin, D. Défourneau, and M. J. L. Sangster, "Inelastic-electron-tunneling spectroscopy 
of metal-insulator-metal junctions," Physical Review B, vol. 7, no. 6, pp. 2336–2348, Mar. 1973.
doi: [10.1103/PhysRevB.7.2336](https://doi.org/10.1103/PhysRevB.7.2336).




