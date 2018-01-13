# CUDA-based Fractal Dimension

Fractal dimension (FD) is an important parameter of fractal geometry that finds significant applications in various fields including image processing. In mathematics, FD is a ratio indicating a statistical index based on the comparison between measurable scales of fractal patterns.

The applications of FD include medical image analysis, e.g., in one of my research results, my colleague and I used FD for performing prostate cancer detection and ranking on whole slide images [2].

The same method was used for a contest of automated HER2 bio-marker detection (https://warwick.ac.uk/fac/sci/dcs/research/tia/her2contest/), and even won a 3rd place in one of four leader-boards. This approach was one of the most efficient methods proposed during the contest. While other deep leaning based methods required hours even days for training/testing, the FD-based approach requires less than 30 minutes.

The work was based on the code of cudaFD, which has been released at (https://github.com/huangch/cudafd). As usual, I will be more than happy to provide any further information about this work.

Please cite our paper [1] if this package helped your works.

[1] Chao-Hui HUANG, Daniel RACOCEANU, “Automated high-grade prostate cancer detection and ranking on whole slide images,” Proc. SPIE 10140, Medical Imaging 2017: Digital Pathology, 101400A (March 1, 2017); doi: 10.1117/12.2253642
