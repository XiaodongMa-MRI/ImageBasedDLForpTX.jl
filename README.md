# ImageBasedDLForpTX

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ma000311.github.io/ImageBasedDLForpTX.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ma000311.github.io/ImageBasedDLForpTX.jl/dev)
[![Build Status](https://travis-ci.com/ma000311/ImageBasedDLForpTX.jl.svg?branch=master)](https://travis-ci.com/ma000311/ImageBasedDLForpTX.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/ma000311/ImageBasedDLForpTX.jl?svg=true)](https://ci.appveyor.com/project/ma000311/ImageBasedDLForpTX-jl)

This repo contains the Julia scripts for end-to-end image mapping from 1TX to pTX using deep learning.
Specifically, a CNN is used as the neural network (e.g., Unet), with 1TX image as the input and pTX image as the output.

You can run the program by running the julia scripts in the folder src/.

All my ppts including ISMRM abstracts are documented in the folder reports/.

More details on the method and its utility for diffusion can be found in the paper: X. Ma, K. Uğurbil, and X. Wu, Mitigating transmit-B1 artifacts by predicting parallel transmission images with deep learning: a feasibility study using high-resolution whole-brain diffusion at 7 Tesla. submitted to MRM (minor revision)

The code is made publicly available at https://github.com/XiaodongMa-MRI/ImageBasedDLForpTX.jl/releases
