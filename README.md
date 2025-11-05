# ImageBasedDLForpTX

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ma000311.github.io/ImageBasedDLForpTX.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ma000311.github.io/ImageBasedDLForpTX.jl/dev)
[![Build Status](https://travis-ci.com/ma000311/ImageBasedDLForpTX.jl.svg?branch=master)](https://travis-ci.com/ma000311/ImageBasedDLForpTX.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/ma000311/ImageBasedDLForpTX.jl?svg=true)](https://ci.appveyor.com/project/ma000311/ImageBasedDLForpTX-jl)

This repo contains the Julia scripts for end-to-end image mapping from 1TX to pTX using deep learning.
Specifically, a CNN is used as the neural network (e.g., Unet), with 1TX image as the input and pTX image as the output.

You can run the program by running the julia scripts in the folder src/.

If you use this code for your research, please consider citing the following paper: 

Ma X, Uğurbil K, Wu X. Mitigating transmit-B1 artifacts by predicting parallel transmission images with deep learning: A feasibility study using high-resolution whole-brain diffusion at 7 Tesla. Magn Reson Med. 2022; 88: 727-741. doi:10.1002/mrm.29238.

The code is made publicly available at https://github.com/XiaodongMa-MRI/ImageBasedDLForpTX.jl/releases


### Copyright & License Notice
This software is copyrighted by Regents of the University of Minnesota and covered by US 11,982,725. Regents of the University of Minnesota will license the use of this software solely for educational and research purposes by non-profit institutions and US government agencies only. For other proposed uses, contact umotc@umn.edu. The software may not be sold or redistributed without prior approval. One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice.

