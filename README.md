# [James-Stein Gradient Combiner for Inverse Monte Carlo Rendering](https://cglab.gist.ac.kr/sig25jscombiner/)

[Jeongmin Gu](https://jeongmingu.github.io/JeongminGu/), [Bochang Moon](https://cglab.gist.ac.kr/people/bochang.html)

<!-- ![Teaser](teaser.png) -->

## Overview

This repository provides the example codes for SIGGRAPH 2025 paper, [James-Stein Gradient Combiner for Inverse Monte Carlo Rendering](https://cglab.gist.ac.kr/sig25jscombiner/).



## Requirements
We recommend running this code through [Docker](https://docs.docker.com/) and [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker) on Ubuntu.
Please refer to the detailed instruction for the installation of [Docker](https://docs.docker.com/engine/install/ubuntu/) and [Nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).



## Test with example codes
For running the provided codes, you can proceed in the following order:

1. Build and run docker image 
```
bash run_docker.sh
```
2. Build PyTorch custom operators (CUDA) 
```
cd custom_ops
python setup_js.py install
python setup_filter.py install

```
3. Run the script
```
bash run_mts.sh
```

To test the Plane scene, you can download other scene components in this [repository (Chang et al.)](https://github.com/wchang22/stadam). 

## Contact

If there are any questions, issues or comments, please feel free to send an e-mail to [jeongmingu12@gmail.com](mailto:jeongmingu12@gamil.com).

