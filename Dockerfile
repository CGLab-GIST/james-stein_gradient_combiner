# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.10-py3
FROM $BASE_IMAGE

RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \
    dvipng \
    texlive-latex-extra \
    texlive-fonts-recommended \
    cm-super \
    libeigen3-dev \
    git \
    build-essential \
    libx11-dev \
    mesa-common-dev libglu1-mesa-dev \
    libxrandr-dev \
    libxi-dev \
    libxmu-dev \
    libblas-dev \
    libxinerama-dev \
    libxcursor-dev
    
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl


COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# install nvdiffrast
RUN pip install git+https://github.com/NVlabs/nvdiffrast/

# additional libraries
RUN pip install ninja imageio xatlas gdown

RUN pip install cholespy>=0.1.4 \
		matplotlib>=3.3.2 \
		pythreejs \
		git+https://github.com/skoch9/meshplot.git@725e4a7 \
		numpy>=1.19.4 \
		pandas>=1.2.4 \
		seaborn>=0.11.1 \
		tqdm>=4.54.1 \
		libigl \
		drjit \
		mitsuba \
		gpytoobox


# HDR image support
RUN imageio_download_bin freeimage

ADD /code/ext/botsch /code/ext/botsch
RUN cd /code/ext/botsch && \
	mkdir -p build && \
	cd build && \ 
	cmake .. && \
	make -j && \
	source setpath.sh
	

VOLUME /code
WORKDIR /code




