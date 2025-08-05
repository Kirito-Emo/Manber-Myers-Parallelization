FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

# System prerequisites
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    nano \
    wget \
    unzip \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    gcc-13 g++-13 \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 13 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# Set working directory
WORKDIR /project

# Default command
CMD ["/bin/bash"]