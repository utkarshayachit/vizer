#!/usr/bin/env bash

set -e -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install Python 3.9
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3.9 \
      python3.9-distutils \
      python-is-python3 \
      python3.9-venv \
      libglapi-mesa \
      libgomp1 \
      libcurl4 \
      wget \
      git \
      build-essential \
      pybind11-dev \
      python3.9-dev
rm -rf /var/lib/apt/lists/*

# Set python3 to python3.9 (otherwise, it will be python3.8)
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install ParaView
mkdir -p /opt/paraview
wget "https://www.paraview.org/files/$PARAVIEW_PACKAGE_DIR/$PARAVIEW_PACKAGE_NAME"
tar xf "${PARAVIEW_PACKAGE_NAME}" --strip-component 1 -C /opt/paraview
rm "${PARAVIEW_PACKAGE_NAME}"

# setup virtual environment
pushd /opt/vizer
. $SCRIPT_DIR/setup_venv.sh /opt/trame/env
popd
