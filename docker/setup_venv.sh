#!/usr/bin/env bash

# Setups a virtual environment for the project
# including installing all necessary dependencies and the vizer package itself.
set -e -x

if [[ -z "$1" ]]; then
    echo "Usage: $0 <venv_dir>"
    exit 1
fi

venv_dir=$1

# create virutal environment
mkdir -p $venv_dir
python3 -m venv $venv_dir

# activate virtual environment
source $venv_dir/bin/activate

export  PIP_NO_CACHE_DIR=1

# install python packages
pip install --upgrade pip
pip install -e .

# install color_mapyer package
# this is needed for accelerated categorical colormapping
pip install git+https://github.com/utkarshayachit/color_mappyer.git
