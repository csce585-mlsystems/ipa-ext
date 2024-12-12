#!/bin/bash

# Update and install necessary packages
function install_packages() {
    echo "Updating packages..."
    sudo apt-get update -y
    echo "Installing required packages..."
    sudo apt-get install -y unzip ffmpeg libsm6 libxext6
    echo "Packages installation complete."
    echo
}

# Install and activate Conda environment
function install_conda_environment() {
    echo "Checking for existing Conda environment..."
    if conda env list | grep -q "^central\s"; then
        echo "Removing existing Conda environment..."
        conda deactivate || true
        conda env remove --name central
    fi

    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -O miniconda-39.sh
    bash miniconda-39.sh -b -p $HOME/miniconda3
    rm miniconda-39.sh

    echo "Initializing Conda..."
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda init
    conda env create --file ~/ipa/environment.yml
    conda activate central
    echo "Conda environment setup complete."
    echo
}

# Install customized MLServer
function install_custom_mlserver() {
    echo "Installing the customized MLServer"
    cd ~/ipa/MLServer
    git checkout configure-custom-1
    make install-dev
    cd ..
    echo "MLServer installation complete"
    echo
}

# Install ipa requirements
function install_inference_pipeline() {
    echo "Installing ipa requirements"
    cd ~/ipa
    pip install -r requirements.txt
    cd ..
    echo "ipa requirements installation complete"
    echo
}

# Install load_tester
function install_load_tester() {
    echo "Installing load tester..."
    local load_tester_dir="$HOME/ipa/load_tester"

    if [ -d "$load_tester_dir" ]; then
        cd "$load_tester_dir"
        pip install -e .
        cd
        echo "Load tester installation complete."
    else
        echo "Directory $load_tester_dir does not exist. Skipping load tester installation."
    fi
    echo
}

# Call the functions
install_packages
install_conda_environment
# install_custom_mlserver
install_inference_pipeline
install_load_tester
