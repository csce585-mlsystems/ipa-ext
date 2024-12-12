#!/bin/bash

# Install Google Cloud SDK
function install_gcloud() {
    echo "Installing Google Cloud SDK"
    # Install required dependencies
    sudo apt-get install -y apt-transport-https ca-certificates gnupg curl

    # Add the Google Cloud SDK distribution URI as a package source
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

    # Import the Google Cloud public key
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg

    # Update and install the Google Cloud SDK
    sudo apt-get update && sudo apt-get install -y google-cloud-cli

    echo "Google Cloud SDK installation complete"
    echo
}

# Install Helm
function install_helm() {
    echo "Installing Helm"
    wget https://get.helm.sh/helm-v3.11.3-linux-amd64.tar.gz -O helm.tar.gz
    tar -xf helm.tar.gz
    sudo mv linux-amd64/helm /usr/local/bin/helm
    rm -f helm.tar.gz
    rm -rf linux-amd64
    echo "Helm installation complete"
    echo
}

# Install MicroK8s
function install_microk8s() {
    echo "Installing MicroK8s"
    sudo snap install microk8s --classic --channel=1.23/edge
    sudo usermod -a -G microk8s cc
    mkdir -p $HOME/.kube
    sudo chown -f -R cc ~/.kube
    microk8s config > $HOME/.kube/config
    # sudo ufw allow in on cni0
    # sudo ufw allow out on cni0
    # sudo ufw default allow routed
    sudo microk8s enable dns
    echo "alias k='kubectl'" >> ~/.zshrc
    echo "MicroK8s installation complete"
    echo
}

# Main script
echo "Running script"

install_gcloud
install_helm
install_microk8s

echo "Script execution complete"
