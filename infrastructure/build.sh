#!/bin/bash

install_packages() {
    if [ -z "$1" ];
    then
        echo "You must provide public IP: ./build.sh PUBLIC_IP"
        exit 0; 
    fi

    hack_dir="$HOME/ipa/infrastructure/hack"
    disable_firewall="${hack_dir}/disable_firewall.sh"
    repos_script="${hack_dir}/repos.sh"
    kubernetes_script="${hack_dir}/kubernetes.sh"
    utilities_script="${hack_dir}/utilities.sh"
    storage_script="${hack_dir}/storage.sh"
    gurobi_script="${hack_dir}/gurobi.sh"
    download_data="${hack_dir}/download_data.sh"
    jupyters="${hack_dir}/jupyters.sh"

    source "$disable_firewall"
    echo "disable_firewall.sh completed"
    source "$repos_script"
    echo "repos.sh completed"
    bash "$kubernetes_script"
    echo "kubernetes.sh completed"
    bash "$utilities_script"
    echo "utilities.sh completed"
    bash "$storage_script" "$1"
    echo "storage.sh completed"
    bash "$gurobi_script"
    echo "gurobi.sh completed"
    bash "$download_data"
    echo "download_data.sh completed"
    bash "$jupyters"
    echo "jupyters.sh completed"


    echo "Installation of all packages and dependencies completed"
}

# Call the function with the public IP as argument
install_packages "$1"
