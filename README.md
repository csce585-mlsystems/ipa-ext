# Inference Pipeline Adapter (IPA) Extension Project

A CSCE 585 - Machine Learning Systems Project

## Installation
1. Follow the instructions in the original ipa repository (https://github.com/reconfigurable-ml-pipeline/ipa/blob/main/infrastructure/automated.md) 

2. For measuring energy consumption, install `perf` tool using the following commands:
```
sudo apt update
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
perf --version
```

## Usage

1. **Run Experiments:**
   - Use the `experiments/runner/run_revision_perf.sh` script to run the experiments.
   - This script internally calls all experiments in `experiments/runner/run-revision.sh` and uses `perf` to measure the energy consumption of each experiment.

2. **Energy Measurement Logs:**
   - The energy consumption logs are saved in the `energy_logs` directory.

3. **Generate Plots:**
   - Use the `experiments/runner/run_revision_perf_plot.ipynb` Jupyter Notebook to generate plots for the energy benchmarks of the workloads.
   - The generated plots are saved in the `plots` directory.

## Final Report
Please find the Milestone 3 report [here](IPA_Ext_milestone3.pdf).