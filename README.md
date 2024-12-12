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


4. **Segment Anything:**
   - Set up a conda environment using `conda create -n sam python=3.8`
   - `pip install git+https://github.com/facebookresearch/segment-anything.git`
   - `cd segment-anything; pip install -e .`
   - Download sam weights from and place them in `segment-anything/model_checkpoints/`:
      -`vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
      -`vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
      -`vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
   
   - To generate energy consumption results use `segment-anything/sam_energy.sh`
   - The energy logs will be in `segment-anything/sam_energy_log/`
   - The segmentation masks will be saved in `segment-anything/segmentation_benchmark/{model_type}_results/`
   - To graph the energy consumption results you can run the `segment_anything/graph_sam_energy_consumption.ipynb` file
   
   - To calculate IoU run `segment-anything/metrics.sh`

## Final Deliverables
Please find the Final report [here](FinalReport.pdf).
Please find the Final presentation slides [here](FinalPresentation.pdf).