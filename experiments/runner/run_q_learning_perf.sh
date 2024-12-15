#!/bin/bash

conda activate central

# Array of experiment numbers
experiments=("16" "19")

# Loop through each experiment
for exp in "${experiments[@]}"
do
    # Run the experiment with perf energy measurement per second, including RAM energy
    perf stat -I 1000 -o ${HOME}/ipa/energy_comparison/energy-video-mul-${exp}.log -e power/energy-pkg/ python runner_script.py --config-name video-mul-${exp}
    sleep 60
done

# Draw the results of the experiment
jupyter nbconvert --execute --to notebook --inplace /home/cc/ipa/experiments/runner/notebooks/Jsys-reviewers_revision_modified.ipynb
