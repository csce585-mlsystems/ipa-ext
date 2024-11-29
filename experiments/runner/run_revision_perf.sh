#!/bin/bash

conda activate central

mkdir logs

# Array of experiment numbers
experiments=("1" "2" "3" "5" "6" "7" "8" "10" "11" "12" "13" "15" "16" "17" "18" "20")

# Loop through each experiment
for exp in "${experiments[@]}"
do
    # Run the experiment with perf energy measurement per second, including RAM energy
    perf stat -I 1000 -o ${HOME}/ipa-ext/energy_logs/energy-video-mul-${exp}.log -e power/energy-pkg/,power/energy-ram/ python runner_script.py --config-name video-mul-${exp}
    sleep 60
done

# Draw the results of the experiment
jupyter nbconvert --execute --to notebook --inplace ~/ipa/experiments/runner/notebooks/Jsys-reviewers-revision.ipynb
