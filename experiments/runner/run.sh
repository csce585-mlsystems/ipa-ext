#!/bin/bash

conda activate central
export MPLCONFIGDIR=/tmp/matplotlib


python runner_script.py --config-name video-1
sleep 60
# python runner_script.py --config-name video-2
# sleep 60
# python runner_script.py --config-name video-3
# sleep 60
python runner_script.py --config-name video-4
sleep 60
# python runner_script.py --config-name video-5
# sleep 60
python runner_script.py --config-name video-6
sleep 60
# python runner_script.py --config-name video-7
# sleep 60
# python runner_script.py --config-name video-8
# sleep 60
python runner_script.py --config-name video-9
sleep 60
# python runner_script.py --config-name video-10
# sleep 60
python runner_script.py --config-name video-11
sleep 60
# python runner_script.py --config-name video-12
# sleep 60
# python runner_script.py --config-name video-13
# sleep 60
python runner_script.py --config-name video-14
sleep 60
# python runner_script.py --config-name video-15
# sleep 60
python runner_script.py --config-name video-16
sleep 60
# python runner_script.py --config-name video-17
# sleep 60
# python runner_script.py --config-name video-18
# sleep 60
python runner_script.py --config-name video-19
sleep 60
# python runner_script.py --config-name video-20
# sleep 60

# Draw the results of the experiment
jupyter nbconvert --execute --to notebook --inplace ~/ipa/experiments/runner/notebooks/Jsys-reviewers_modified.ipynb
