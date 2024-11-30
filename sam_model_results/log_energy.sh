#!/bin/bash

# Log file to save energy consumption data
LOG_FILE="vit_l_energy_log.txt"

# Clear previous log file content if it exists
> "$LOG_FILE"

# Run your Python script in the background
python batch_process_sam.py &

# Get the PID of the running script
SCRIPT_PID=$!

# Measure energy consumption every second while the script is running
while ps -p $SCRIPT_PID > /dev/null; do
    # Run perf command and append output to the log file
    sudo perf stat -a -e power/energy-pkg/ sleep 1 2>> "$LOG_FILE"
done

# Wait for the Python script to finish
wait $SCRIPT_PID

