#!/bin/bash

# chmod +x multiple_training.sh
# ./multiple_training.sh

# Install necessary Python packages
pip install numpy matplotlib torch psutil

# Set parameters
training_data=100000
epochs=100

# Outer loop for values of m in the range 1 to 3
for m in {1..3};
do
    # Inner loop for values of k in the range 1 to 20
    for k in {1..20};
    do
        # Record the start time
        start_time=$(date +%s)

        python Core.py "$m" "$k" "$training_data" "$epochs"

        # Record the end time
        end_time=$(date +%s)

        # Calculate the duration
        duration=$((end_time - start_time))

        # Convert the duration to hours and minutes
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        seconds=$(((duration % 3600) % 60))

        echo ""
        echo ""
        echo "Total time taken for m = $m, k=$k: $hours : $minutes : $seconds"
        echo ""
    done
    python plot_models_training.py "$m" "$training_data" "$epochs"
done