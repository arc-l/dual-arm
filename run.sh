#!/bin/bash

# Check if method is passed
if [ -z "$1" ]; then
    echo "Usage: $0 <method>"
    exit 1
fi

METHOD=$1

# Directories
DATA_DIR="data"
RESULTS_DIR="results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Loop through all JSON files in the data directory
for json_file in "$DATA_DIR"/*.json; do
    if [[ -f "$json_file" ]]; then
        filename=$(basename "$json_file")
        
        # Skip files that start with "double_cycled" if method is baseline
        if [[ "$METHOD" == "baseline" && "$filename" == double_cycled* ]]; then
            echo "Skipping $filename for baseline method."
            continue
        fi

        echo "Running evaluation on $filename with method $METHOD..."
        python evaluation.py --input "$json_file" --method "$METHOD"
        sleep 5
    fi
done

echo "Evaluation completed for all JSON files."
