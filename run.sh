#!/bin/bash

# Directory containing JSON files
DATA_DIR="data"

# Loop through all JSON files in the data directory
for json_file in "$DATA_DIR"/*.json; do
    if [[ -f "$json_file" ]]; then
        echo "Running evaluation on $json_file..."
        python evaluation.py --input "$json_file" --method baseline
        sleep 5
    else
        echo "No JSON files found in $DATA_DIR"
        exit 1
    fi
done

echo "Evaluation completed for all JSON files."
