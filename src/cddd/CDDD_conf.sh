#!/bin/bash

# Export encodings
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Create dir
if [ ! -d "files" ]; then
    mkdir files
fi

CDDD_DIR=$(pwd)
IO_PATH="${CDDD_DIR}/files"
INPUT="${IO_PATH}/cddd_input.tsv"
OUTPUT="${IO_PATH}/cddd_output.pkl"
PYTHON=$(which python)
WRAPPER="${IO_PATH}/CDDD_wrapper.py"
MODEL_DIR="${CDDD_DIR}/default_model"

JSON_FILE="${IO_PATH}/CDDD_paths.json"

python - <<EOF
import json
data = {
    "io": "$IO_PATH",
    "input": "$INPUT",
    "output": "$OUTPUT",
    "python": "$PYTHON",
    "wrapper": "$WRAPPER",
    "model": "$MODEL_DIR"
}

with open("$JSON_FILE", "w") as json_file:
    json.dump(data, json_file, indent=4)
EOF

echo "Output written to ${JSON_FILE}"
