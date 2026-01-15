#!/bin/bash
# Wrapper script to run license plate detection with virtual environment

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the detection script
python license_plate_detection.py "$@"

# Deactivate virtual environment
deactivate
