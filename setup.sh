#!/bin/bash

# Clone the repository
git clone https://github.com/thinhhoang95/project-mayflower.git

# Change to project directory
cd project-mayflower

# Get current directory
CURRENT_DIR=$(pwd)

# Create .env file with environment variables if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOL
PROJECT_ROOT=$CURRENT_DIR
DATA_DIR=$CURRENT_DIR/data
OUTPUT_DIR=$CURRENT_DIR/output
EOL
fi

# Create output directory if it doesn't exist
mkdir -p $CURRENT_DIR/output

# Run training script
python train.py
