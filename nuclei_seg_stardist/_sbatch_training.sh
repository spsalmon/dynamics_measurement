#!/bin/bash

#SBATCH -J train_stardist
#SBATCH -o train_stardist-%j.out
#SBATCH -e train_stardist-%j.err
#SBATCH -c 64
#SBATCH -t 72:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx6000:1

# Default configuration file
DEFAULT_CONFIG_FILE="./config.yaml"
CONFIG_FILE="$DEFAULT_CONFIG_FILE"

# Function to show usage
usage() {
    echo "Usage: $0 [-c <config_file> | --config <config_file>]" >&2
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--config)
        CONFIG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        usage
        ;;
    esac
done

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE" >&2
    exit 1
fi

# Run the Python script with the specified or default configuration file
CUDA_LAUNCH_BLOCKING=1 ~/.local/bin/micromamba run -n stardist python3 train.py -c "$CONFIG_FILE"
