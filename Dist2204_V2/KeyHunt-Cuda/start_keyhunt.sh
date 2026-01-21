#!/bin/bash
# KeyHunt-Cuda Command Line Launcher
# Runs KeyHunt directly from command line

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if KeyHunt binary exists
if [ ! -f "./KeyHunt" ]; then
    echo "Error: KeyHunt binary not found!"
    echo "Please run ./setup.sh first to build the application."
    exit 1
fi

# Check for GPU
if nvidia-smi &>/dev/null; then
    echo "GPU detected. Running with GPU acceleration..."
    GPU_FLAG="-g"
else
    echo "No GPU detected. Running CPU-only mode..."
    GPU_FLAG=""
fi

# Show help if no arguments
if [ $# -eq 0 ]; then
    echo ""
    echo "KeyHunt-Cuda - Bitcoin Private Key Hunter"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    ./KeyHunt -h
    exit 0
fi

# Run KeyHunt with provided arguments
./KeyHunt "$@"
