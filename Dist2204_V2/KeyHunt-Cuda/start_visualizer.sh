#!/bin/bash
# KeyHunt-Cuda Visualizer Launcher
# Starts the web-based visualizer interface

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if KeyHunt binary exists
if [ ! -f "./KeyHunt" ]; then
    echo "Error: KeyHunt binary not found!"
    echo "Please run ./setup.sh first to build the application."
    exit 1
fi

# Check Python dependencies
python3 -c "import requests, bs4, lxml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
fi

# Start the visualizer
echo "Starting KeyHunt Visualizer..."
echo "Open your browser to: http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""
python3 keyhunt_visualizer.py
