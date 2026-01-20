# KeyHunt-Cuda Distribution Package

This package contains everything needed to install and run KeyHunt-Cuda with the Visualizer v3.0 web interface on a fresh Ubuntu 20.04 system.

## Contents

```
Dist/
├── install.sh          # Main installation script
├── package.sh          # Script to bundle source files (run on source machine)
├── README.md           # This file
└── KeyHunt-Cuda/       # Source files (created by package.sh)
    ├── *.cpp, *.h      # Core source files
    ├── GPU/            # CUDA GPU sources
    ├── hash/           # Hash function sources
    ├── Makefile        # Build configuration
    ├── keyhunt_visualizer.py  # Web-based visualizer v3.0
    └── requirements.txt       # Python dependencies
```

## Quick Start

### On the Source Machine (where you have working KeyHunt)

1. Run the packaging script to bundle source files:
   ```bash
   cd /home/server/KeyHunt-Cuda/Dist
   chmod +x package.sh
   ./package.sh
   ```

2. Copy the entire `Dist` folder to your target machine(s)

### On the Target Machine (fresh Ubuntu 20.04)

1. Copy the Dist folder to the target machine

2. Run the installer:
   ```bash
   cd Dist
   chmod +x install.sh

   # For GPU build (recommended):
   ./install.sh --gpu

   # For CPU-only build:
   ./install.sh
   ```

3. Start the Visualizer:
   ```bash
   ./start_visualizer.sh
   ```
   Then open http://localhost:8080 in your browser

## Installation Options

```bash
./install.sh              # CPU-only build
./install.sh --gpu        # GPU build (auto-detect compute capability)
./install.sh --gpu --ccap 86  # GPU build with specific compute capability
./install.sh --deps-only  # Install dependencies only (no build)
./install.sh --skip-deps  # Build only (dependencies already installed)
./install.sh --help       # Show help
```

## GPU Compute Capability Reference

| GPU Series | Compute Capability |
|------------|-------------------|
| GTX 750/750 Ti | 50 |
| GTX 1050-1080 | 61 |
| Tesla V100 | 70 |
| RTX 2060-2080 | 75 |
| A100 | 80 |
| RTX 3060-3090 | 86 |
| RTX 4090 | 89 |

To find your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

## System Requirements

- **OS**: Ubuntu 20.04 LTS (also works on Debian 10+)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk**: 5GB free space
- **GPU** (optional): NVIDIA GPU with compute capability 5.0+
- **NVIDIA Driver**: 450+ (for GPU builds)

## Dependencies Installed

### System Packages
- build-essential, g++-8, gcc-8, make
- libgmp-dev (GNU Multiple Precision library)
- Python 3 with pip

### Python Packages
- requests (HTTP library for pool scraping)
- beautifulsoup4 (HTML parsing)
- lxml (XML/HTML parser)

### CUDA (GPU builds only)
- CUDA Toolkit 12.0

## Visualizer Features

The KeyHunt Visualizer v3.0 provides:

- **Grid Visualization**: Visual representation of the search keyspace
- **Drill-down Zoom**: Click on cells to zoom into sub-ranges
- **Mining Control**: Start/stop mining from any grid cell
- **Pool Scraping**: Automatically fetch scanned ranges from btcpuzzle.info
- **Database Tracking**: SQLite databases track all scanned ranges
- **Multi-GPU Support**: Manage multiple GPUs for parallel scanning
- **Cloud Sync**: Optional synchronization between multiple machines
- **Real-time Stats**: Live speed and progress monitoring

## Usage

### Start the Visualizer
```bash
./start_visualizer.sh
# Opens web interface at http://localhost:8080
```

### Run KeyHunt Directly
```bash
# GPU mode (single address):
./KeyHunt-Cuda/KeyHunt -m address -g -i 0 \
    --coin BTC \
    -o Found.txt \
    --range 40000000000000000:7FFFFFFFFFFFFFFFF \
    1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU

# CPU mode (multiple threads):
./KeyHunt-Cuda/KeyHunt -m address -t 4 \
    --coin BTC \
    -o Found.txt \
    --range 40000000000000000:7FFFFFFFFFFFFFFFF \
    1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
```

## Troubleshooting

### Build fails with "g++-8 not found"
```bash
sudo apt-get install g++-8 gcc-8
```

### CUDA compilation errors
Ensure CUDA is properly installed:
```bash
nvcc --version
# If not found, add to PATH:
export PATH=/usr/local/cuda/bin:$PATH
```

### "libgmp.so not found" at runtime
```bash
sudo apt-get install libgmp-dev
```

### Visualizer won't start
Check Python dependencies:
```bash
python3 -c "import requests; from bs4 import BeautifulSoup; print('OK')"
# If error, install:
pip3 install requests beautifulsoup4 lxml
```

### Port 8080 already in use
Either stop the existing service or use a different port:
```bash
python3 keyhunt_visualizer.py --port 8888
```

## File Locations After Install

```
Dist/
├── install.log           # Installation log
├── start_keyhunt.sh      # KeyHunt launcher
├── start_visualizer.sh   # Visualizer launcher
└── KeyHunt-Cuda/
    ├── KeyHunt           # Compiled binary
    ├── keyhunt           # Symlink to KeyHunt
    ├── keyhunt_visualizer.py
    ├── scan_data_puzzle_*.db  # Created at runtime
    └── Found.txt         # Results file
```

## Support

For issues with:
- **KeyHunt-Cuda**: Check the original repository
- **Visualizer**: Review `keyhunt_visualizer.py` logs
- **Installation**: Check `install.log` for errors
