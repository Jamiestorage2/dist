#!/bin/bash
#===============================================================================
# KeyHunt-Cuda Complete Installation Script for Ubuntu 22.04 / 24.04
# Version: 3.0
#
# This script performs a COMPLETE installation including:
# - NVIDIA GPU drivers
# - CUDA toolkit
# - All build dependencies
# - KeyHunt-Cuda compilation
# - Visualizer setup
# - Distributed sync configuration
#
# Usage:
#   ./setup.sh              # Full installation with GPU support
#   ./setup.sh --cpu-only   # CPU-only build (no NVIDIA/CUDA)
#   ./setup.sh --skip-nvidia # Skip NVIDIA driver (already installed)
#   ./setup.sh --help       # Show help
#
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/install.log"

# Options
CPU_ONLY=false
SKIP_NVIDIA=false
CCAP=""

#-------------------------------------------------------------------------------
# Logging
#-------------------------------------------------------------------------------
log() { echo -e "${GREEN}[INFO]${NC} $1"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "$LOG_FILE"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "$LOG_FILE"; }
header() { echo ""; echo -e "${CYAN}========================================${NC}"; echo -e "${CYAN} $1${NC}"; echo -e "${CYAN}========================================${NC}"; echo ""; }

#-------------------------------------------------------------------------------
# Help
#-------------------------------------------------------------------------------
show_help() {
    cat << 'EOF'
KeyHunt-Cuda Complete Installation Script for Ubuntu 22.04 / 24.04

Usage: ./setup.sh [OPTIONS]

Options:
  --cpu-only        Build without GPU support (no NVIDIA/CUDA needed)
  --skip-nvidia     Skip NVIDIA driver installation (use if already installed)
  --ccap <value>    Specify GPU compute capability (auto-detected if not set)
  --help            Show this help message

Examples:
  ./setup.sh                    # Full GPU installation
  ./setup.sh --skip-nvidia      # GPU build, but skip driver install
  ./setup.sh --cpu-only         # CPU-only build
  ./setup.sh --ccap 86          # Force compute capability 86 (RTX 3000)

Compute Capability Reference:
  GTX 1050-1080:    61
  RTX 2060-2080:    75
  RTX 3060-3090:    86
  RTX 4070-4090:    89
  RTX 5090:         120

After installation:
  ./start_visualizer.sh         # Start web interface on http://localhost:8080
  ./start_keyhunt.sh -h         # Show KeyHunt help

EOF
    exit 0
}

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cpu-only) CPU_ONLY=true; shift ;;
            --skip-nvidia) SKIP_NVIDIA=true; shift ;;
            --ccap) CCAP="$2"; shift 2 ;;
            --help|-h) show_help ;;
            *) error "Unknown option: $1"; show_help ;;
        esac
    done
}

#-------------------------------------------------------------------------------
# Check Prerequisites
#-------------------------------------------------------------------------------
check_prerequisites() {
    header "Checking Prerequisites"

    # Not root
    if [[ $EUID -eq 0 ]]; then
        error "Do not run as root. Run as regular user with sudo access."
        exit 1
    fi

    # Sudo access
    if ! sudo -n true 2>/dev/null; then
        log "Sudo password required..."
        sudo true || { error "Sudo access required"; exit 1; }
    fi
    log "Sudo access: OK"

    # OS Check
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        log "OS: $NAME $VERSION_ID"
        if [[ "$ID" != "ubuntu" ]]; then
            warn "This script is optimized for Ubuntu. Proceeding anyway..."
        fi
    fi

    # Internet
    if ! ping -c 1 -W 3 google.com &>/dev/null; then
        error "No internet connection"
        exit 1
    fi
    log "Internet: OK"

    # Source files
    if [[ ! -f "${SCRIPT_DIR}/KeyHunt-Cuda/Makefile" ]]; then
        error "KeyHunt-Cuda source not found"
        exit 1
    fi
    log "Source files: OK"
}

#-------------------------------------------------------------------------------
# Install System Dependencies
#-------------------------------------------------------------------------------
install_system_deps() {
    header "Installing System Dependencies"

    log "Updating package lists..."
    sudo apt-get update

    log "Installing build tools..."
    sudo apt-get install -y \
        build-essential \
        g++ \
        gcc \
        g++-11 \
        gcc-11 \
        make \
        git \
        wget \
        curl \
        pkg-config \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release

    log "Installing libraries..."
    sudo apt-get install -y \
        libgmp-dev \
        libssl-dev

    log "Installing Python..."
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-requests \
        python3-bs4 \
        python3-lxml

    log "System dependencies installed!"
}

#-------------------------------------------------------------------------------
# Install NVIDIA Drivers
#-------------------------------------------------------------------------------
install_nvidia_drivers() {
    header "Installing NVIDIA GPU Drivers"

    # Check if already installed
    if command -v nvidia-smi &>/dev/null; then
        local driver_version
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "$driver_version" ]]; then
            log "NVIDIA driver already installed: version $driver_version"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            return 0
        fi
    fi

    log "Installing NVIDIA drivers..."

    # Add NVIDIA repository
    sudo apt-get install -y ubuntu-drivers-common

    # Install recommended driver
    log "Detecting and installing recommended NVIDIA driver..."
    sudo ubuntu-drivers install nvidia --gpgpu 2>/dev/null || sudo ubuntu-drivers autoinstall

    # Alternative: Install specific driver version
    # sudo apt-get install -y nvidia-driver-535

    log "NVIDIA driver installation complete!"
    warn "A REBOOT may be required for the driver to work properly."
    warn "After reboot, run this script again with --skip-nvidia"
}

#-------------------------------------------------------------------------------
# Install CUDA Toolkit
#-------------------------------------------------------------------------------
install_cuda() {
    header "Installing CUDA Toolkit"

    # Check if nvcc exists
    if command -v nvcc &>/dev/null; then
        local cuda_version
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        log "CUDA already installed: version $cuda_version"
        return 0
    fi

    log "Installing CUDA toolkit from Ubuntu repositories..."
    sudo apt-get install -y nvidia-cuda-toolkit

    # Verify
    if command -v nvcc &>/dev/null; then
        nvcc --version | grep "release"
        log "CUDA toolkit installed successfully!"
    else
        error "CUDA installation failed"
        exit 1
    fi
}

#-------------------------------------------------------------------------------
# Detect GPU
#-------------------------------------------------------------------------------
detect_gpu() {
    header "Detecting GPU"

    if ! command -v nvidia-smi &>/dev/null; then
        if [[ "$CPU_ONLY" != true ]]; then
            error "nvidia-smi not found. Install NVIDIA drivers first or use --cpu-only"
            exit 1
        fi
        return
    fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null)

    if [[ -z "$gpu_info" ]]; then
        error "No NVIDIA GPU detected"
        exit 1
    fi

    log "Detected GPU:"
    echo "$gpu_info" | while read -r line; do
        log "  $line"
    done

    # Auto-detect compute capability
    if [[ -z "$CCAP" ]]; then
        CCAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        log "Compute capability: $CCAP"
    fi
}

#-------------------------------------------------------------------------------
# Get Best CCAP for Compilation
#-------------------------------------------------------------------------------
get_build_ccap() {
    local detected="$1"

    # Get max supported by nvcc
    local max_supported
    max_supported=$(nvcc --list-gpu-arch 2>/dev/null | grep compute_ | tail -1 | sed 's/compute_//' || echo "87")

    log "NVCC max supported: $max_supported"
    log "GPU compute capability: $detected"

    if [[ "$detected" -gt "$max_supported" ]]; then
        log "Using $max_supported with PTX for forward compatibility"
        echo "$max_supported"
    else
        echo "$detected"
    fi
}

#-------------------------------------------------------------------------------
# Fix Makefile
#-------------------------------------------------------------------------------
fix_makefile() {
    header "Configuring Makefile"

    local makefile="${SCRIPT_DIR}/KeyHunt-Cuda/Makefile"

    # Backup
    cp "$makefile" "${makefile}.backup" 2>/dev/null || true

    log "Applying fixes for Ubuntu 22.04/24.04..."

    # Fix compiler paths
    sed -i 's/CXX        = g++-8/CXX        = g++/' "$makefile"
    sed -i 's|CUDA       = /usr/local/cuda|CUDA       = /usr/lib/cuda|' "$makefile"
    sed -i 's|CXXCUDA    = /usr/bin/g++-8|CXXCUDA    = /usr/bin/g++-11|' "$makefile"
    sed -i 's|NVCC       = $(CUDA)/bin/nvcc|NVCC       = /usr/bin/nvcc|' "$makefile"

    # Use PTX for forward compatibility with newer GPUs
    sed -i 's/code=sm_$(ccap)/code=compute_$(ccap)/g' "$makefile"

    log "Makefile configured!"
}

#-------------------------------------------------------------------------------
# Build KeyHunt
#-------------------------------------------------------------------------------
build_keyhunt() {
    header "Building KeyHunt"

    cd "${SCRIPT_DIR}/KeyHunt-Cuda"

    log "Cleaning previous build..."
    make clean 2>/dev/null || true
    rm -rf obj/

    log "Creating build directories..."
    mkdir -p obj/GPU obj/hash

    if [[ "$CPU_ONLY" == true ]]; then
        log "Building CPU-only version..."
        make 2>&1 | tee -a "$LOG_FILE"
    else
        local build_ccap
        build_ccap=$(get_build_ccap "$CCAP")

        log "Building with GPU support (CCAP: $build_ccap)..."
        make KeyHunt gpu=1 CCAP="$build_ccap" 2>&1 | tee -a "$LOG_FILE"
    fi

    # Verify
    if [[ -x "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" ]]; then
        log "Build successful!"
        ls -lh "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt"
    else
        error "Build failed!"
        exit 1
    fi
}

#-------------------------------------------------------------------------------
# Configure Sync
#-------------------------------------------------------------------------------
configure_sync() {
    header "Configuring Distributed Sync"

    local sync_config="${SCRIPT_DIR}/KeyHunt-Cuda/sync_config.json"

    if [[ ! -f "$sync_config" ]]; then
        warn "sync_config.json not found, skipping"
        return
    fi

    # Check if client_id needs to be generated
    local current_id
    current_id=$(python3 -c "import json; print(json.load(open('$sync_config')).get('client_id', ''))" 2>/dev/null || echo "")

    if [[ -z "$current_id" ]]; then
        log "Generating unique client ID..."

        # Machine ID + random
        local machine_id
        if [[ -f /etc/machine-id ]]; then
            machine_id=$(head -c 8 /etc/machine-id)
        else
            machine_id=$(hostname | md5sum | head -c 8)
        fi
        local random_part=$(head -c 4 /dev/urandom | xxd -p)
        local new_id="${machine_id}${random_part}"

        # Client name with GPU info
        local client_name=$(hostname)
        if command -v nvidia-smi &>/dev/null; then
            local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/NVIDIA //' | tr ' ' '_')
            [[ -n "$gpu_name" ]] && client_name="${client_name}_${gpu_name}"
        fi

        # Update config
        python3 << EOF
import json
with open('$sync_config', 'r') as f:
    config = json.load(f)
config['client_id'] = '$new_id'
config['client_name'] = '$client_name'
with open('$sync_config', 'w') as f:
    json.dump(config, f, indent=2)
EOF

        log "Client ID: $new_id"
        log "Client Name: $client_name"
    else
        log "Sync already configured: $current_id"
    fi
}

#-------------------------------------------------------------------------------
# Create Launcher Scripts
#-------------------------------------------------------------------------------
create_launchers() {
    header "Creating Launcher Scripts"

    # KeyHunt launcher
    cat > "${SCRIPT_DIR}/start_keyhunt.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KeyHunt-Cuda"
./KeyHunt "$@"
EOF
    chmod +x "${SCRIPT_DIR}/start_keyhunt.sh"
    log "Created: start_keyhunt.sh"

    # Visualizer launcher
    cat > "${SCRIPT_DIR}/start_visualizer.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KeyHunt-Cuda"
echo "============================================"
echo " KeyHunt Visualizer v3.0"
echo " Open http://localhost:8080 in your browser"
echo "============================================"
python3 keyhunt_visualizer.py "$@"
EOF
    chmod +x "${SCRIPT_DIR}/start_visualizer.sh"
    log "Created: start_visualizer.sh"

    # Background visualizer launcher
    cat > "${SCRIPT_DIR}/start_visualizer_bg.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KeyHunt-Cuda"
nohup python3 keyhunt_visualizer.py --no-browser -p 8080 > visualizer.log 2>&1 &
echo "Visualizer started in background (PID: $!)"
echo "Log: $SCRIPT_DIR/KeyHunt-Cuda/visualizer.log"
echo "URL: http://localhost:8080"
EOF
    chmod +x "${SCRIPT_DIR}/start_visualizer_bg.sh"
    log "Created: start_visualizer_bg.sh"

    # Stop script
    cat > "${SCRIPT_DIR}/stop_visualizer.sh" << 'EOF'
#!/bin/bash
pkill -f keyhunt_visualizer.py && echo "Visualizer stopped" || echo "Visualizer not running"
EOF
    chmod +x "${SCRIPT_DIR}/stop_visualizer.sh"
    log "Created: stop_visualizer.sh"
}

#-------------------------------------------------------------------------------
# Verify Installation
#-------------------------------------------------------------------------------
verify_installation() {
    header "Verifying Installation"

    local errors=0

    # KeyHunt binary
    if [[ -x "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" ]]; then
        log "KeyHunt binary: OK"
        "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" -h 2>&1 | head -3
    else
        error "KeyHunt binary: MISSING"
        ((errors++))
    fi

    # Python deps
    log "Python dependencies:"
    python3 -c "import requests; print('  requests: OK')" 2>/dev/null || warn "  requests: MISSING"
    python3 -c "from bs4 import BeautifulSoup; print('  bs4: OK')" 2>/dev/null || warn "  bs4: MISSING"
    python3 -c "import lxml; print('  lxml: OK')" 2>/dev/null || warn "  lxml: MISSING"

    # GPU (if not CPU-only)
    if [[ "$CPU_ONLY" != true ]]; then
        log "GPU:"
        if command -v nvidia-smi &>/dev/null; then
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
            log "  NVIDIA driver: OK"
        else
            warn "  NVIDIA driver: NOT DETECTED"
        fi

        if command -v nvcc &>/dev/null; then
            log "  CUDA: $(nvcc --version | grep release | sed 's/.*release //')"
        else
            warn "  CUDA: NOT DETECTED"
        fi

        # Test GPU mining
        log "Testing GPU..."
        if "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" -g -m ADDRESS --coin BTC --range 1:FF 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH 2>&1 | grep -q "GPU #"; then
            log "  GPU mining: OK"
        else
            warn "  GPU mining: Test inconclusive"
        fi
    fi

    echo ""
    if [[ $errors -eq 0 ]]; then
        log "All checks passed!"
    else
        error "$errors check(s) failed"
    fi
}

#-------------------------------------------------------------------------------
# Print Summary
#-------------------------------------------------------------------------------
print_summary() {
    header "Installation Complete!"

    echo -e "${GREEN}KeyHunt-Cuda has been installed successfully!${NC}"
    echo ""
    echo "Location: ${SCRIPT_DIR}"
    echo ""
    echo -e "${CYAN}Quick Start:${NC}"
    echo ""
    echo "  1. Start the Visualizer (web interface):"
    echo -e "     ${YELLOW}./start_visualizer.sh${NC}"
    echo "     Then open http://localhost:8080"
    echo ""
    echo "  2. Start in background:"
    echo -e "     ${YELLOW}./start_visualizer_bg.sh${NC}"
    echo ""
    echo "  3. Run KeyHunt directly:"
    echo -e "     ${YELLOW}./start_keyhunt.sh -h${NC}"
    echo ""
    if [[ "$CPU_ONLY" != true ]]; then
        echo -e "${CYAN}GPU Info:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (GPU info unavailable)"
        echo ""
    fi
    echo -e "${CYAN}Distributed Mining:${NC}"
    echo "  - Sync enabled: Every 90 seconds"
    echo "  - Blocks scanned are shared with team"
    echo "  - Found keys are uploaded to server"
    echo ""
    echo "Log file: ${LOG_FILE}"
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
main() {
    # Init log
    echo "KeyHunt-Cuda Installation Log - $(date)" > "$LOG_FILE"
    echo "==========================================" >> "$LOG_FILE"

    header "KeyHunt-Cuda Complete Installer"
    echo "Ubuntu 22.04 / 24.04"
    echo ""

    parse_args "$@"

    log "Configuration:"
    log "  CPU Only: $CPU_ONLY"
    log "  Skip NVIDIA: $SKIP_NVIDIA"
    log "  CCAP: ${CCAP:-auto}"

    check_prerequisites
    install_system_deps

    if [[ "$CPU_ONLY" != true ]]; then
        if [[ "$SKIP_NVIDIA" != true ]]; then
            install_nvidia_drivers
        fi
        install_cuda
        detect_gpu
    fi

    fix_makefile
    build_keyhunt
    configure_sync
    create_launchers
    verify_installation
    print_summary

    echo "Completed: $(date)" >> "$LOG_FILE"
}

main "$@"
