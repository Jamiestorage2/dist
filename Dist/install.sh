#!/bin/bash
#===============================================================================
# KeyHunt-Cuda Installation Script for Ubuntu 20.04
# Version: 1.0
#
# This script installs all dependencies and builds KeyHunt-Cuda with optional
# GPU support. It also sets up the KeyHunt Visualizer v3.0 web interface.
#
# Usage:
#   ./install.sh              # CPU-only build
#   ./install.sh --gpu        # GPU build (auto-detect compute capability)
#   ./install.sh --gpu --ccap 86  # GPU build with specific compute capability
#   ./install.sh --deps-only  # Install dependencies only (no build)
#   ./install.sh --help       # Show help
#
# Compute Capability Reference:
#   GTX 1050-1080:  61
#   RTX 2060-2080:  75
#   RTX 3060-3090:  86
#   RTX 4090:       89
#===============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/install.log"

# Default values
GPU_BUILD=false
CCAP=""
DEPS_ONLY=false
SKIP_DEPS=false

#-------------------------------------------------------------------------------
# Logging functions
#-------------------------------------------------------------------------------
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "$LOG_FILE"
}

header() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN} $1${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

#-------------------------------------------------------------------------------
# Help
#-------------------------------------------------------------------------------
show_help() {
    cat << EOF
KeyHunt-Cuda Installation Script for Ubuntu 20.04

Usage: $0 [OPTIONS]

Options:
  --gpu             Enable GPU (CUDA) support
  --ccap <value>    Specify GPU compute capability (e.g., 86 for RTX 3000)
                    If not specified, will auto-detect from nvidia-smi
  --deps-only       Install dependencies only, don't build
  --skip-deps       Skip dependency installation, build only
  --help            Show this help message

Examples:
  $0                    # CPU-only build
  $0 --gpu              # GPU build with auto-detected compute capability
  $0 --gpu --ccap 86    # GPU build for RTX 3000 series
  $0 --deps-only        # Install dependencies without building

Compute Capability Reference:
  GTX 750/750 Ti:   50
  GTX 1050-1080:    61
  Tesla V100:       70
  RTX 2060-2080:    75
  A100:             80
  RTX 3060-3090:    86
  RTX 4090:         89

EOF
    exit 0
}

#-------------------------------------------------------------------------------
# Parse arguments
#-------------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)
                GPU_BUILD=true
                shift
                ;;
            --ccap)
                CCAP="$2"
                shift 2
                ;;
            --deps-only)
                DEPS_ONLY=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

#-------------------------------------------------------------------------------
# Check prerequisites
#-------------------------------------------------------------------------------
check_prerequisites() {
    header "Checking Prerequisites"

    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Please run as a regular user with sudo privileges."
        exit 1
    fi

    # Check sudo access (try a simple command instead of sudo -v for non-TTY)
    if ! sudo true 2>/dev/null; then
        error "This script requires sudo privileges."
        exit 1
    fi

    # Check OS
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        log "Detected OS: $NAME $VERSION_ID"

        if [[ "$ID" != "ubuntu" ]] && [[ "$ID" != "debian" ]]; then
            warn "This script is designed for Ubuntu/Debian. Proceeding anyway..."
        fi
    else
        warn "Could not detect OS version. Proceeding anyway..."
    fi

    # Check internet connection
    if ! ping -c 1 google.com &> /dev/null; then
        error "No internet connection detected. Please check your network."
        exit 1
    fi
    log "Internet connection: OK"

    # Check if source files exist
    if [[ ! -f "${SCRIPT_DIR}/KeyHunt-Cuda/Makefile" ]]; then
        error "Source files not found. Please ensure KeyHunt-Cuda directory exists."
        exit 1
    fi
    log "Source files: Found"
}

#-------------------------------------------------------------------------------
# Install system dependencies
#-------------------------------------------------------------------------------
install_system_deps() {
    header "Installing System Dependencies"

    log "Updating package lists..."
    sudo apt-get update

    log "Installing build essentials..."
    sudo apt-get install -y \
        build-essential \
        g++-8 \
        gcc-8 \
        make \
        git \
        wget \
        curl

    log "Installing GMP library (required for big integer operations)..."
    sudo apt-get install -y libgmp-dev

    log "Installing Python 3 and pip..."
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-dev

    log "System dependencies installed successfully!"
}

#-------------------------------------------------------------------------------
# Install Python dependencies
#-------------------------------------------------------------------------------
install_python_deps() {
    header "Installing Python Dependencies"

    log "Upgrading pip..."
    python3 -m pip install --upgrade pip --user

    log "Installing required Python packages..."
    python3 -m pip install --user \
        requests>=2.25.0 \
        beautifulsoup4>=4.9.0 \
        lxml>=4.6.0

    log "Python dependencies installed successfully!"
}

#-------------------------------------------------------------------------------
# Detect GPU and compute capability
#-------------------------------------------------------------------------------
detect_gpu() {
    header "Detecting GPU"

    if ! command -v nvidia-smi &> /dev/null; then
        warn "nvidia-smi not found. GPU detection skipped."
        return 1
    fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true)

    if [[ -z "$gpu_info" ]]; then
        warn "No NVIDIA GPU detected."
        return 1
    fi

    log "Detected GPU(s):"
    echo "$gpu_info" | while read -r line; do
        log "  $line"
    done

    # Auto-detect compute capability if not specified
    if [[ -z "$CCAP" ]]; then
        CCAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        log "Auto-detected compute capability: $CCAP"
    fi

    return 0
}

#-------------------------------------------------------------------------------
# Install CUDA toolkit
#-------------------------------------------------------------------------------
install_cuda() {
    header "Installing CUDA Toolkit"

    # Check if CUDA is already installed
    if [[ -d "/usr/local/cuda" ]] && [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        local cuda_version
        cuda_version=$(/usr/local/cuda/bin/nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
        log "CUDA already installed: version $cuda_version"
        return 0
    fi

    log "Installing CUDA repository keyring..."

    # Download and install CUDA keyring
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm -f cuda-keyring_1.1-1_all.deb

    log "Updating package lists with CUDA repository..."
    sudo apt-get update

    log "Installing CUDA toolkit 12.0 (this may take a while)..."
    sudo apt-get install -y cuda-toolkit-12-0

    # Add CUDA to PATH
    log "Configuring CUDA environment..."

    # Add to .bashrc if not already present
    if ! grep -q "CUDA_HOME" ~/.bashrc; then
        cat >> ~/.bashrc << 'EOF'

# CUDA Configuration
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    fi

    # Export for current session
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    log "CUDA toolkit installed successfully!"
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

    if [[ "$GPU_BUILD" == true ]]; then
        if [[ -z "$CCAP" ]]; then
            error "GPU build requested but no compute capability specified or detected."
            error "Please specify with --ccap <value> or ensure nvidia-smi is available."
            exit 1
        fi

        log "Building with GPU support (compute capability: $CCAP)..."
        make gpu=1 CCAP="$CCAP"
    else
        log "Building CPU-only version..."
        make
    fi

    # Verify build
    if [[ -x "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" ]]; then
        log "Build successful!"

        # Create symlink
        ln -sf KeyHunt keyhunt 2>/dev/null || true
    else
        error "Build failed. Check the output above for errors."
        exit 1
    fi
}

#-------------------------------------------------------------------------------
# Create launcher script
#-------------------------------------------------------------------------------
create_launcher() {
    header "Creating Launcher Scripts"

    # Create KeyHunt launcher
    cat > "${SCRIPT_DIR}/start_keyhunt.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KeyHunt-Cuda"
./KeyHunt "$@"
EOF
    chmod +x "${SCRIPT_DIR}/start_keyhunt.sh"
    log "Created: start_keyhunt.sh"

    # Create Visualizer launcher
    cat > "${SCRIPT_DIR}/start_visualizer.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KeyHunt-Cuda"
echo "Starting KeyHunt Visualizer v3.0..."
echo "Open http://localhost:8080 in your browser"
python3 keyhunt_visualizer.py "$@"
EOF
    chmod +x "${SCRIPT_DIR}/start_visualizer.sh"
    log "Created: start_visualizer.sh"
}

#-------------------------------------------------------------------------------
# Verify installation
#-------------------------------------------------------------------------------
verify_installation() {
    header "Verifying Installation"

    local errors=0

    # Check KeyHunt binary
    if [[ -x "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" ]]; then
        log "KeyHunt binary: OK"
        "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" -h 2>&1 | head -5
    else
        error "KeyHunt binary: NOT FOUND"
        ((errors++))
    fi

    echo ""

    # Check Python dependencies
    log "Checking Python dependencies..."
    if python3 -c "import requests; print('  requests:', requests.__version__)" 2>/dev/null; then
        log "  requests: OK"
    else
        warn "  requests: NOT INSTALLED"
    fi

    if python3 -c "from bs4 import BeautifulSoup; print('  beautifulsoup4: OK')" 2>/dev/null; then
        log "  beautifulsoup4: OK"
    else
        warn "  beautifulsoup4: NOT INSTALLED"
    fi

    # Check CUDA if GPU build
    if [[ "$GPU_BUILD" == true ]]; then
        echo ""
        log "Checking CUDA..."
        if command -v nvcc &> /dev/null; then
            nvcc --version | grep "release"
            log "CUDA: OK"
        else
            error "CUDA: NOT FOUND"
            ((errors++))
        fi

        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            log "GPU: OK"
        else
            warn "nvidia-smi: NOT FOUND"
        fi
    fi

    echo ""

    if [[ $errors -eq 0 ]]; then
        log "All verifications passed!"
    else
        error "$errors verification(s) failed. Please check the errors above."
    fi

    return $errors
}

#-------------------------------------------------------------------------------
# Print summary
#-------------------------------------------------------------------------------
print_summary() {
    header "Installation Complete!"

    echo -e "${GREEN}KeyHunt-Cuda has been installed successfully!${NC}"
    echo ""
    echo "Installation directory: ${SCRIPT_DIR}"
    echo ""
    echo -e "${CYAN}Quick Start:${NC}"
    echo ""
    echo "  1. Start the Visualizer (web interface):"
    echo "     ${YELLOW}./start_visualizer.sh${NC}"
    echo "     Then open http://localhost:8080 in your browser"
    echo ""
    echo "  2. Run KeyHunt directly:"
    echo "     ${YELLOW}./start_keyhunt.sh -h${NC}"
    echo ""
    echo "  3. Example KeyHunt command:"
    if [[ "$GPU_BUILD" == true ]]; then
        echo "     ${YELLOW}./KeyHunt-Cuda/KeyHunt -m address -f address.txt -r 1:FFFFF -g -i 0${NC}"
    else
        echo "     ${YELLOW}./KeyHunt-Cuda/KeyHunt -m address -f address.txt -r 1:FFFFF -t 4${NC}"
    fi
    echo ""
    echo "Log file: ${LOG_FILE}"
    echo ""
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
main() {
    # Initialize log
    echo "KeyHunt-Cuda Installation Log" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"

    header "KeyHunt-Cuda Installer for Ubuntu 20.04"

    parse_args "$@"

    log "Configuration:"
    log "  GPU Build: $GPU_BUILD"
    log "  Compute Capability: ${CCAP:-auto-detect}"
    log "  Dependencies Only: $DEPS_ONLY"
    log "  Skip Dependencies: $SKIP_DEPS"

    check_prerequisites

    if [[ "$SKIP_DEPS" != true ]]; then
        install_system_deps
        install_python_deps

        if [[ "$GPU_BUILD" == true ]]; then
            detect_gpu
            install_cuda
        fi
    fi

    if [[ "$DEPS_ONLY" != true ]]; then
        build_keyhunt
        create_launcher
    fi

    verify_installation
    print_summary

    echo "Completed: $(date)" >> "$LOG_FILE"
}

# Run main
main "$@"
