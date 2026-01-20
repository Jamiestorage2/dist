#!/bin/bash
#===============================================================================
# KeyHunt-Cuda Verification Script
#
# Run this after installation to verify everything is working correctly.
#
# Usage: ./verify.sh
#===============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ERRORS=0
WARNINGS=0

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} KeyHunt-Cuda Verification${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((ERRORS++))
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

info() {
    echo -e "       $1"
}

#-------------------------------------------------------------------------------
# Check system dependencies
#-------------------------------------------------------------------------------
echo -e "${CYAN}Checking System Dependencies...${NC}"

# g++-8
if command -v g++-8 &> /dev/null; then
    VERSION=$(g++-8 --version | head -1)
    pass "g++-8: $VERSION"
else
    fail "g++-8 not installed"
fi

# make
if command -v make &> /dev/null; then
    pass "make: installed"
else
    fail "make not installed"
fi

# libgmp
if ldconfig -p | grep -q libgmp; then
    pass "libgmp: installed"
else
    fail "libgmp not installed"
fi

echo ""

#-------------------------------------------------------------------------------
# Check Python dependencies
#-------------------------------------------------------------------------------
echo -e "${CYAN}Checking Python Dependencies...${NC}"

# Python3
if command -v python3 &> /dev/null; then
    VERSION=$(python3 --version)
    pass "Python: $VERSION"
else
    fail "Python3 not installed"
fi

# requests
if python3 -c "import requests" 2>/dev/null; then
    VERSION=$(python3 -c "import requests; print(requests.__version__)")
    pass "requests: $VERSION"
else
    warn "requests not installed (pool scraping will not work)"
fi

# beautifulsoup4
if python3 -c "from bs4 import BeautifulSoup" 2>/dev/null; then
    pass "beautifulsoup4: installed"
else
    warn "beautifulsoup4 not installed (pool scraping will not work)"
fi

# lxml
if python3 -c "import lxml" 2>/dev/null; then
    pass "lxml: installed"
else
    warn "lxml not installed (will use slower parser)"
fi

echo ""

#-------------------------------------------------------------------------------
# Check CUDA (optional)
#-------------------------------------------------------------------------------
echo -e "${CYAN}Checking CUDA (optional)...${NC}"

if command -v nvcc &> /dev/null; then
    VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    pass "CUDA: version $VERSION"
else
    warn "CUDA not installed (GPU mode will not be available)"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "none")
    if [[ "$GPU_INFO" != "none" ]]; then
        pass "NVIDIA GPU: $GPU_INFO"
        CCAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        info "Compute Capability: $CCAP"
    else
        warn "No NVIDIA GPU detected"
    fi
else
    warn "nvidia-smi not available"
fi

echo ""

#-------------------------------------------------------------------------------
# Check KeyHunt binary
#-------------------------------------------------------------------------------
echo -e "${CYAN}Checking KeyHunt Binary...${NC}"

if [[ -x "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" ]]; then
    pass "KeyHunt binary: found"

    # Test run
    if "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" -h &>/dev/null; then
        pass "KeyHunt execution: OK"

        # Check if GPU-enabled
        if "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" -h 2>&1 | grep -q "\-g"; then
            if "${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt" -h 2>&1 | grep -q "WITHGPU\|GPU"; then
                pass "KeyHunt GPU support: enabled"
            else
                info "KeyHunt: CPU-only build"
            fi
        fi
    else
        fail "KeyHunt cannot execute (missing dependencies?)"
    fi
else
    fail "KeyHunt binary not found at ${SCRIPT_DIR}/KeyHunt-Cuda/KeyHunt"
fi

echo ""

#-------------------------------------------------------------------------------
# Check Visualizer
#-------------------------------------------------------------------------------
echo -e "${CYAN}Checking Visualizer...${NC}"

if [[ -f "${SCRIPT_DIR}/KeyHunt-Cuda/keyhunt_visualizer.py" ]]; then
    pass "Visualizer script: found"

    # Check syntax
    if python3 -m py_compile "${SCRIPT_DIR}/KeyHunt-Cuda/keyhunt_visualizer.py" 2>/dev/null; then
        pass "Visualizer syntax: OK"
    else
        fail "Visualizer has syntax errors"
    fi
else
    fail "Visualizer script not found"
fi

echo ""

#-------------------------------------------------------------------------------
# Check launcher scripts
#-------------------------------------------------------------------------------
echo -e "${CYAN}Checking Launcher Scripts...${NC}"

if [[ -x "${SCRIPT_DIR}/start_visualizer.sh" ]]; then
    pass "start_visualizer.sh: found and executable"
else
    warn "start_visualizer.sh not found (run install.sh first)"
fi

if [[ -x "${SCRIPT_DIR}/start_keyhunt.sh" ]]; then
    pass "start_keyhunt.sh: found and executable"
else
    warn "start_keyhunt.sh not found (run install.sh first)"
fi

echo ""

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} Summary${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}All checks passed! Installation is complete.${NC}"
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}Installation complete with $WARNINGS warning(s).${NC}"
    echo "Warnings are non-critical but may affect some features."
else
    echo -e "${RED}Installation has $ERRORS error(s) and $WARNINGS warning(s).${NC}"
    echo "Please resolve the errors before using KeyHunt."
fi

echo ""
echo "Next steps:"
echo "  1. Start visualizer: ./start_visualizer.sh"
echo "  2. Open browser: http://localhost:8080"
echo ""

exit $ERRORS
