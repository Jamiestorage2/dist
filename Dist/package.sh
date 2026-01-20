#!/bin/bash
#===============================================================================
# KeyHunt-Cuda Packaging Script
#
# This script packages all required source files for distribution.
# Run this on the source machine before copying the Dist folder to new machines.
#
# Usage: ./package.sh
#===============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="/home/server/KeyHunt-Cuda/KeyHunt-Cuda"
GPU_SOURCE_DIR="/home/server/KeyHunt-Cuda/KeyHunt-Cuda/x64/GPU"
DEST_DIR="${SCRIPT_DIR}/KeyHunt-Cuda"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN} KeyHunt-Cuda Packaging Script${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Create destination directory
echo -e "${GREEN}[INFO]${NC} Creating destination directory..."
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"
mkdir -p "$DEST_DIR/GPU"
mkdir -p "$DEST_DIR/hash"

# Copy source files
echo -e "${GREEN}[INFO]${NC} Copying source files..."

# Main source files
SRC_FILES=(
    "Base58.cpp" "Base58.h"
    "Bloom.cpp" "Bloom.h"
    "CmdParse.cpp" "CmdParse.h"
    "GmpUtil.cpp" "GmpUtil.h"
    "Int.cpp" "Int.h"
    "IntGroup.cpp" "IntGroup.h"
    "IntMod.cpp"
    "KeyHunt.cpp" "KeyHunt.h"
    "Main.cpp"
    "Point.cpp" "Point.h"
    "Random.cpp" "Random.h"
    "SECP256K1.cpp" "SECP256k1.h"
    "Timer.cpp" "Timer.h"
    "Makefile"
)

for file in "${SRC_FILES[@]}"; do
    if [[ -f "${SOURCE_DIR}/${file}" ]]; then
        cp "${SOURCE_DIR}/${file}" "${DEST_DIR}/"
        echo "  Copied: ${file}"
    else
        echo -e "${YELLOW}  Warning: ${file} not found${NC}"
    fi
done

# GPU source files
echo -e "${GREEN}[INFO]${NC} Copying GPU source files..."
GPU_FILES=(
    "GPUBase58.h"
    "GPUCompute.h"
    "GPUEngine.cu"
    "GPUEngine.h"
    "GPUGenerate.cpp"
    "GPUGroup.h"
    "GPUHash.h"
    "GPUMath.h"
)

for file in "${GPU_FILES[@]}"; do
    if [[ -f "${GPU_SOURCE_DIR}/${file}" ]]; then
        cp "${GPU_SOURCE_DIR}/${file}" "${DEST_DIR}/GPU/"
        echo "  Copied: GPU/${file}"
    elif [[ -f "${SOURCE_DIR}/GPU/${file}" ]]; then
        cp "${SOURCE_DIR}/GPU/${file}" "${DEST_DIR}/GPU/"
        echo "  Copied: GPU/${file}"
    else
        echo -e "${YELLOW}  Warning: GPU/${file} not found${NC}"
    fi
done

# Hash source files
echo -e "${GREEN}[INFO]${NC} Copying hash source files..."
HASH_FILES=(
    "keccak160.cpp" "keccak160.h"
    "ripemd160.cpp" "ripemd160.h"
    "ripemd160_sse.cpp"
    "sha256.cpp" "sha256.h"
    "sha256_sse.cpp"
    "sha512.cpp" "sha512.h"
)

for file in "${HASH_FILES[@]}"; do
    if [[ -f "${SOURCE_DIR}/hash/${file}" ]]; then
        cp "${SOURCE_DIR}/hash/${file}" "${DEST_DIR}/hash/"
        echo "  Copied: hash/${file}"
    else
        echo -e "${YELLOW}  Warning: hash/${file} not found${NC}"
    fi
done

# Copy Python visualizer
echo -e "${GREEN}[INFO]${NC} Copying KeyHunt Visualizer v3.0..."
if [[ -f "${SOURCE_DIR}/keyhunt_visualizer.py" ]]; then
    cp "${SOURCE_DIR}/keyhunt_visualizer.py" "${DEST_DIR}/"
    echo "  Copied: keyhunt_visualizer.py"
else
    echo -e "${YELLOW}  Warning: keyhunt_visualizer.py not found${NC}"
fi

# Copy backup database utility
if [[ -f "${SOURCE_DIR}/backup_database.py" ]]; then
    cp "${SOURCE_DIR}/backup_database.py" "${DEST_DIR}/"
    echo "  Copied: backup_database.py"
fi

# Copy sync configuration if it exists
if [[ -f "${SOURCE_DIR}/sync_config.json" ]]; then
    cp "${SOURCE_DIR}/sync_config.json" "${DEST_DIR}/"
    echo "  Copied: sync_config.json (sync server configuration)"
fi

# Create requirements.txt for Python
echo -e "${GREEN}[INFO]${NC} Creating Python requirements.txt..."
cat > "${DEST_DIR}/requirements.txt" << 'EOF'
# KeyHunt Visualizer v3.0 - Python Dependencies
# Install with: pip3 install -r requirements.txt

# HTTP library for pool scraping
requests>=2.25.0

# HTML parsing for pool data extraction
beautifulsoup4>=4.9.0

# Fast XML/HTML parser (optional but recommended)
lxml>=4.6.0
EOF
echo "  Created: requirements.txt"

# Count files
echo ""
echo -e "${GREEN}[INFO]${NC} Package summary:"
TOTAL_FILES=$(find "$DEST_DIR" -type f | wc -l)
echo "  Total files packaged: ${TOTAL_FILES}"
echo "  Package location: ${DEST_DIR}"

# Calculate size
SIZE=$(du -sh "$DEST_DIR" | cut -f1)
echo "  Package size: ${SIZE}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Packaging Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "The Dist folder is ready for distribution."
echo "Copy the entire Dist folder to new machines and run:"
echo ""
echo "  cd Dist"
echo "  chmod +x install.sh"
echo "  ./install.sh --gpu  # For GPU build"
echo "  # OR"
echo "  ./install.sh        # For CPU-only build"
echo ""
