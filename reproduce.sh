#!/bin/bash
#
# Master Reproducibility Script for FOA 2024 Replication
#
# This script ensures complete end-to-end reproducibility of:
# - Stage 1: Source and TENT baseline implementation and evaluation
# - Stage 2: FOA methodology (to be implemented)
# - Stage 3: Comparative evaluation (to be implemented)
#
# Usage: ./reproduce.sh [--stage <1|2|3|all>] [--test-only]
#

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
STAGE="1"
TEST_ONLY=false
DATA_ROOT="./data/imagenet-c"
BATCH_SIZE=64
DEVICE="cuda"  # Will auto-detect

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./reproduce.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage <1|2|3|all>    Which stage to run (default: 1)"
            echo "  --test-only            Run verification tests only"
            echo "  --data_root <path>     Path to ImageNet-C dataset (default: ./data/imagenet-c)"
            echo "  --batch_size <int>     Batch size for evaluation (default: 64)"
            echo "  --device <cuda|mps|cpu> Device to use (default: auto-detect)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Stages:"
            echo "  1: Source and TENT baseline evaluation"
            echo "  2: FOA methodology implementation and evaluation"
            echo "  3: Comparative analysis and ablation studies"
            echo "  all: Run all stages sequentially"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FOA 2024 Replication - Reproducibility${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Check Python version
echo -e "${GREEN}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo -e "${RED}Error: Python 3.12 or higher required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version compatible${NC}"
echo ""

# Step 2: Set up virtual environment
echo -e "${GREEN}[2/6] Setting up virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo -e "${GREEN}✓ Virtual environment ready${NC}"
echo ""

# Step 3: Install dependencies
echo -e "${GREEN}[3/6] Installing dependencies...${NC}"
echo "This may take several minutes..."

pip install --quiet --upgrade pip

# Install PyTorch (device-specific)
if [[ "$DEVICE" == "cuda" ]] && command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu118
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "Installing PyTorch with MPS (Apple Silicon) support..."
    pip install --quiet torch torchvision
else
    echo "Installing PyTorch (CPU only)..."
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
pip install --quiet timm numpy pillow tqdm matplotlib scipy scikit-learn pandas seaborn requests pyyaml

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 4: Verify implementation
echo -e "${GREEN}[4/6] Running verification tests...${NC}"
cd workflow

python test_implementation.py

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Verification tests failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All verification tests passed${NC}"
cd ..
echo ""

# If test-only mode, stop here
if [ "$TEST_ONLY" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Verification Complete${NC}"
    echo -e "${GREEN}========================================${NC}"
    exit 0
fi

# Step 5: Check for ImageNet-C dataset
echo -e "${GREEN}[5/6] Checking for ImageNet-C dataset...${NC}"
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${YELLOW}Warning: ImageNet-C dataset not found at: $DATA_ROOT${NC}"
    echo ""
    echo "ImageNet-C must be downloaded manually from:"
    echo "https://zenodo.org/record/2235448"
    echo ""
    echo "After downloading:"
    echo "1. Extract the dataset"
    echo "2. Ensure the structure is: $DATA_ROOT/corruption_type/severity/class/images"
    echo "3. Re-run this script"
    echo ""
    echo -e "${RED}[HALT_ROUTINE]${NC}"
    echo "FAILURE: ImageNet-C dataset unavailable"
    echo "REASON: Manual download required due to licensing"
    echo "SCIENTIFIC VALIDITY: Cannot use synthetic data for benchmarking"
    exit 1
fi

# Verify dataset has the expected structure
SAMPLE_CORRUPTION="gaussian_noise"
SAMPLE_PATH="$DATA_ROOT/$SAMPLE_CORRUPTION/1"
if [ ! -d "$SAMPLE_PATH" ]; then
    echo -e "${RED}Error: ImageNet-C structure incorrect${NC}"
    echo "Expected: $SAMPLE_PATH"
    echo "Please verify the dataset structure"
    exit 1
fi

echo -e "${GREEN}✓ ImageNet-C dataset found and verified${NC}"
echo ""

# Step 6: Run evaluation based on stage
echo -e "${GREEN}[6/6] Running evaluation...${NC}"
echo "Stage: $STAGE"
echo "Data root: $DATA_ROOT"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

cd workflow

case $STAGE in
    1)
        echo -e "${BLUE}Running Stage 1: Baseline Evaluation${NC}"
        echo "This will evaluate Source and TENT on all 75 conditions (15 corruptions × 5 severities)"
        echo "Estimated time: 2-4 hours on GPU, 12-24 hours on CPU"
        echo ""

        python evaluate_baselines.py \
            --data_root "../$DATA_ROOT" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --method both \
            --results_dir ../results

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Stage 1 complete${NC}"
        else
            echo -e "${RED}✗ Stage 1 failed${NC}"
            exit 1
        fi
        ;;

    2)
        echo -e "${YELLOW}Stage 2: FOA Implementation${NC}"
        echo "Not yet implemented"
        echo "This stage will implement:"
        echo "  - Forward-only prompt adaptation with CMA-ES"
        echo "  - Back-to-source activation shifting"
        echo "  - Combined fitness function optimization"
        exit 1
        ;;

    3)
        echo -e "${YELLOW}Stage 3: Comparative Evaluation${NC}"
        echo "Not yet implemented"
        echo "This stage will:"
        echo "  - Compare FOA vs. Source vs. TENT"
        echo "  - Run ablation studies"
        echo "  - Evaluate on secondary benchmarks"
        exit 1
        ;;

    all)
        echo -e "${BLUE}Running All Stages${NC}"
        echo "This will run stages 1, 2, and 3 sequentially"
        echo ""

        # Stage 1
        echo -e "${BLUE}Stage 1: Baseline Evaluation${NC}"
        python evaluate_baselines.py \
            --data_root "../$DATA_ROOT" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --method both \
            --results_dir ../results

        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Stage 1 failed${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Stage 1 complete${NC}"
        echo ""

        # Stage 2 (not yet implemented)
        echo -e "${YELLOW}Stage 2: Not yet implemented${NC}"
        exit 1
        ;;

    *)
        echo -e "${RED}Invalid stage: $STAGE${NC}"
        echo "Valid stages: 1, 2, 3, all"
        exit 1
        ;;
esac

cd ..

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Reproduction Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: results/"
echo ""
echo "Generated files:"
find results/ -type f -name "*.csv" -o -name "*.json" -o -name "*.png" 2>/dev/null | head -10
echo ""
echo -e "${GREEN}✓ Reproduction successful${NC}"
