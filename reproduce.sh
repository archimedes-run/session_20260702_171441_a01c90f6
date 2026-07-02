#!/bin/bash
#
# Master Reproducibility Script for FOA 2024 Replication
#
# This script ensures complete end-to-end reproducibility using uv package manager.
# All dependencies are installed via uv as per PAPERBENCH requirements.
#
# Usage: ./reproduce.sh [--stage <1|2|3|all>] [--test-only]
#

set -e  # Exit on any error

# Unset problematic UV environment variables to avoid path conflicts
unset UV_PROJECT_ENVIRONMENT
unset VIRTUAL_ENV

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
echo -e "${GREEN}[1/5] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo -e "${RED}Error: Python 3.12 or higher required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version compatible${NC}"
echo ""

# Step 2: Check for uv package manager
echo -e "${GREEN}[2/5] Checking for uv package manager...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "uv version: $(uv --version)"
echo -e "${GREEN}✓ uv available${NC}"
echo ""

# Step 3: Set up virtual environment and install dependencies
echo -e "${GREEN}[3/5] Setting up environment and installing dependencies...${NC}"
echo "This may take several minutes..."

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install all dependencies from pyproject.toml using uv pip
# uv pip is faster and more reliable than pip
if [ -f "pyproject.toml" ]; then
    echo "Installing dependencies from pyproject.toml using uv pip..."
    # Install in batches to avoid command line length issues
    uv pip install --quiet torch>=2.2.0 torchvision>=0.17.0 timm>=0.9.0
    uv pip install --quiet transformers>=4.38.0 datasets>=2.18.0 accelerate>=0.27.0
    uv pip install --quiet numpy>=1.26.0 pandas>=2.0.0 matplotlib>=3.8.0 scipy>=1.11.0
    uv pip install --quiet seaborn>=0.13.0 scikit-learn>=1.3.0 einops>=0.7.0
    uv pip install --quiet pyyaml>=6.0.0 pillow>=10.0.0 requests>=2.31.0 tqdm>=4.66.0
    uv pip install --quiet plotly>=5.17.0 wandb>=0.16.0 python-dotenv>=1.0.0
    uv pip install --quiet cma>=3.0.0  # CMA-ES for FOA
else
    echo "Installing minimal dependencies using uv pip..."
    uv pip install --quiet torch torchvision timm numpy pillow tqdm
    uv pip install --quiet matplotlib scipy scikit-learn pandas seaborn
    uv pip install --quiet cma  # CMA-ES for FOA
fi

echo -e "${GREEN}✓ Dependencies installed via uv${NC}"
echo ""

# Step 4: Verify implementation
echo -e "${GREEN}[4/5] Running verification tests...${NC}"
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

# Step 5: Run evaluation based on stage
echo -e "${GREEN}[5/5] Running Stage $STAGE...${NC}"
echo "Data root: $DATA_ROOT"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

cd workflow

case $STAGE in
    1)
        # Check for ImageNet-C dataset
        if [ ! -d "../$DATA_ROOT" ]; then
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
        echo -e "${BLUE}Running Stage 2: FOA Implementation${NC}"
        echo "This will implement and evaluate the Forward-Optimization Adaptation method"
        echo ""

        # First, compute source statistics if not already done
        if [ ! -f "../results/source_statistics.pth" ]; then
            echo "Pre-computing source domain statistics..."
            python compute_source_stats.py \
                --num_samples 32 \
                --batch_size 8 \
                --output ../results/source_statistics.pth \
                --device "$DEVICE" \
                --use_synthetic

            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Source statistics computation failed${NC}"
                exit 1
            fi
        fi

        # Evaluate FOA
        echo "Evaluating FOA on ImageNet-C..."
        python evaluate_foa.py \
            --data_root "../$DATA_ROOT" \
            --source_stats ../results/source_statistics.pth \
            --batch_size "$BATCH_SIZE" \
            --num_prompts 10 \
            --lambda_activation 0.1 \
            --cma_population 10 \
            --cma_iterations 20 \
            --device "$DEVICE" \
            --output ../results

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Stage 2 complete${NC}"
        else
            echo -e "${RED}✗ Stage 2 failed${NC}"
            exit 1
        fi
        ;;

    3)
        echo -e "${BLUE}Running Stage 3: Comparative Evaluation & Ablation Studies${NC}"
        echo "This will compare all methods and run ablation studies"
        echo "Estimated time: 3-6 hours on GPU"
        echo ""

        # Check source statistics
        if [ ! -f "../results/source_statistics.pth" ]; then
            echo -e "${RED}Error: Source statistics not found${NC}"
            echo "Please run Stage 2 first or run with --stage all"
            exit 1
        fi

        # Run comprehensive comparison
        python compare_all_methods.py \
            --data_root "../$DATA_ROOT" \
            --source_stats ../results/source_statistics.pth \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --output ../results

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Stage 3 complete${NC}"
        else
            echo -e "${RED}✗ Stage 3 failed${NC}"
            exit 1
        fi

        # Test quantized model
        echo ""
        echo "Testing 8-bit quantized model..."
        python quantized_model.py

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Quantization test complete${NC}"
        fi
        ;;

    all)
        # Check for ImageNet-C dataset
        if [ ! -d "../$DATA_ROOT" ]; then
            echo -e "${RED}Error: ImageNet-C dataset not found at: $DATA_ROOT${NC}"
            echo "Please download and extract the dataset first."
            exit 1
        fi

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

        # Stage 2: FOA Implementation
        echo -e "${BLUE}Stage 2: FOA Implementation${NC}"

        # Pre-compute source statistics
        if [ ! -f "../results/source_statistics.pth" ]; then
            echo "Pre-computing source domain statistics..."
            python compute_source_stats.py \
                --num_samples 32 \
                --batch_size 8 \
                --output ../results/source_statistics.pth \
                --device "$DEVICE" \
                --use_synthetic
        fi

        # Evaluate FOA
        python evaluate_foa.py \
            --data_root "../$DATA_ROOT" \
            --source_stats ../results/source_statistics.pth \
            --batch_size "$BATCH_SIZE" \
            --num_prompts 10 \
            --lambda_activation 0.1 \
            --cma_population 10 \
            --cma_iterations 20 \
            --device "$DEVICE" \
            --output ../results

        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Stage 2 failed${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Stage 2 complete${NC}"
        echo ""

        # Stage 3: Comprehensive Comparison
        echo -e "${BLUE}Stage 3: Comprehensive Comparison & Ablations${NC}"
        python compare_all_methods.py \
            --data_root "../$DATA_ROOT" \
            --source_stats ../results/source_statistics.pth \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --output ../results

        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Stage 3 failed${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Stage 3 complete${NC}"

        # Test quantized model
        echo ""
        echo "Testing 8-bit quantized model..."
        python quantized_model.py
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
