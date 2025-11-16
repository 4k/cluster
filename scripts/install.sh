#!/bin/bash
# Voice Assistant Installation Script Wrapper
# This script provides a convenient way to run the Python installation script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

print_status "Using Python command: $PYTHON_CMD"

# Check if we're in the right directory
if [ ! -f "scripts/install.py" ]; then
    print_error "install.py not found. Please run this script from the project root directory."
    exit 1
fi

# Parse command line arguments
VERBOSE=""
SKIP_STEPS=""
MODELS_DIR=""
CONFIG_DIR=""
DATA_DIR=""
VOICES_DIR=""
LOGS_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --skip)
            SKIP_STEPS="--skip $2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="--models-dir $2"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="--config-dir $2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="--data-dir $2"
            shift 2
            ;;
        --voices-dir)
            VOICES_DIR="--voices-dir $2"
            shift 2
            ;;
        --logs-dir)
            LOGS_DIR="--logs-dir $2"
            shift 2
            ;;
        --help|-h)
            echo "Voice Assistant Installation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --verbose, -v           Enable verbose logging"
            echo "  --skip STEPS           Skip installation steps (space-separated)"
            echo "  --models-dir DIR       Models directory (default: models)"
            echo "  --config-dir DIR       Configuration directory (default: config)"
            echo "  --data-dir DIR         Data directory (default: data)"
            echo "  --voices-dir DIR       Voices directory (default: voices)"
            echo "  --logs-dir DIR         Logs directory (default: logs)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Available steps to skip:"
            echo "  create_directories, download_default_model, setup_configuration, verify_installation"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Full installation"
            echo "  $0 --verbose                          # Verbose installation"
            echo "  $0 --skip download_default_model      # Skip model download"
            echo "  $0 --models-dir /custom/models        # Custom models directory"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the installation
print_status "Starting Voice Assistant installation..."
print_status "Arguments: $VERBOSE $SKIP_STEPS $MODELS_DIR $CONFIG_DIR $DATA_DIR $VOICES_DIR $LOGS_DIR"

if $PYTHON_CMD scripts/install.py $VERBOSE $SKIP_STEPS $MODELS_DIR $CONFIG_DIR $DATA_DIR $VOICES_DIR $LOGS_DIR; then
    print_success "Installation completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "  1. Run the application: $PYTHON_CMD src/main.py"
    echo "  2. Or use Docker: docker-compose up"
    echo "  3. Check the logs in the logs/ directory"
else
    print_error "Installation failed!"
    exit 1
fi
