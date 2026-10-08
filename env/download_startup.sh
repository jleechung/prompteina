#!/bin/bash
# =============================================================================
# Protein Foundation Models - Startup Script
# =============================================================================
# This script downloads all required model weights for the protein foundation
# models project. Run this from the project root directory.
# =============================================================================

# Modified by the Prompteina authors in 2026.
# Changes are licensed under Apache 2.0; see LICENSE and NOTICE.

set -e

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Helper Functions
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                   ║"
    echo "║          🧬  Protein Foundation Models - Setup Wizard  🧬         ║"
    echo "║                                                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${WHITE}  $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_progress() {
    echo -e "${CYAN}→${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed. Please install it and try again."
        return 1
    fi
    return 0
}

# =============================================================================
# Download Functions
# =============================================================================

download_pmpnn_weights() {
    print_section "Downloading ProteinMPNN Weights"
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d "community_models/ProteinMPNN" ]; then
        print_error "ProteinMPNN directory not found at $PROJECT_ROOT/community_models/ProteinMPNN"
        return 1
    fi
    
    cd community_models/ProteinMPNN
    mkdir -p ca_model_weights vanilla_model_weights
    
    local downloaded=0
    local skipped=0
    
    # Download CA model weights
    print_progress "Checking CA model weights..."
    cd ca_model_weights
    local ca_files=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt")
    for file in "${ca_files[@]}"; do
        if [ -f "$file" ] && [ -s "$file" ]; then
            print_info "  ✓ $file already exists, skipping"
            skipped=$((skipped + 1))
        else
            print_info "  Downloading $file..."
            wget -q --show-progress "https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/$file" -O "$file"
            downloaded=$((downloaded + 1))
        fi
    done
    cd ..
    
    # Download vanilla model weights
    print_progress "Checking vanilla model weights..."
    cd vanilla_model_weights
    local vanilla_files=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt" "v_48_030.pt")
    for file in "${vanilla_files[@]}"; do
        if [ -f "$file" ] && [ -s "$file" ]; then
            print_info "  ✓ $file already exists, skipping"
            skipped=$((skipped + 1))
        else
            print_info "  Downloading $file..."
            wget -q --show-progress "https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/$file" -O "$file"
            downloaded=$((downloaded + 1))
        fi
    done
    cd ..
    
    if [ $downloaded -eq 0 ]; then
        print_success "ProteinMPNN weights already installed ($skipped files)"
    else
        print_success "ProteinMPNN weights: downloaded $downloaded, skipped $skipped"
    fi
    cd "$PROJECT_ROOT"
}

download_complexa_weights() {
    print_section "Downloading Complexa Weights"
    
    cd "$PROJECT_ROOT"
    
    local ckpt_dir="${COMPLEXA_CKPT_DIR:-./ckpts}"
    local fm_ckpt="${ckpt_dir}/complexa.ckpt"
    local ae_ckpt="${ckpt_dir}/complexa_ae.ckpt"
    
    # Check if weights already exist
    if [ -f "$fm_ckpt" ] && [ -s "$fm_ckpt" ] && [ -f "$ae_ckpt" ] && [ -s "$ae_ckpt" ]; then
        print_success "Complexa weights already installed, skipping download"
        return 0
    fi
    
    print_progress "Creating ckpts directory..."
    mkdir -p "${ckpt_dir}" || { print_error "Failed to create ckpts directory"; return 1; }
    
    local ngc_model="proteina_complexa"
    local ngc_base="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/${ngc_model}/1.0/files?redirect=true&path="

    if [ ! -f "$fm_ckpt" ] || [ ! -s "$fm_ckpt" ]; then
        print_progress "Downloading complexa.ckpt (flow matching model)..."
        wget --content-disposition --show-progress -O "$fm_ckpt" \
            "${ngc_base}complexa.ckpt" || {
            print_error "Failed to download complexa.ckpt"
            return 1
        }
        [ -s "$fm_ckpt" ] || { print_error "Downloaded complexa.ckpt is empty"; return 1; }
    fi

    if [ ! -f "$ae_ckpt" ] || [ ! -s "$ae_ckpt" ]; then
        print_progress "Downloading complexa_ae.ckpt (autoencoder)..."
        wget --content-disposition --show-progress -O "$ae_ckpt" \
            "${ngc_base}complexa_ae.ckpt" || {
            print_error "Failed to download complexa_ae.ckpt"
            return 1
        }
        [ -s "$ae_ckpt" ] || { print_error "Downloaded complexa_ae.ckpt is empty"; return 1; }
    fi

    print_success "Complexa weights downloaded to ckpts/ successfully!"
}

# =============================================================================
# Status Check Functions
# =============================================================================

check_pmpnn_status() {
    local missing=()
    local ca_files=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt")
    local vanilla_files=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt" "v_48_030.pt")
    
    for file in "${ca_files[@]}"; do
        [ ! -f "$PROJECT_ROOT/community_models/ProteinMPNN/ca_model_weights/$file" ] && missing+=("ca_model_weights/$file")
    done
    for file in "${vanilla_files[@]}"; do
        [ ! -f "$PROJECT_ROOT/community_models/ProteinMPNN/vanilla_model_weights/$file" ] && missing+=("vanilla_model_weights/$file")
    done
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (community_models/ProteinMPNN/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (community_models/ProteinMPNN/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

check_complexa_status() {
    local missing=()
    local ckpt_dir="$PROJECT_ROOT/ckpts"
    
    [ ! -f "$ckpt_dir/complexa.ckpt" ] && missing+=("complexa.ckpt (flow matching model)")
    [ ! -f "$ckpt_dir/complexa_ae.ckpt" ] && missing+=("complexa_ae.ckpt (autoencoder)")
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ Installed (ckpts/)${NC}"
    else
        echo -e "${YELLOW}○ Missing (ckpts/):${NC}"
        for f in "${missing[@]}"; do
            echo -e "      ${RED}✗${NC} $f"
        done
    fi
}

print_status() {
    print_section "Current Installation Status"
    echo -e "  ${BOLD}Complexa (required):${NC} $(check_complexa_status)"
    echo -e "  ${BOLD}ProteinMPNN (needed for --sequences mpnn / mpnn-fi):${NC} $(check_pmpnn_status)"
    echo ""
}

# =============================================================================
# Interactive Menu
# =============================================================================

show_menu() {
    echo -e "${BOLD}${WHITE}What would you like to download?${NC}
"
    echo -e "  ${CYAN}c${NC}) ${BOLD}Complexa${NC} base denoiser and autoencoder (required)"
    echo -e "  ${CYAN}m${NC}) ${BOLD}ProteinMPNN${NC} weights (~50MB), for the mpnn and mpnn-fi eval conditions"
    echo -e "  ${CYAN}b${NC}) ${BOLD}Both${NC}"
    echo -e ""
    echo -e "  ${CYAN}s${NC}) Check installation ${BOLD}status${NC}"
    echo -e "  ${CYAN}q${NC}) ${BOLD}Quit${NC}"
    echo ""
    echo -ne "${YELLOW}Enter your choice [c, m, b, s, q]: ${NC}"
}

run_interactive() {
    while true; do
        show_menu
        read -r choice
        
        case $choice in
            c|C)
                download_complexa_weights || true
                ;;
            m|M)
                download_pmpnn_weights
                ;;
            b|B)
                download_complexa_weights || true
                download_pmpnn_weights
                print_section "Downloads Complete"
                print_status
                ;;
            s|S)
                print_status
                ;;
            q|Q)
                echo -e "\n${GREEN}Enjoy designing!${NC}\n"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please try again."
                ;;
        esac
        
        echo ""
        echo -ne "${YELLOW}Press Enter to continue...${NC}"
        read -r
        clear
        print_banner
    done
}

# =============================================================================
# CLI Mode
# =============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --complexa      Download the Complexa base denoiser and autoencoder."
    echo "                  AE_CKPT must point at the autoencoder; both are required."
    echo "  --pmpnn         Download ProteinMPNN weights (~50MB), needed only for the"
    echo "                  mpnn and mpnn-fi sequence conditions in evaluate_binders."
    echo "  --all           Both of the above."
    echo "  --status        Show installation status."
    echo "  --help, -h      Show this help message."
    echo ""
    echo "The Prompteina checkpoints are on Hugging Face, not here:"
    echo "  hf download jleechung/prompteina --local-dir ./weights"
    echo ""
    echo "If no options are provided, runs in interactive mode."
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Check for required commands
    check_command wget || exit 1
    check_command tar || exit 1
    
    # Change to project root
    cd "$PROJECT_ROOT"
    print_info "Working directory: $PROJECT_ROOT"
    
    # Parse arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        clear
        print_banner
        print_status
        run_interactive
    else
        # CLI mode
        print_banner
        
        for arg in "$@"; do
            case $arg in
                --complexa)
                    download_complexa_weights || true
                    ;;
                --pmpnn)
                    download_pmpnn_weights
                    ;;
                --all|--everything)
                    download_complexa_weights || true
                    download_pmpnn_weights
                    print_section "Downloads Complete"
                    ;;
                --status)
                    print_status
                    ;;
                --help|-h)
                    show_help
                    exit 0
                    ;;
                *)
                    print_error "Unknown option: $arg"
                    show_help
                    exit 1
                    ;;
            esac
        done
        
        print_status
    fi
}

main "$@"

