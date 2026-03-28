#!/usr/bin/env bash
set -euo pipefail

# Installs a user-local CUDA toolkit (no root), configures ~/.bashrc,
# and prepares environment so `uv sync` can build/use flash-attn.

CUDA_VERSION="${CUDA_VERSION:-12.2.2}"
CUDA_RUNFILE_URL="${CUDA_RUNFILE_URL:-https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_VERSION}_535.104.05_linux.run}"
CUDA_INSTALL_DIR="${CUDA_INSTALL_DIR:-$HOME/.local/cuda-12.2}"
TORCH_CUDA_ARCH_LIST_VALUE="${TORCH_CUDA_ARCH_LIST_VALUE:-8.0}"
BASHRC_PATH="${BASHRC_PATH:-$HOME/.bashrc}"
RUNFILE_PATH="${RUNFILE_PATH:-$HOME/.cache/cuda_${CUDA_VERSION}.run}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --cuda-version <v>      CUDA version for runfile URL (default: ${CUDA_VERSION})
  --runfile-url <url>     Full NVIDIA runfile URL
  --install-dir <path>    Local CUDA install path (default: ${CUDA_INSTALL_DIR})
  --arch-list <list>      TORCH_CUDA_ARCH_LIST value (default: ${TORCH_CUDA_ARCH_LIST_VALUE})
  --bashrc <path>         Bashrc path to patch (default: ${BASHRC_PATH})
  -h, --help              Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --install-dir "$HOME/scratch/cuda-12.2" --arch-list 8.0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --runfile-url)
            CUDA_RUNFILE_URL="$2"
            shift 2
            ;;
        --install-dir)
            CUDA_INSTALL_DIR="$2"
            shift 2
            ;;
        --arch-list)
            TORCH_CUDA_ARCH_LIST_VALUE="$2"
            shift 2
            ;;
        --bashrc)
            BASHRC_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "$RUNFILE_PATH")"
mkdir -p "$CUDA_INSTALL_DIR"

if [[ ! -x "$CUDA_INSTALL_DIR/bin/nvcc" ]]; then
    echo "[1/4] Downloading CUDA runfile..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL "$CUDA_RUNFILE_URL" -o "$RUNFILE_PATH"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$RUNFILE_PATH" "$CUDA_RUNFILE_URL"
    else
        echo "Need curl or wget to download CUDA runfile." >&2
        exit 1
    fi

    echo "[2/4] Installing CUDA toolkit to: $CUDA_INSTALL_DIR"
    sh "$RUNFILE_PATH" \
        --silent \
        --toolkit \
        --toolkitpath="$CUDA_INSTALL_DIR" \
        --override
    # Reclaim ~4GB — on HPC clusters with quotas this matters
    echo "[cleanup] Removing runfile to reclaim disk space"
    rm -f "$RUNFILE_PATH"
else
    echo "[skip] nvcc already present at $CUDA_INSTALL_DIR/bin/nvcc"
fi

if [[ ! -x "$CUDA_INSTALL_DIR/bin/nvcc" ]]; then
    echo "CUDA installation appears incomplete: nvcc not found." >&2
    exit 1
fi

BEGIN_MARKER="# >>> LOCAL CUDA TOOLKIT FOR FLASH-ATTN (managed) >>>"
END_MARKER="# <<< LOCAL CUDA TOOLKIT FOR FLASH-ATTN (managed) <<<"

TMP_BASHRC="$(mktemp)"
if [[ -f "$BASHRC_PATH" ]]; then
    awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" '
        $0 == begin { skip=1; next }
        $0 == end { skip=0; next }
        !skip { print }
    ' "$BASHRC_PATH" > "$TMP_BASHRC"
else
    : > "$TMP_BASHRC"
fi

cat >> "$TMP_BASHRC" <<EOF

$BEGIN_MARKER
if [[ -d "$CUDA_INSTALL_DIR" ]]; then
    export CUDA_HOME="$CUDA_INSTALL_DIR"

    case ":\$PATH:" in
        *":\$CUDA_HOME/bin:"*) ;;
        *) export PATH="\$CUDA_HOME/bin:\$PATH" ;;
    esac

    case ":\${LD_LIBRARY_PATH:-}:" in
        *":\$CUDA_HOME/lib64:"*) ;;
        *) export LD_LIBRARY_PATH="\$CUDA_HOME/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" ;;
    esac

    export CUDACXX="\$CUDA_HOME/bin/nvcc"
    export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VALUE"
fi
$END_MARKER
EOF

mv "$TMP_BASHRC" "$BASHRC_PATH"

echo "[3/4] ~/.bashrc updated at: $BASHRC_PATH"

echo "[4/4] Verifying nvcc from local install"
"$CUDA_INSTALL_DIR/bin/nvcc" --version | head -n 5

echo
echo "Done. Next steps for the target account:"
echo "  1) source $BASHRC_PATH"
echo "  2) cd /path/to/multimodal_finetuned_representations"
echo "  3) uv sync"
echo "  4) python -c \"import flash_attn; print('flash_attn ok')\""
