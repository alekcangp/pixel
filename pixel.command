#!/bin/bash
# ============================================================
#  Pixel - unified install/run/uninstall script (macOS)
#
#  Usage:
#    ./pixel.command            install (if needed) + run
#    ./pixel.command install    install only (venv + deps + model)
#    ./pixel.command run        run only
#    ./pixel.command uninstall  uninstall (venv, DB, model cache)
# ============================================================
cd "$(dirname "$0")"

CMD="${1:-}"

log() { echo -e "\033[1;34m[Pixel]\033[0m $*"; }
fail() { echo -e "\033[1;31m[Pixel] Error:\033[0m $*" >&2; read -p "Press any key to exit..."; exit 1; }

auto_install_python_macos() {
    local PKG_URL="https://www.python.org/ftp/python/3.12.9/python-3.12.9-macos11.pkg"
    local PKG_FILE="$TMPDIR/python-3.12.9-macos11.pkg"
    log "Downloading Python 3.12.9 installer..."
    if ! curl -fsSL "$PKG_URL" -o "$PKG_FILE"; then
        log "Download failed. Opening Python.org download page..."
        open "https://www.python.org/downloads/release/python-3129/"
        read -p "Press any key to exit..."
        return 1
    fi
    log "Installing Python 3.12.9 silently..."
    if ! sudo installer -pkg "$PKG_FILE" -target /; then
        log "Silent install failed. Opening Python.org download page..."
        open "https://www.python.org/downloads/release/python-3129/"
        read -p "Press any key to exit..."
        return 1
    fi
    rm -f "$PKG_FILE"
    return 0
}

# Detect available Python executable
PYTHON_CMD=""
for p in python3 python python3.12; do
    if command -v "$p" >/dev/null 2>&1; then
        PYTHON_CMD="$p"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    log "Python not found. Installing Python 3.12..."
    if command -v brew >/dev/null 2>&1; then
        log "Homebrew found. Installing python@3.12..."
        brew install python@3.12 || fail "brew install python@3.12 failed"
        for p in python3 python python3.12; do
            if command -v "$p" >/dev/null 2>&1; then
                PYTHON_CMD="$p"
                break
            fi
        done
    else
        log "Homebrew not found. Installing Python 3.12 from python.org..."
        if ! auto_install_python_macos; then
            fail "Python install failed. Restart the terminal and run ./pixel.command again"
        fi
        for p in python3 python python3.12; do
            if command -v "$p" >/dev/null 2>&1; then
                PYTHON_CMD="$p"
                break
            fi
        done
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    fail "Python not found after install attempt. Restart the terminal and run ./pixel.command again"
fi

echo "[Pixel] Using Python: $PYTHON_CMD"

# No argument: install (if venv missing) + run
if [ -z "$CMD" ]; then
    if [ ! -x ".venv/bin/python3" ]; then
        CMD="install"
    else
        CMD="run"
    fi
fi

case "$CMD" in
    install)
        log "Setting up virtual environment..."
        if [ ! -x ".venv/bin/python3" ]; then
            "$PYTHON_CMD" -m venv .venv || fail "failed to create venv"
        fi
        source .venv/bin/activate

        log "Installing dependencies..."
        python -m pip install --upgrade pip
        pip install -r requirements.txt || fail "pip install"

        log "Downloading SigLIP model (first run)..."
        python - <<'PY' || fail "model download"
from transformers import AutoProcessor, AutoModel
m = "google/siglip-base-patch16-224"
AutoProcessor.from_pretrained(m)
AutoModel.from_pretrained(m)
print("Model loaded:", m)
PY
        log "Setup complete."
        read -p "Press any key to exit..."
        ;;
    run)
        if [ ! -x ".venv/bin/python3" ]; then
            fail "environment not found. Run: ./pixel.command install"
        fi
        log "Launching Pixel..."
        ./.venv/bin/python main.py ui-flet
        read -p "Press any key to exit..."
        ;;
    uninstall)
        log "Removing virtual environment..."
        rm -rf ".venv"
        log "Removing database storage/*.db..."
        rm -f storage/*.db storage/*.db-journal storage/*.db-wal storage/*.db-shm 2>/dev/null || true
        log "Removing HuggingFace model cache..."
        rm -rf ~/.cache/huggingface/hub/models--google--siglip-base-patch16-224 2>/dev/null || true
        rm -rf ~/.cache/torch/transformers/google--siglip-base-patch16-224 2>/dev/null || true
        log "Uninstall complete."
        read -p "Press any key to exit..."
        ;;
    *)
        fail "unknown command '$CMD'. Use: install | run | uninstall"
        ;;
esac
