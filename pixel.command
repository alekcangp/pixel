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
            python3 -m venv .venv || fail "failed to create venv"
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
