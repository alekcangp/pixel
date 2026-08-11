#!/bin/bash
cd "$(dirname "$0")"

REPO_URL="https://github.com/alekcangp/pixel.git"
PROJECT_DIR="pixel"

# ---------- 0. Скачивание проекта ----------
if [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo "Проект не найден. Скачиваю $REPO_URL ..."
    if ! command -v git >/dev/null 2>&1; then
        echo "git не найден. Устанавливаю..."
        brew install git
    fi
    git clone --branch master "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
else
    cd "$PROJECT_DIR"
fi

REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12

# ---------- helpers ----------
log()  { echo -e "\033[1;34m[install]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[ok]\033[0m $*"; }
err()  { echo -e "\033[1;31m[error]\033[0m $*" >&2; }

# ---------- 1. Command Line Tools ----------
if ! xcode-select -p >/dev/null 2>&1; then
    log "Устанавливаю Xcode Command Line Tools..."
    xcode-select --install || true
    until xcode-select -p >/dev/null 2>&1; do
        sleep 2
    done
    ok "Command Line Tools готовы."
else
    ok "Command Line Tools уже установлены."
fi

# ---------- 2. Homebrew ----------
if ! command -v brew >/dev/null 2>&1; then
    log "Устанавливаю Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -d "/opt/homebrew/bin" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -d "/usr/local/bin" ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
else
    ok "Homebrew уже установлен: $(brew --version | head -1)"
fi

# ---------- 3. Python ----------
# Проект зафиксирован на Python 3.12 (python@3.12 из Homebrew).
# Если найдена другая (в т.ч. более новая) версия — принудительно ставим python@3.12.
if command -v python3 >/dev/null 2>&1; then
    PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [ "$PY_MAJOR" -gt "$REQUIRED_PYTHON_MAJOR" ] || { [ "$PY_MAJOR" -eq "$REQUIRED_PYTHON_MAJOR" ] && [ "$PY_MINOR" -ge "$REQUIRED_PYTHON_MINOR" ]; }; then
        # Проект зафиксирован на Python 3.12 — для более новых версий ставим python@3.12
        if [ "$PY_MAJOR" -gt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; }; then
            log "Python $PY_VER найден, но проект зафиксирован на Python 3.12. Устанавливаю Python 3.12..."
            brew install "python@${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}"
        else
            ok "Python $PY_VER найден."
        fi
    else
        log "Python $PY_VER устарел. Устанавливаю Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR..."
        brew install "python@${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}"
    fi
else
    log "Python не найден. Устанавливаю Python $REQUIRED_PYTHON_MAJOR.$REQUIRED_PYTHON_MINOR..."
    brew install "python@${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}"
fi

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
PYTHON_BIN="python3.12"

# ---------- 4. venv ----------
NEED_NEW_VENV=0
if [ ! -d ".venv" ]; then
    NEED_NEW_VENV=1
else
    # Если venv создан не под python3.12 — пересоздаём
    if [ ! -x ".venv/bin/python3.12" ]; then
        log "Виртуальное окружение создано под другой Python. Пересоздаю..."
        rm -rf ".venv"
        NEED_NEW_VENV=1
    fi
fi

if [ "$NEED_NEW_VENV" -eq 1 ]; then
    log "Создаю виртуальное окружение через python3.12..."
    "$PYTHON_BIN" -m venv .venv
else
    log "Виртуальное окружение .venv уже существует и подходит."
fi

source .venv/bin/activate

# ---------- 5. Зависимости ----------
if [ ! -f ".venv/.deps_installed" ]; then
    log "Устанавливаю зависимости Python..."
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    touch .venv/.deps_installed
else
    log "Зависимости уже установлены (флаг .venv/.deps_installed найден)."
fi

# ---------- 6. Модель SigLIP ----------
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

if [ ! -f ".venv/.model_downloaded" ]; then
    log "Скачиваю модель SigLIP в кэш..."
    python - <<'PY'
from transformers import AutoProcessor, AutoModel
model_id = "google/siglip-base-patch16-224"
AutoProcessor.from_pretrained(model_id)
AutoModel.from_pretrained(model_id)
print("Модель загружена:", model_id)
PY
    touch .venv/.model_downloaded
else
    log "Модель уже загружена (флаг .venv/.model_downloaded найден)."
fi

# ---------- 7. Запуск ----------
log "Запускаю Pixel UI..."
python main.py ui-flet
