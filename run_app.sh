#!/usr/bin/env bash
# YoloBliss - Lance l'application Streamlit
# Usage : bash run_app.sh [port]
#
# Crée automatiquement un environnement virtuel .venv si absent,
# installe les dépendances, puis démarre l'application.

set -euo pipefail

PORT="${1:-8501}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS="$SCRIPT_DIR/app/requirements.txt"
STAMP="$VENV_DIR/.deps_ok"

# ── 1. Trouver un Python système pour bootstrapper le venv ───────────────────
BOOTSTRAP_PY=""
IS_CONDA=false
for candidate in \
    "$HOME/miniconda/bin/python" \
    "$HOME/miniconda3/bin/python" \
    "$HOME/anaconda/bin/python" \
    "$HOME/anaconda3/bin/python" \
    "$(command -v python3 2>/dev/null)" \
    "$(command -v python 2>/dev/null)"; do
    if [[ -x "$candidate" ]]; then
        BOOTSTRAP_PY="$candidate"
        [[ "$candidate" == *miniconda* || "$candidate" == *anaconda* ]] && IS_CONDA=true
        break
    fi
done

if [[ -z "$BOOTSTRAP_PY" ]]; then
    echo "[ERREUR] Python introuvable. Installez Python 3.10+ ou conda."
    exit 1
fi

echo "[INFO] Python bootstrap : $($BOOTSTRAP_PY --version 2>&1)"

# ── 2. Créer le venv si absent ───────────────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/python" ]]; then
    echo "[INFO] Création de l'environnement virtuel : $VENV_DIR"
    if $IS_CONDA; then
        # Hérite torch/ultralytics/etc déjà installés dans conda → gain de temps
        "$BOOTSTRAP_PY" -m venv --system-site-packages "$VENV_DIR"
        echo "[INFO] Venv avec héritage conda (torch/ultralytics réutilisés)."
    else
        "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
        echo "[INFO] Venv isolé créé (installation complète requise)."
    fi
fi

PYTHON="$VENV_DIR/bin/python"

# ── 3. Installer / mettre à jour les dépendances ─────────────────────────────
# Re-installe si le fichier requirements a changé depuis la dernière install
if [[ ! -f "$STAMP" ]] || [[ "$REQUIREMENTS" -nt "$STAMP" ]]; then
    echo "[INFO] Installation des dépendances depuis $REQUIREMENTS ..."
    "$PYTHON" -m pip install --upgrade pip --quiet
    "$PYTHON" -m pip install -r "$REQUIREMENTS" --quiet
    touch "$STAMP"
    echo "[OK]   Dépendances installées."
else
    echo "[INFO] Dépendances à jour (supprimez .venv pour forcer la réinstallation)."
fi

# ── 4. Libérer le port si déjà occupé ────────────────────────────────────────
if fuser "${PORT}/tcp" &>/dev/null; then
    echo "[INFO] Port ${PORT} occupé — arrêt de l'ancienne instance..."
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 1
fi

# ── 5. Lancer l'application ───────────────────────────────────────────────────
echo ""
echo "[INFO] Démarrage de l'application sur http://localhost:${PORT}"
echo "       Ctrl+C pour arrêter."
echo ""
exec "$PYTHON" -m streamlit run "$SCRIPT_DIR/app/app.py" \
    --server.port "$PORT" \
    --server.headless true
