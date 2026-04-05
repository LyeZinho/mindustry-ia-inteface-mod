#!/bin/bash
set -e

# Training runner with venv activation, dependency check, and process supervision

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$REPO_ROOT/venv"
CONFIG_FILE="$SCRIPT_DIR/training_config.env"
LOGS_DIR="$REPO_ROOT/rl/logs_v2"
MODELS_DIR="$REPO_ROOT/rl/models_v2"

# Source defaults
if [ -f "$CONFIG_FILE" ]; then
    set +e  # Don't exit if sourcing fails
    source "$CONFIG_FILE"
    set -e
fi

# Default values if not set
TRAINING_MAPS="${TRAINING_MAPS:-Ancient Caldera}"
TRAINING_TIMESTEPS="${TRAINING_TIMESTEPS:-1000000}"
TRAINING_LR="${TRAINING_LR:-0.0003}"
DASHBOARD_PORT="${DASHBOARD_PORT:-5000}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"
MIMI_HOST="${MIMI_HOST:-localhost}"
MIMI_PORT="${MIMI_PORT:-9000}"

# Override with CLI args
if [ $# -ge 1 ]; then
    TRAINING_MAPS="$1"
fi
if [ $# -ge 2 ]; then
    TRAINING_TIMESTEPS="$2"
fi
if [ $# -ge 3 ]; then
    TRAINING_LR="$3"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 A2C Training Runner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 Maps: $TRAINING_MAPS"
echo "⏱️  Steps: $TRAINING_TIMESTEPS"
echo "📈 LR: $TRAINING_LR"
echo "📊 Dashboard: http://localhost:$DASHBOARD_PORT"
echo ""

# Check venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "   Create with: python -m venv $VENV_PATH"
    exit 1
fi

PYTHON="$VENV_PATH/bin/python"
PIP="$VENV_PATH/bin/pip"

# Check Python executable
if [ ! -x "$PYTHON" ]; then
    echo "❌ Python executable not found: $PYTHON"
    exit 1
fi

echo "✅ Found venv: $VENV_PATH"
echo "✅ Using Python: $($PYTHON --version 2>&1)"

# Install/update dependencies
echo ""
echo "📦 Checking dependencies..."
$PIP install -q -r "$SCRIPT_DIR/requirements_simple.txt" 2>/dev/null || {
    echo "⚠️  pip install had issues, but continuing..."
}
echo "✅ Dependencies ready"

# Create output directories
mkdir -p "$LOGS_DIR" "$MODELS_DIR"

# Export for dashboard
export TRAINING_LOGS_DIR="$LOGS_DIR"
export TRAINING_MODELS_DIR="$MODELS_DIR"

# Start dashboard in background
echo ""
echo "🚀 Starting dashboard on port $DASHBOARD_PORT..."
DASHBOARD_LOG="$LOGS_DIR/dashboard_$(date +%Y%m%d_%H%M%S).log"
$PYTHON -m rl.dashboard \
    --host localhost \
    --port "$DASHBOARD_PORT" \
    > "$DASHBOARD_LOG" 2>&1 &
DASHBOARD_PID=$!
echo "✅ Dashboard started (PID: $DASHBOARD_PID) - logs: $DASHBOARD_LOG"

# Wait for dashboard to be ready
sleep 2

# Trap Ctrl+C to kill both processes
cleanup() {
    echo ""
    echo "⏹️  Shutting down..."
    kill $DASHBOARD_PID 2>/dev/null || true
    kill $TRAINING_PID 2>/dev/null || true
    wait $DASHBOARD_PID 2>/dev/null || true
    wait $TRAINING_PID 2>/dev/null || true
    echo "✅ Cleanup complete"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Training loop with restart logic
RESTART_COUNT=0
RESTART_DELAYS=(5 10 30)

while true; do
    # Log file for this run
    TRAINING_LOG="$LOGS_DIR/train_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📚 Training Run #$((RESTART_COUNT + 1)) - $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Logs: $TRAINING_LOG"
    echo "Dashboard: http://localhost:$DASHBOARD_PORT"
    echo ""
    
    # Run training
    $PYTHON -m rl.train_simple \
        --maps "$TRAINING_MAPS" \
        --timesteps "$TRAINING_TIMESTEPS" \
        --lr "$TRAINING_LR" \
        --host "$MIMI_HOST" \
        --port "$MIMI_PORT" \
        --models-dir "$MODELS_DIR" \
        --logs-dir "$LOGS_DIR" \
        2>&1 | tee -a "$TRAINING_LOG" &
    
    TRAINING_PID=$!
    echo "✅ Training started (PID: $TRAINING_PID)"
    
    # Wait for training to finish
    wait $TRAINING_PID
    TRAINING_EXIT=$?
    
    if [ $TRAINING_EXIT -eq 0 ]; then
        echo "✅ Training completed successfully"
        break
    elif [ $TRAINING_EXIT -eq 130 ] || [ $TRAINING_EXIT -eq 143 ]; then
        # Ctrl+C or SIGTERM
        echo "⏸️  Training interrupted by user"
        break
    else
        # Training crashed
        RESTART_COUNT=$((RESTART_COUNT + 1))
        
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "❌ Max restarts ($MAX_RESTARTS) reached. Giving up."
            break
        fi
        
        # Get restart delay
        DELAY_INDEX=$((RESTART_COUNT - 1))
        if [ $DELAY_INDEX -lt ${#RESTART_DELAYS[@]} ]; then
            DELAY=${RESTART_DELAYS[$DELAY_INDEX]}
        else
            DELAY=30
        fi
        
        echo "❌ Training exited with code $TRAINING_EXIT"
        echo "⏳ Restarting in ${DELAY}s (restart $RESTART_COUNT/$MAX_RESTARTS)..."
        sleep "$DELAY"
    fi
done

# Cleanup
cleanup
