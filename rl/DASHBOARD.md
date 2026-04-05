# Training Dashboard & Auto-Run System

Real-time monitoring of A2C training with automatic process management and recovery.

## Quick Start

### 1. Install dependencies (one-time)

```bash
pip install -r rl/requirements_simple.txt
```

### 2. Start training with dashboard

```bash
# Start with defaults (Ancient Caldera, 1M steps)
bash rl/run_training.sh

# Or customize maps and timesteps
bash rl/run_training.sh "Ancient Caldera,Glacier" 2000000

# Or customize learning rate too
bash rl/run_training.sh "Ancient Caldera,Glacier,Wasteland" 2000000 0.0005
```

### 3. Open dashboard

Navigate to **http://localhost:5000** in your browser

You'll see real-time plots of:
- Episode reward (running average)
- Policy loss
- Value loss  
- Steps per second

## System Components

### Dashboard (rl/dashboard.py)

Flask web server that reads TensorBoard event files and serves interactive plots.

**Features:**
- Auto-refresh every 2 seconds
- Responsive grid layout
- Plotly.js for smooth, interactive charts
- Lightweight (~50KB transfer per update)

**Runs on:** `http://localhost:5000` (configurable via `DASHBOARD_PORT`)

### Training Runner (rl/run_training.sh)

Bash script that orchestrates training with automatic features:

- **Venv activation**: Automatically uses `/venv/bin/python`
- **Dependency check**: Installs `rl/requirements_simple.txt` if needed
- **Dashboard supervision**: Starts dashboard in background
- **Process logging**: All output to `rl/logs_v2/train_*.log`
- **Auto-restart**: Restarts training on crash with exponential backoff
  - 1st restart: 5s delay
  - 2nd restart: 10s delay
  - 3rd+ restarts: 30s delay
  - Max 10 restarts before giving up
- **Graceful shutdown**: Ctrl+C stops both dashboard and training cleanly

### Configuration (rl/training_config.env)

Default parameters stored in `rl/training_config.env`. Override via CLI args:

```bash
# These override config file defaults
bash rl/run_training.sh "Ancient Caldera,Glacier" 2000000 0.0003
```

| Parameter | Default | Override |
|-----------|---------|----------|
| Maps | "Ancient Caldera" | arg 1 |
| Timesteps | 1000000 | arg 2 |
| Learning rate | 0.0003 | arg 3 |
| Dashboard port | 5000 | `DASHBOARD_PORT` env var |
| Max restarts | 10 | `MAX_RESTARTS` env var |

## Usage Examples

### Basic training on one map

```bash
bash rl/run_training.sh
```

### Training on multiple maps for robustness

```bash
bash rl/run_training.sh "Ancient Caldera,Glacier,Wasteland"
```

### Extended training session

```bash
bash rl/run_training.sh "Ancient Caldera,Glacier" 5000000 0.0002
```

### Monitoring logs while training

```bash
# In another terminal
tail -f rl/logs_v2/train_*.log

# Or watch training PID
ps aux | grep train_simple
```

### Resuming from checkpoint

```bash
# Train with resume flag
bash rl/run_training.sh "Ancient Caldera,Glacier" 2000000
```

Then in Python, use `--resume` flag in train_simple.py:

```python
python -m rl.train_simple --maps "Ancient Caldera,Glacier" --resume
```

## Troubleshooting

### Dashboard won't start

**Error:** `Address already in use`

**Fix:** Change port in config or environment:

```bash
DASHBOARD_PORT=5001 bash rl/run_training.sh
```

### Dashboard shows no data

**Reasons:**
1. Training just started - wait 10s for first metrics
2. Wrong logs directory - check `TRAINING_LOGS_DIR` is set
3. TensorBoard events not written yet

**Fix:** Check logs:

```bash
ls -la rl/logs_v2/  # Should have events.out.tfevents.* files
tail -f rl/logs_v2/train_*.log  # Check training output
```

### Training crashes repeatedly

**Check max restarts:** Edit `MAX_RESTARTS` in `rl/training_config.env`

**Check logs:** 

```bash
cat rl/logs_v2/train_*.log | grep -i error
```

### Venv not found

**Error:** `Virtual environment not found at: /path/to/venv`

**Fix:** Create venv first:

```bash
python -m venv venv
```

## Architecture Diagram

```
run_training.sh
├─ Source training_config.env (defaults)
├─ Activate venv
├─ Install requirements_simple.txt
├─ Create rl/logs_v2, rl/models_v2 dirs
├─ Start dashboard.py (Flask server)
│  └─ Reads: rl/logs_v2/events.out.tfevents.*
│     └─ Serves: http://localhost:5000
└─ Start train_simple.py
   └─ Writes: rl/logs_v2/events.out.tfevents.*
      └─ Dashboard reads & displays live
```

## Performance Notes

- **Dashboard overhead:** <2% CPU, ~1MB memory
- **Metric update lag:** 1-2 seconds behind training
- **TensorBoard event parsing:** Incremental (only reads new events)
- **Browser memory:** ~50MB for large training runs

Metrics are written by A2CTrainer every 100 steps by default. Increase frequency by modifying `rl/agent/trainer.py` `self.writer.add_scalar()` calls.

## See Also

- [TRAINING_SIMPLE.md](TRAINING_SIMPLE.md) - A2C training guide
- [test_agent.py](tests/test_agent.py) - Unit tests
- [requirements_simple.txt](requirements_simple.txt) - Dependencies
