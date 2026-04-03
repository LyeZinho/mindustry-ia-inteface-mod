# Training Dashboard Design

**Date:** 2026-04-03
**Status:** Approved

## Goal

Standalone matplotlib window that shows live training progress while `rl/train.py` runs, reading
`rl/logs/monitor.monitor.csv` without any interaction with the training process.

## Interface

Single Python module: `rl/dashboard.py`, run as `python -m rl.dashboard`.

### Layout — 4 panels, 1200×700 window

```
┌─────────────────────────┬─────────────────┐
│  Reward por episódio     │  Episode Length  │
│  (scatter + rolling 50)  │  (rolling 50)    │
├──────────────────────────┴─────────────────┤
│  Histograma de rewards                      │
├─────────────────────────────────────────────┤
│  Stats text: total eps | mean reward (50ep) │
│  max reward | mean length | last update     │
└─────────────────────────────────────────────┘
```

### Data flow

1. `FuncAnimation` fires every `--interval` seconds (default 3)
2. Read CSV with `pandas.read_csv`, skipping the `#{...}` header line
3. Compute rolling window of 50 episodes
4. Re-draw all 4 panels

### CLI

```
python -m rl.dashboard [--csv PATH] [--interval SECS] [--window INT]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | `rl/logs/monitor.monitor.csv` | Path to monitor CSV |
| `--interval` | 3 | Seconds between refreshes |
| `--window` | 50 | Rolling average window size |

## Constraints

- No new dependencies (matplotlib + pandas already installed)
- Read-only access to CSV — never writes, never imports training code
- Gracefully handles missing file (waits and retries)
- Gracefully handles CSV with <2 data rows (shows empty plots)
