"""
Training dashboard - Web UI for real-time monitoring of A2C training.

Reads TensorBoard event files and serves interactive plots via Flask/Plotly.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import flask
import click
from flask import Flask, render_template_string, jsonify
import numpy as np

try:
    from tensorboard.compat.proto import event_pb2
except ImportError:
    event_pb2 = None


app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True


def read_tensorboard_events(logs_dir: str | None = None) -> dict:
    """Read scalar events from TensorBoard event files.
    
    Returns dict of {tag: [(step, value), ...]}
    """
    if logs_dir is None:
        logs_dir = "rl/logs_v2"
    
    if not event_pb2:
        return {}
    
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return {}
    
    metrics = defaultdict(list)
    
    # Find all event files
    for event_file in sorted(logs_path.glob("events.out.tfevents.*")):
        try:
            for event_raw in event_pb2.Event.FromString.parser.Deserialize(
                open(event_file, "rb").read(), event_pb2.Event()
            ):
                if event_raw.HasField("summary"):
                    for value in event_raw.summary.value:
                        if value.HasField("simple_value"):
                            metrics[value.tag].append(
                                (event_raw.step, value.simple_value)
                            )
        except Exception:
            pass  # Skip corrupted files
    
    # Sort by step
    for tag in metrics:
        metrics[tag].sort(key=lambda x: x[0])
    
    return metrics


def format_metrics(metrics: dict) -> dict:
    """Format metrics for JSON response."""
    formatted = {}
    for tag, values in metrics.items():
        if values:
            steps, vals = zip(*values)
            formatted[tag] = {
                "steps": list(steps),
                "values": list(vals),
                "latest": vals[-1],
                "latest_step": steps[-1],
            }
    return formatted


@app.route("/")
def index():
    """Main dashboard page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>A2C Training Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }
            .header {
                max-width: 1400px;
                margin: 0 auto 20px;
            }
            .header h1 {
                margin: 0 0 10px;
                color: #333;
            }
            .status {
                display: flex;
                gap: 30px;
                font-size: 14px;
                color: #666;
            }
            .plots-container {
                max-width: 1400px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .plot-box {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .plot-box h3 {
                margin: 0 0 10px;
                font-size: 14px;
                color: #333;
            }
            .plot-container {
                width: 100%;
                height: 300px;
            }
            @media (max-width: 1024px) {
                .plots-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 A2C Training Dashboard</h1>
            <div class="status">
                <div>📊 Latest Step: <span id="step">-</span></div>
                <div>⏱️ Last Update: <span id="time">-</span></div>
                <div>🎯 Reward: <span id="reward">-</span></div>
            </div>
        </div>
        
        <div class="plots-container">
            <div class="plot-box">
                <h3>Episode Reward</h3>
                <div id="reward-plot" class="plot-container"></div>
            </div>
            <div class="plot-box">
                <h3>Policy Loss</h3>
                <div id="policy-loss-plot" class="plot-container"></div>
            </div>
            <div class="plot-box">
                <h3>Value Loss</h3>
                <div id="value-loss-plot" class="plot-container"></div>
            </div>
            <div class="plot-box">
                <h3>Steps per Second</h3>
                <div id="steps-sec-plot" class="plot-container"></div>
            </div>
        </div>

        <script>
            async function updateDashboard() {
                try {
                    const response = await fetch("/api/metrics");
                    const metrics = await response.json();
                    
                    if (!metrics.data || Object.keys(metrics.data).length === 0) {
                        console.log("No metrics yet...");
                        return;
                    }
                    
                    // Update header
                    const data = metrics.data;
                    if (data["episode_reward/mean"]) {
                        const reward = data["episode_reward/mean"].values;
                        document.getElementById("reward").textContent = reward[reward.length - 1].toFixed(0);
                    }
                    if (data["episode_reward/mean"]) {
                        const step = data["episode_reward/mean"].latest_step;
                        document.getElementById("step").textContent = step.toLocaleString();
                    }
                    document.getElementById("time").textContent = new Date().toLocaleTimeString();
                    
                    // Plot reward
                    if (data["episode_reward/mean"]) {
                        Plotly.newPlot("reward-plot", [{
                            x: data["episode_reward/mean"].steps,
                            y: data["episode_reward/mean"].values,
                            type: "scatter",
                            mode: "lines",
                            name: "Reward"
                        }], {
                            title: "",
                            xaxis: {title: "Step"},
                            yaxis: {title: "Reward"},
                            margin: {l: 40, r: 30, t: 20, b: 40},
                            hovermode: "x"
                        }, {responsive: true});
                    }
                    
                    // Plot policy loss
                    if (data["loss/policy"]) {
                        Plotly.newPlot("policy-loss-plot", [{
                            x: data["loss/policy"].steps,
                            y: data["loss/policy"].values,
                            type: "scatter",
                            mode: "lines",
                            name: "Policy Loss"
                        }], {
                            title: "",
                            xaxis: {title: "Step"},
                            yaxis: {title: "Loss"},
                            margin: {l: 40, r: 30, t: 20, b: 40},
                            hovermode: "x"
                        }, {responsive: true});
                    }
                    
                    // Plot value loss
                    if (data["loss/value"]) {
                        Plotly.newPlot("value-loss-plot", [{
                            x: data["loss/value"].steps,
                            y: data["loss/value"].values,
                            type: "scatter",
                            mode: "lines",
                            name: "Value Loss"
                        }], {
                            title: "",
                            xaxis: {title: "Step"},
                            yaxis: {title: "Loss"},
                            margin: {l: 40, r: 30, t: 20, b: 40},
                            hovermode: "x"
                        }, {responsive: true});
                    }
                    
                    // Plot steps per second
                    if (data["performance/steps_per_sec"]) {
                        Plotly.newPlot("steps-sec-plot", [{
                            x: data["performance/steps_per_sec"].steps,
                            y: data["performance/steps_per_sec"].values,
                            type: "scatter",
                            mode: "lines",
                            name: "Steps/sec"
                        }], {
                            title: "",
                            xaxis: {title: "Step"},
                            yaxis: {title: "Steps/sec"},
                            margin: {l: 40, r: 30, t: 20, b: 40},
                            hovermode: "x"
                        }, {responsive: true});
                    }
                    
                } catch (error) {
                    console.error("Dashboard error:", error);
                }
            }

            // Update every 2 seconds
            updateDashboard();
            setInterval(updateDashboard, 2000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/api/metrics")
def api_metrics():
    """API endpoint returning current metrics as JSON."""
    logs_dir = os.getenv("TRAINING_LOGS_DIR", "rl/logs_v2")
    metrics = read_tensorboard_events(logs_dir)
    formatted = format_metrics(metrics)
    return jsonify({"data": formatted, "timestamp": datetime.now().isoformat()})


def run(host: str = "localhost", port: int = 5000, debug: bool = False) -> None:
    """Start the dashboard server."""
    print(f"🚀 Dashboard starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


@app.cli.command()
@click.option('--host', default='localhost')
@click.option('--port', type=int, default=5000)
def run_cmd(host, port):
    """Run the dashboard server."""
    run(host=host, port=port)


if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    run(host=host, port=port, debug=False)
