"""
Standardized Metrics Logger for Scientific Reproducibility.
Strictly adheres to repository logging rules defined in GEMINI.md.
"""

import os
import sys
import json
import time
import platform
import subprocess
from typing import Dict, Any, List, Optional
import numpy as np


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, or 'unknown' if git fails."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_hardware_info() -> Dict[str, Any]:
    """Captures CPU, OS, and memory information."""
    info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        # Fallback if psutil is not available
        info["ram_total_gb"] = "unknown"
        info["cpu_count_logical"] = os.cpu_count()
        
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
        
    return info


class BenchmarkTracker:
    """
    Tracks and records experiment metrics:
    - wall_clock_time
    - function_evaluation_time
    - internal_overhead_time
    - total_evaluations
    - convergence_speed (evaluations or iterations to reach 90% of final accuracy)
    """
    def __init__(self, experiment_name: str, config: Dict[str, Any], seed: int):
        self.experiment_name = experiment_name
        self.config = config
        self.seed = seed
        self.commit_hash = get_git_commit_hash()
        self.hardware_info = get_hardware_info()
        
        self.t_start: float = 0.0
        self.t_end: float = 0.0
        self.function_eval_time: float = 0.0
        self.total_evaluations: int = 0
        self.history: List[Dict[str, Any]] = []

    def start(self):
        """Starts timer."""
        self.t_start = time.perf_counter()

    def record_step(self, iteration: int, eval_time: float, evaluations: int, current_acc: float):
        """Records an evaluation step."""
        self.function_eval_time += eval_time
        self.total_evaluations += evaluations
        self.history.append({
            "iteration": iteration,
            "eval_time": eval_time,
            "cumulative_eval_time": self.function_eval_time,
            "evaluations": evaluations,
            "cumulative_evaluations": self.total_evaluations,
            "accuracy": current_acc
        })

    def finish(self, final_objective: float, extra_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculates final metrics and returns dictionary."""
        self.t_end = time.perf_counter()
        wall_clock = self.t_end - self.t_start
        internal_overhead = max(0.0, wall_clock - self.function_eval_time)
        
        # Calculate convergence speed (iterations to reach 90% of final_objective)
        threshold_90 = 0.90 * final_objective
        conv_speed = None
        for step in self.history:
            if step["accuracy"] >= threshold_90:
                conv_speed = step["iteration"]
                break
        if conv_speed is None:
            conv_speed = len(self.history)
            
        result = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "commit_hash": self.commit_hash,
            "hardware_info": self.hardware_info,
            "full_config": self.config,
            "final_objective": float(final_objective),
            "total_evaluations": int(self.total_evaluations),
            "wall_clock_time": float(round(wall_clock, 4)),
            "function_evaluation_time": float(round(self.function_eval_time, 4)),
            "internal_overhead_time": float(round(internal_overhead, 4)),
            "convergence_speed": int(conv_speed),
            "trajectory": self.history
        }
        if extra_metrics:
            result.update(extra_metrics)
            
        return result

    def save_raw(self, output_dir: str = "results/raw", result_data: Optional[Dict[str, Any]] = None) -> str:
        """Saves individual run metrics to results/raw/."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.experiment_name}_seed{self.seed}.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
        return path


def aggregate_and_save_summary(
    experiment_name: str,
    raw_results: List[Dict[str, Any]],
    output_dir: str = "results/summary"
) -> Dict[str, Any]:
    """
    Aggregates multi-seed experiment results and writes summary to results/summary/.
    Computes mean ± std for all numerical metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_seeds = len(raw_results)
    
    objectives = [r["final_objective"] for r in raw_results]
    wall_clocks = [r["wall_clock_time"] for r in raw_results]
    eval_times = [r["function_evaluation_time"] for r in raw_results]
    overheads = [r["internal_overhead_time"] for r in raw_results]
    evaluations = [r["total_evaluations"] for r in raw_results]
    conv_speeds = [r["convergence_speed"] for r in raw_results]
    
    summary = {
        "experiment_name": experiment_name,
        "num_seeds": num_seeds,
        "commit_hash": raw_results[0].get("commit_hash", "unknown"),
        "hardware_info": raw_results[0].get("hardware_info", {}),
        "full_config": raw_results[0].get("full_config", {}),
        "metrics": {
            "final_objective": {
                "mean": float(np.mean(objectives)),
                "std": float(np.std(objectives)),
                "min": float(np.min(objectives)),
                "max": float(np.max(objectives)),
            },
            "wall_clock_time": {
                "mean": float(np.mean(wall_clocks)),
                "std": float(np.std(wall_clocks)),
            },
            "function_evaluation_time": {
                "mean": float(np.mean(eval_times)),
                "std": float(np.std(eval_times)),
            },
            "internal_overhead_time": {
                "mean": float(np.mean(overheads)),
                "std": float(np.std(overheads)),
            },
            "total_evaluations": {
                "mean": float(np.mean(evaluations)),
                "std": float(np.std(evaluations)),
            },
            "convergence_speed": {
                "mean": float(np.mean(conv_speeds)),
                "std": float(np.std(conv_speeds)),
            }
        }
    }
    
    out_path = os.path.join(output_dir, f"{experiment_name}_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return summary
